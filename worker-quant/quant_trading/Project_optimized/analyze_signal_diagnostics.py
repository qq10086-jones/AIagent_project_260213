"""Production-signal diagnostics for IC / t-stat / subgroup behavior.

Uses factor_signals logged by ss7_sqlite_news_overlay.py, joins forward returns
from daily_prices, and evaluates both the composite signal (pred_return) and
raw factors on the actual production universe.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from trade_schema import connect, ensure_learning_tables, ensure_trade_tables


def load_signal_panel(conn, H: int) -> pd.DataFrame:
    prod_dates = pd.read_sql_query(
        """
        SELECT DISTINCT asof
        FROM factor_signals
        WHERE pred_return IS NOT NULL
        ORDER BY asof
        """,
        conn,
    )["asof"].tolist()
    if not prod_dates:
        raise RuntimeError("No production factor_signals with pred_return found. Run run_pipeline.py first.")

    placeholders = ",".join("?" for _ in prod_dates)
    fs = pd.read_sql_query(
        f"""
        SELECT asof, symbol, factor_name, raw_score, z_score, pred_return
        FROM factor_signals
        WHERE asof IN ({placeholders})
        """,
        conn,
        params=prod_dates,
    )
    if fs.empty:
        raise RuntimeError("factor_signals is empty for production dates.")

    raw_wide = fs.pivot_table(
        index=["asof", "symbol"],
        columns="factor_name",
        values="raw_score",
        aggfunc="last",
    )
    pred = fs.dropna(subset=["pred_return"]).groupby(["asof", "symbol"], as_index=True)["pred_return"].last()
    panel = raw_wide.join(pred.rename("pred_return"), how="left").reset_index()

    prices = pd.read_sql_query(
        "SELECT date, symbol, close FROM daily_prices ORDER BY date, symbol",
        conn,
    )
    prices["date"] = pd.to_datetime(prices["date"])
    px = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    fwd_ret = px.pct_change(H).shift(-H).stack().rename("fwd_ret").reset_index()
    fwd_ret = fwd_ret.rename(columns={"date": "asof"})
    fwd_ret["asof"] = fwd_ret["asof"].dt.strftime("%Y-%m-%d")

    sector = pd.read_sql_query("SELECT symbol, COALESCE(sector, 'Unknown') AS sector FROM tickers", conn)
    panel = panel.merge(fwd_ret, on=["asof", "symbol"], how="left")
    panel = panel.merge(sector, on="symbol", how="left")
    panel["sector"] = panel["sector"].fillna("Unknown")
    return panel


def add_regime_buckets(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for factor_name, bucket_name in [("vol20", "vol_bucket"), ("high52w", "high52w_bucket")]:
        if factor_name not in out.columns:
            continue

        def _bucket(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if len(valid) < 6:
                return pd.Series(index=s.index, dtype=object)
            try:
                ranked = valid.rank(method="first")
                b = pd.qcut(ranked, 3, labels=["low", "mid", "high"])
                result = pd.Series(index=s.index, dtype=object)
                result.loc[valid.index] = b.astype(str)
                return result
            except Exception:
                return pd.Series(index=s.index, dtype=object)

        out[bucket_name] = out.groupby("asof")[factor_name].transform(_bucket)
    return out


def summarize_ic(
    panel: pd.DataFrame,
    signal_col: str,
    min_xs_n: int,
    group_col: str | None = None,
) -> pd.DataFrame:
    group_values = [("__ALL__", panel)] if group_col is None else list(panel.groupby(group_col, dropna=True))
    rows = []
    for group_value, grp in group_values:
        date_rows = []
        for asof, day in grp.groupby("asof"):
            subset = day[[signal_col, "fwd_ret"]].dropna()
            if len(subset) < min_xs_n:
                continue
            rho, _ = stats.spearmanr(subset[signal_col], subset["fwd_ret"])
            if np.isfinite(rho):
                date_rows.append((asof, float(rho), len(subset)))
        if not date_rows:
            continue
        ic = np.array([x[1] for x in date_rows], dtype=float)
        xs_counts = np.array([x[2] for x in date_rows], dtype=float)
        ic_mean = float(ic.mean())
        ic_std = float(ic.std()) if len(ic) > 1 else 0.0
        t_stat = abs(ic_mean) / max(ic_std / math.sqrt(len(ic)), 1e-9)
        rows.append(
            {
                "signal": signal_col,
                "group": str(group_value) if group_col is not None else "all",
                "group_col": group_col or "overall",
                "periods": int(len(ic)),
                "avg_xs_n": float(xs_counts.mean()),
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "t_stat": t_stat,
            }
        )
    return pd.DataFrame(rows)


def add_shadow_composites(panel: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    eligible = overall[(overall["signal"] != "pred_return") & (overall["ic_mean"] > 0)].copy()
    if eligible.empty:
        return out

    top = eligible.sort_values(["t_stat", "ic_mean"], ascending=[False, False]).head(5)
    top_signals = [s for s in top["signal"].tolist() if s in out.columns]
    if not top_signals:
        return out

    weight_map = dict(zip(top["signal"], top["t_stat"]))

    def _build(group: pd.DataFrame, weighted: bool) -> pd.Series:
        z_cols = []
        weights = []
        for signal in top_signals:
            s = group[signal]
            z = (s - s.mean()) / (s.std() + 1e-12)
            z_cols.append(z)
            weights.append(weight_map[signal] if weighted else 1.0)
        mat = pd.concat(z_cols, axis=1)
        w = np.asarray(weights, dtype=float)
        score = mat.to_numpy(dtype=float) @ w
        return pd.Series(score, index=group.index)

    out["shadow_eq_composite"] = out.groupby("asof", group_keys=False).apply(lambda g: _build(g, weighted=False))
    out["shadow_ic_composite"] = out.groupby("asof", group_keys=False).apply(lambda g: _build(g, weighted=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze production signal diagnostics")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--min_xs_n", type=int, default=15)
    ap.add_argument("--group_min_xs_n", type=int, default=5)
    ap.add_argument("--out_dir", default="reports/signal_diagnostics")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    ensure_trade_tables(conn)
    ensure_learning_tables(conn)

    panel = add_regime_buckets(load_signal_panel(conn, args.H))
    conn.close()

    signal_cols = ["pred_return"] + [c for c in panel.columns if c not in {
        "asof", "symbol", "fwd_ret", "sector", "vol_bucket", "high52w_bucket"
    } and c != "pred_return"]

    overall_frames = []
    for signal in signal_cols:
        if signal in panel.columns:
            overall_frames.append(summarize_ic(panel, signal, args.min_xs_n))
    overall = pd.concat(overall_frames, ignore_index=True).sort_values(
        ["t_stat", "ic_mean"], ascending=[False, False]
    )
    panel = add_shadow_composites(panel, overall)
    for signal in ["shadow_eq_composite", "shadow_ic_composite"]:
        if signal in panel.columns:
            overall = pd.concat([overall, summarize_ic(panel, signal, args.min_xs_n)], ignore_index=True)
    overall = overall.sort_values(["t_stat", "ic_mean"], ascending=[False, False])
    overall.to_csv(out_dir / "overall_ic_summary.csv", index=False)

    focus_signals = [
        s for s in [
            "pred_return",
            "shadow_eq_composite",
            "shadow_ic_composite",
            "vol_adj_mom20",
            "high52w",
            "mom_12_1",
            "rsi14",
        ] if s in panel.columns
    ]
    grouped_frames = []
    for signal in focus_signals:
        grouped_frames.append(summarize_ic(panel, signal, args.group_min_xs_n, group_col="sector"))
        if "vol_bucket" in panel.columns:
            grouped_frames.append(summarize_ic(panel, signal, args.group_min_xs_n, group_col="vol_bucket"))
        if "high52w_bucket" in panel.columns:
            grouped_frames.append(summarize_ic(panel, signal, args.group_min_xs_n, group_col="high52w_bucket"))

    grouped = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else pd.DataFrame()
    if not grouped.empty:
        grouped = grouped.sort_values(["signal", "group_col", "t_stat"], ascending=[True, True, False])
        grouped.to_csv(out_dir / "grouped_ic_summary.csv", index=False)

    prod_dates = panel["asof"].nunique()
    print(f"Production dates: {prod_dates}")
    print(f"Universe size range: {int(panel.groupby('asof')['symbol'].nunique().min())} - {int(panel.groupby('asof')['symbol'].nunique().max())}")
    print()
    print("Top overall signals by t-stat:")
    print(overall[["signal", "periods", "avg_xs_n", "ic_mean", "t_stat"]].head(10).to_string(index=False))
    if not grouped.empty:
        print()
        print("Grouped diagnostics written to:")
        print(out_dir / "grouped_ic_summary.csv")
    print("Overall diagnostics written to:")
    print(out_dir / "overall_ic_summary.csv")


if __name__ == "__main__":
    main()
