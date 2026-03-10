"""Learning M1: factor IC computation aligned to the screened trading universe.

Reads daily_prices from SQLite, recomputes the same technical features used by
the strategy, and evaluates Spearman IC versus H-period forward returns.

Results written to:
  - factor_signals: per-factor z-scores per (asof, ticker)
  - factor_registry: EWMA IC weights per factor
  - learning_audit: immutable log of every IC update / shadow evaluation

Guardrails:
  - single-period cross-section must have at least min_cross_section_n names
  - time-series t-stat = |mean_IC| / (std_IC / sqrt(n_periods)) must be >= 1.5
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from screener import ScreenConfig, screen
from trade_schema import connect, ensure_learning_tables, ensure_trade_tables, save_screening_history


FACTOR_NAMES: List[str] = [
    "ret1", "ret5", "ret20", "ret60",
    "vol20", "vol60",
    "ma_gap", "z_20", "rsi14", "slope60",
    "mom_12_1", "high52w", "vol_adj_mom20", "mom_consist", "vol_z",
]

HALF_LIFE_DAYS: int = 60
TSTAT_GUARD: float = 1.5
MIN_CROSS_SECTION_N: int = 25


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta).clip(lower=0).rolling(n).mean()
    rs = up / (dn + 1e-9)
    return 100.0 - 100.0 / (1.0 + rs)


def _log_slope(series: pd.Series, n: int = 60) -> pd.Series:
    log_s = np.log(series.clip(lower=1e-6))

    def _fit(x: pd.Series) -> float:
        if x.isna().any():
            return float("nan")
        t = np.arange(len(x), dtype=float)
        return float(np.polyfit(t, x.values, 1)[0])

    return log_s.rolling(n).apply(_fit, raw=False)


def compute_features(px: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute all 15 features on a (date x ticker) price DataFrame."""
    ret = px.pct_change()
    feats: Dict[str, pd.DataFrame] = {}

    feats["ret1"] = ret
    feats["ret5"] = px.pct_change(5)
    feats["ret20"] = px.pct_change(20)
    feats["ret60"] = px.pct_change(60)
    feats["vol20"] = ret.rolling(20).std()
    feats["vol60"] = ret.rolling(60).std()
    ma50 = px.rolling(50).mean()
    ma200 = px.rolling(200).mean()
    feats["ma_gap"] = (ma50 - ma200) / (ma200 + 1e-9)
    mu20 = px.rolling(20).mean()
    sd20 = px.rolling(20).std()
    feats["z_20"] = (px - mu20) / (sd20 + 1e-9)
    feats["rsi14"] = px.apply(_rsi)
    feats["slope60"] = px.apply(_log_slope)

    feats["mom_12_1"] = px.pct_change(252) - px.pct_change(21)
    feats["high52w"] = (px / px.rolling(252).max().clip(lower=1e-6)) - 1.0
    feats["vol_adj_mom20"] = feats["ret20"] / (feats["vol20"] + 1e-9)
    feats["mom_consist"] = ret.rolling(63).apply(lambda x: float((x > 0).mean()), raw=True)
    feats["vol_z"] = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    return feats


def ewma_update(old: float, new: float, n_obs: int, half_life: float = HALF_LIFE_DAYS) -> float:
    if n_obs < 5:
        alpha = 1.0 / (n_obs + 1)
    else:
        alpha = 1.0 - math.exp(-math.log(2) / half_life)
    return (1.0 - alpha) * old + alpha * new


def load_screening_universe(conn, lookback_periods: int) -> tuple[dict[str, list[str]], list[str]]:
    rows = conn.execute(
        """
        SELECT asof, symbol
        FROM screening_history
        WHERE selected = 1
        ORDER BY asof ASC, rank ASC, symbol ASC
        """
    ).fetchall()
    if not rows:
        return {}, []

    universe_by_asof: dict[str, list[str]] = {}
    for asof, symbol in rows:
        universe_by_asof.setdefault(str(asof), []).append(str(symbol))

    eval_asofs = sorted(universe_by_asof)[-lookback_periods:]
    trimmed = {asof: universe_by_asof[asof] for asof in eval_asofs}
    return trimmed, eval_asofs


def latest_screen_config(conn) -> ScreenConfig:
    row = conn.execute(
        """
        SELECT filters_json
        FROM screening_history
        ORDER BY asof DESC, rank ASC
        LIMIT 1
        """
    ).fetchone()
    filters = json.loads(row[0]) if row and row[0] else {}
    return ScreenConfig(
        lookback_days=int(filters.get("lookback_days", 252)),
        min_adv=float(filters.get("min_adv", 20_000_000)),
        max_missing=float(filters.get("max_missing", 0.02)),
        min_vol=float(filters.get("min_vol", 0.005)),
        max_vol=float(filters.get("max_vol", 0.06)),
        top_k=int(filters.get("top_k", 50)),
        lot_size_default=int(filters.get("lot_size_default", 100)),
        max_cost_per_lot=float(filters.get("max_cost_per_lot", 150_000)),
    )


def backfill_screening_history(
    conn,
    db_path: str,
    eval_dates: list[pd.Timestamp],
    cfg: ScreenConfig,
) -> int:
    missing_asofs = []
    for dt in eval_dates:
        asof = dt.strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT 1 FROM screening_history WHERE asof=? LIMIT 1",
            (asof,),
        ).fetchone()
        if row is None:
            missing_asofs.append(asof)

    inserted = 0
    for asof in missing_asofs:
        payload = screen(
            db_path=db_path,
            symbols=None,
            asof=asof,
            out_path=None,
            cfg=cfg,
            write_db=False,
        )
        inserted += save_screening_history(conn, payload)
    return inserted


def load_logged_factor_scores(conn, asofs: list[str]) -> dict[str, pd.DataFrame]:
    if not asofs:
        return {}
    placeholders = ",".join("?" for _ in asofs)
    rows = pd.read_sql_query(
        f"""
        SELECT asof, symbol, factor_name, raw_score
        FROM factor_signals
        WHERE asof IN ({placeholders})
        """,
        conn,
        params=asofs,
    )
    if rows.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for factor_name, grp in rows.groupby("factor_name"):
        frame = grp.pivot(index="asof", columns="symbol", values="raw_score").sort_index()
        frame.index = pd.to_datetime(frame.index)
        out[str(factor_name)] = frame
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute factor IC and update factor_registry")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--H", type=int, default=20, help="Holding period in trading days")
    ap.add_argument("--rebalance_every", type=int, default=20, help="Fallback rebalance frequency")
    ap.add_argument("--lookback_periods", type=int, default=60, help="Recent rebalance periods to evaluate")
    ap.add_argument(
        "--min_cross_section_n",
        type=int,
        default=MIN_CROSS_SECTION_N,
        help="Minimum screened names required on a rebalance date to count IC",
    )
    ap.add_argument("--shadow", action="store_true", help="Compute IC but do not update live weights")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    ensure_learning_tables(conn)

    print("Loading daily_prices from DB...")
    raw = pd.read_sql_query(
        "SELECT date, symbol, close FROM daily_prices ORDER BY date, symbol",
        conn,
    )
    if raw.empty:
        print("No data in daily_prices. Run db_update.py first.")
        conn.close()
        return

    raw["date"] = pd.to_datetime(raw["date"])
    px = raw.pivot(index="date", columns="symbol", values="close").sort_index()
    tickers = px.columns.tolist()
    print(f"Loaded {len(px)} dates x {len(tickers)} tickers")

    print("Computing features...")
    feats = compute_features(px)
    fwd_ret = px.pct_change(args.H).shift(-args.H)

    min_history = 210
    fallback_eval_dates = px.index[min_history::args.rebalance_every]
    fallback_eval_dates = fallback_eval_dates[fallback_eval_dates <= px.index[-args.H - 1]]
    fallback_eval_dates = list(fallback_eval_dates[-args.lookback_periods:])

    if fallback_eval_dates:
        screen_cfg = latest_screen_config(conn)
        backfilled = backfill_screening_history(conn, args.db, fallback_eval_dates, screen_cfg)
        if backfilled:
            print(f"Backfilled screening_history: {backfilled} rows across missing rebalance dates")

    universe_by_asof, eval_asofs = load_screening_universe(conn, args.lookback_periods)
    if universe_by_asof:
        eval_dates = [pd.Timestamp(asof) for asof in eval_asofs if pd.Timestamp(asof) in px.index]
        eval_dates = [dt for dt in eval_dates if dt <= px.index[-args.H - 1]]
        print(
            f"Evaluating IC on {len(eval_dates)} screened rebalance dates "
            f"(H={args.H}, shadow={args.shadow}, min_cross_section_n={args.min_cross_section_n})"
        )
    else:
        eval_dates = fallback_eval_dates
        print("screening_history empty; falling back to full-universe evaluation")
        print(
            f"Evaluating IC on {len(eval_dates)} rebalance dates "
            f"(H={args.H}, shadow={args.shadow}, min_cross_section_n={args.min_cross_section_n})"
        )

    logged_scores = load_logged_factor_scores(conn, [dt.strftime("%Y-%m-%d") for dt in eval_dates])

    reg_rows = conn.execute(
        "SELECT factor_name, ic_mean, ic_std, icir, weight, n_observations FROM factor_registry"
    ).fetchall()
    registry: Dict[str, dict] = {
        row[0]: {"ic_mean": row[1], "ic_std": row[2], "icir": row[3], "weight": row[4], "n_obs": row[5]}
        for row in reg_rows
    }

    ic_series: Dict[str, List[float]] = {factor: [] for factor in FACTOR_NAMES}
    ic_cross_counts: Dict[str, List[int]] = {factor: [] for factor in FACTOR_NAMES}
    skipped_by_factor: Dict[str, int] = {factor: 0 for factor in FACTOR_NAMES}
    signal_rows: List[dict] = []
    used_dates = 0

    for dt in eval_dates:
        if dt not in fwd_ret.index:
            continue

        asof = dt.strftime("%Y-%m-%d")
        selected_symbols = universe_by_asof.get(asof, tickers)
        available_symbols = [symbol for symbol in selected_symbols if symbol in tickers]
        if len(available_symbols) < args.min_cross_section_n:
            continue

        used_dates += 1
        fwd = fwd_ret.loc[dt, available_symbols]

        for factor in FACTOR_NAMES:
            logged = logged_scores.get(factor)
            if logged is not None and dt in logged.index:
                f_vals = logged.loc[dt].reindex(available_symbols)
            else:
                f_vals = feats[factor].loc[dt, available_symbols]
            combined = pd.concat([f_vals, fwd], axis=1).dropna()
            cross_n = len(combined)
            if cross_n >= args.min_cross_section_n:
                rho, _ = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
                if np.isfinite(rho):
                    ic_series[factor].append(float(rho))
                    ic_cross_counts[factor].append(cross_n)
            else:
                skipped_by_factor[factor] += 1

            valid = f_vals.dropna()
            if len(valid) > 1:
                mu = valid.mean()
                sigma = valid.std()
                z_vals = (f_vals - mu) / (sigma + 1e-9) if sigma > 1e-9 else f_vals * 0.0
            else:
                z_vals = f_vals * 0.0

            for symbol in available_symbols:
                raw_score = float(f_vals.get(symbol, float("nan")))
                z_score = float(z_vals.get(symbol, float("nan")))
                if np.isfinite(raw_score) or np.isfinite(z_score):
                    signal_rows.append(
                        {
                            "asof": asof,
                            "symbol": symbol,
                            "factor_name": factor,
                            "raw_score": raw_score if np.isfinite(raw_score) else None,
                            "z_score": z_score if np.isfinite(z_score) else None,
                        }
                    )

    if signal_rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO factor_signals
            (asof, symbol, factor_name, raw_score, z_score)
            VALUES (:asof, :symbol, :factor_name, :raw_score, :z_score)
            """,
            signal_rows,
        )
        conn.commit()
        print(f"Saved {len(signal_rows):,} rows to factor_signals")
    print(f"Used {used_dates} rebalance dates after screened-universe gating")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    audit_rows: List[dict] = []

    print()
    print(
        f"{'Factor':12s}  {'IC_mean':>8s}  {'IC_std':>7s}  {'ICIR':>6s}  "
        f"{'t-stat':>7s}  {'n':>4s}  {'xs_n':>6s}  {'Guard':>6s}  {'Weight':>7s}"
    )
    print("-" * 86)

    for factor in FACTOR_NAMES:
        ics = ic_series[factor]
        if not ics:
            print(f"{factor:12s}  (no data)")
            continue

        ic_arr = np.array(ics)
        batch_ic_mean = float(ic_arr.mean())
        batch_ic_std = float(ic_arr.std()) if len(ic_arr) > 1 else 0.01
        se = batch_ic_std / math.sqrt(len(ic_arr))
        t_stat = abs(batch_ic_mean) / max(se, 1e-9)
        passes = t_stat >= TSTAT_GUARD
        avg_cross_n = float(np.mean(ic_cross_counts[factor])) if ic_cross_counts[factor] else 0.0

        old = registry.get(
            factor,
            {"ic_mean": 0.0, "ic_std": 1.0, "icir": 0.0, "weight": 1.0, "n_obs": 0},
        )
        n_obs = int(old["n_obs"])
        upd_ic_mean = ewma_update(old["ic_mean"], batch_ic_mean, n_obs)
        upd_ic_std = max(0.01, ewma_update(old["ic_std"], batch_ic_std, n_obs))
        upd_icir = upd_ic_mean / upd_ic_std
        upd_weight = max(0.0, upd_icir)
        new_n_obs = n_obs + len(ics)

        event_type = "IC_UPDATE" if (passes and not args.shadow) else "IC_SHADOW"
        audit_rows.append(
            {
                "asof": now_str[:10],
                "event_type": event_type,
                "factor_name": factor,
                "old_value": old["ic_mean"],
                "new_value": batch_ic_mean,
                "ic_value": upd_ic_mean,
                "notes": (
                    f"t={t_stat:.2f} n={len(ics)} xs_n={avg_cross_n:.1f} "
                    f"min_xs={args.min_cross_section_n} skipped={skipped_by_factor[factor]} "
                    f"icir={upd_icir:.3f} guard={'PASS' if passes else 'FAIL'}"
                    f"{' [SHADOW]' if args.shadow else ''}"
                ),
            }
        )

        guard_str = "PASS" if passes else "FAIL"
        print(
            f"{factor:12s}  {batch_ic_mean:+8.4f}  {batch_ic_std:7.4f}  "
            f"{upd_icir:+6.3f}  {t_stat:7.2f}  {len(ics):4d}  {avg_cross_n:6.1f}  "
            f"{guard_str}  {upd_weight:7.4f}"
        )

        if not args.shadow and passes:
            conn.execute(
                """
                INSERT OR REPLACE INTO factor_registry
                (factor_name, ic_mean, ic_std, icir, weight, n_observations, last_updated, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (factor, upd_ic_mean, upd_ic_std, upd_icir, upd_weight, new_n_obs, now_str),
            )
        elif factor not in registry:
            conn.execute(
                """
                INSERT OR IGNORE INTO factor_registry
                (factor_name, ic_mean, ic_std, icir, weight, n_observations, last_updated, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    factor,
                    batch_ic_mean,
                    batch_ic_std,
                    batch_ic_mean / max(batch_ic_std, 0.01),
                    max(0.0, batch_ic_mean / max(batch_ic_std, 0.01)),
                    len(ics),
                    now_str,
                ),
            )

    if audit_rows:
        conn.executemany(
            """
            INSERT INTO learning_audit
            (asof, event_type, factor_name, old_value, new_value, ic_value, notes)
            VALUES (:asof, :event_type, :factor_name, :old_value, :new_value, :ic_value, :notes)
            """,
            audit_rows,
        )
    conn.commit()
    print()
    print(f"Learning audit: {len(audit_rows)} entries written")
    if args.shadow:
        print("Shadow mode: factor_registry weights NOT updated (run without --shadow to apply)")

    conn.close()


if __name__ == "__main__":
    main()
