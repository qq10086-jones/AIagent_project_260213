"""walk_forward_runner.py — v0 minimal OOS research harness for sprint factors.

Purpose (per 2026-04-14 senior-quant review):
    Answer the single binary question — does the current sprint 3-factor
    stack produce a positive OOS edge after lot/fee/slippage, or not?

Design
------
- Rebalance monthly (first trading day of each calendar month).
- For each rebalance date `d`:
    * Compute factors on closes strictly up to `d-1` (no lookahead).
    * Rank cross-sectionally, z-score, equal-weight average → composite.
    * Pick top-K (default 5) as target portfolio.
    * Execute at `d+1` CLOSE (proxy for T+1 open; open column is sparse).
    * Deduct fee_bps + slippage_bps per side.
- Hold until next rebalance.
- Benchmarks: 1321.T (TOPIX proxy), universe equal-weight.

Scope (v0)
----------
- No lot rounding (fractional weights). Upgrade later.
- No purge / embargo (first cut).
- Factors: the three production sprint factors (mom_consist, high52w,
  vol_z) computed from daily_prices only.
- Liquidity filter: 60d ADV > min_adv_floor.

Output
------
JSON report at reports/walk_forward_{tag}.json with per-month returns,
cumulative, IR, Sharpe, maxDD vs 2 benchmarks.

This is a SKELETON. Do not use for sizing decisions until it has been
extended with lot-rounding, T+1 open, sqrt slippage, and block-bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── Data loading ─────────────────────────────────────────────────────

def load_price_panel(
    db_path: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT symbol, date, open, high, low, close, volume FROM daily_prices "
            "WHERE date BETWEEN ? AND ? ORDER BY date, symbol",
            conn, params=(start, end),
        )
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# JP ETF/REIT symbols typically fall in 13XX / 14XX / 15XX / 17XX / 18XX /
# 2500-2599 REIT. Excluding them keeps the universe stock-only so the
# equal-weight benchmark is not dominated by index-replicating products.
_ETF_REIT_PATTERNS = (
    "13", "14", "15", "17", "18",  # ETF prefixes (not exhaustive)
)


def is_common_stock(sym: str) -> bool:
    if not sym.endswith(".T"):
        return True
    code = sym.split(".")[0]
    if len(code) != 4 or not code.isdigit():
        return True
    if code.startswith(_ETF_REIT_PATTERNS):
        return False
    # 2500-2599 → JREIT
    if 2500 <= int(code) <= 2599:
        return False
    return True


def pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="symbol", values=col).sort_index()


# ── Sprint factors (matching sprint_signal.py semantics) ─────────────

def factor_mom_consist(close: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Fraction of up-days in the past `window` sessions.

    Matches production (`compute_price_features.py`): window=63 (≈3M).
    """
    ret = close.pct_change(fill_method=None)
    up = (ret > 0).astype(float)
    return up.rolling(window, min_periods=window).mean()


def factor_high52w(close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """close / rolling_max - 1. Closer to 0 = near 52w high."""
    rmax = close.rolling(window, min_periods=window // 2).max()
    return close / rmax - 1.0


def factor_vol_z(vol: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Log-volume z-score vs trailing `window` mean/std.

    Matches production: z of log(volume) over 60d, not raw volume over 20d.
    """
    v = vol.replace(0, np.nan)
    lv = np.log(v.clip(lower=1.0))
    mu = lv.rolling(window, min_periods=window).mean()
    sd = lv.rolling(window, min_periods=window).std()
    return (lv - mu) / sd.replace(0, np.nan)


# ── Ranking / portfolio construction ─────────────────────────────────

def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def alpha_registry_score(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    vol: pd.DataFrame,
    factor_name: str,
    sign: float = 1.0,
) -> pd.DataFrame:
    """Compute a factor from `factors.registry` column-wise across the panel.

    Returns an xs-zscored DataFrame ready to be ranked.
    """
    from factors.registry import FACTOR_REGISTRY
    spec = FACTOR_REGISTRY[factor_name]
    col_map = {"close": close, "high": high, "low": low, "volume": vol}
    missing = [c for c in spec.inputs if c not in col_map]
    if missing:
        raise ValueError(f"missing inputs for {factor_name}: {missing}")

    def _one(sym: str) -> pd.Series:
        df = pd.DataFrame({c: col_map[c][sym] for c in spec.inputs})
        from factors.registry import _wrap
        runner = _wrap(spec.fn, spec.inputs,
                       optional_kwargs=spec.optional_kwargs, **spec.params)
        return runner(df)

    out = {}
    for sym in close.columns:
        try:
            out[sym] = _one(sym)
        except Exception:
            out[sym] = pd.Series(index=close.index, dtype=float)
    raw = pd.DataFrame(out).reindex(index=close.index)
    return zscore_xs(raw).fillna(0) * float(sign)


def composite_score(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    use_factors: tuple[str, ...] = ("mom_consist", "high52w", "vol_z"),
    vol_z_sign: float = 1.0,
) -> pd.DataFrame:
    """Composite = equal-weighted z-score of the selected factors.

    Parameters
    ----------
    use_factors : subset of {"mom_consist","high52w","vol_z"}. Empty tuple
        means "no alpha" (should only be used for sanity baseline).
    vol_z_sign : +1 treats high vol_z as bullish, −1 treats it as bearish.
    """
    parts: list[pd.DataFrame] = []
    if "mom_consist" in use_factors:
        parts.append(zscore_xs(factor_mom_consist(close)).fillna(0))
    if "high52w" in use_factors:
        parts.append(zscore_xs(factor_high52w(close)).fillna(0))
    if "vol_z" in use_factors:
        parts.append(zscore_xs(factor_vol_z(vol)).fillna(0) * vol_z_sign)
    if not parts:
        # Neutral score — equivalent to random selection within liquid set.
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)
    stacked = sum(parts) / float(len(parts))
    return stacked


def liquidity_mask(close: pd.DataFrame, vol: pd.DataFrame,
                   window: int = 60, min_adv: float = 5_000_000) -> pd.DataFrame:
    adv = (close * vol).rolling(window, min_periods=window).mean()
    return adv >= min_adv


# ── Rebalancing loop ────────────────────────────────────────────────

@dataclass
class RunConfig:
    db: str = "japan_market.db"
    start: str = "2024-01-01"
    end: str = "2026-04-14"
    top_k: int = 5
    fee_bps: float = 10.0       # each side (one-way)
    slippage_bps: float = 15.0  # each side
    min_adv: float = 5_000_000
    tag: str = "v1"
    # v1 additions
    use_factors: tuple[str, ...] = ("mom_consist", "high52w", "vol_z")
    vol_z_sign: float = 1.0
    fill_at: str = "open"          # "open" or "close" (at d+1)
    exclude_etf_reit: bool = True
    # Long-short vs long-only: if True also report short bottom-K.
    long_short: bool = False
    # When set, overrides `use_factors` and loads a single named factor
    # from `factors.registry`. `alpha_sign` flips direction (e.g., use -1
    # for Amihud if low-ILLIQ should be preferred).
    alpha_factor: str | None = None
    alpha_sign: float = 1.0
    # Multi-factor composite: list of "factor_name[:sign]" items, averaged
    # after xs-zscore. e.g. ("alpha_amihud_20:+1", "high52w:+1") —
    # allows mixing registry factors with the legacy high52w impl.
    alpha_composite: tuple[str, ...] = ()


def month_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return last trading day of each month present in `index`."""
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    last = df.groupby("ym").tail(1)
    return list(last.index)


def run_walk_forward(cfg: RunConfig) -> dict:
    panel = load_price_panel(cfg.db, cfg.start, cfg.end)
    if panel.empty:
        raise RuntimeError("empty price panel")

    close_full = pivot(panel, "close")
    open_full = pivot(panel, "open")
    vol_full = pivot(panel, "volume")

    if cfg.exclude_etf_reit:
        keep_cols = [c for c in close_full.columns if is_common_stock(c)]
        close = close_full[keep_cols]
        open_ = open_full.reindex(columns=keep_cols)
        vol = vol_full[keep_cols]
    else:
        close, open_, vol = close_full, open_full, vol_full

    if cfg.alpha_factor or cfg.alpha_composite:
        high = pivot(panel, "high")
        low = pivot(panel, "low")
        if cfg.exclude_etf_reit:
            high = high.reindex(columns=close.columns)
            low = low.reindex(columns=close.columns)
        if cfg.alpha_composite:
            parts = []
            for item in cfg.alpha_composite:
                name, _, s = item.partition(":")
                sign = float(s) if s else 1.0
                if name == "high52w":
                    parts.append(zscore_xs(factor_high52w(close)).fillna(0) * sign)
                elif name == "mom_consist":
                    parts.append(zscore_xs(factor_mom_consist(close)).fillna(0) * sign)
                else:
                    parts.append(alpha_registry_score(close, high, low, vol,
                                                     factor_name=name, sign=sign))
            score = sum(parts) / float(len(parts))
        else:
            score = alpha_registry_score(close, high, low, vol,
                                         factor_name=cfg.alpha_factor,
                                         sign=cfg.alpha_sign)
    else:
        score = composite_score(close, vol,
                                use_factors=tuple(cfg.use_factors),
                                vol_z_sign=cfg.vol_z_sign)
    liq = liquidity_mask(close, vol, min_adv=cfg.min_adv)
    masked = score.where(liq)

    rebal_dates = month_ends(close.index)
    if len(rebal_dates) < 3:
        raise RuntimeError(f"too few rebalance dates: {len(rebal_dates)}")

    cost_per_side = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0

    port_returns: list[tuple[pd.Timestamp, float, float, int]] = []
    prev_holdings: set[str] = set()

    # Choose fill price table — open at t+1 (market on open), else close at t+1.
    fill_tbl = open_ if cfg.fill_at == "open" else close

    for i in range(len(rebal_dates) - 1):
        d_signal = rebal_dates[i]
        d_next = rebal_dates[i + 1]

        row = masked.loc[d_signal].dropna()
        if row.empty:
            continue
        top = row.nlargest(cfg.top_k).index.tolist()
        bot = row.nsmallest(cfg.top_k).index.tolist() if cfg.long_short else []

        try:
            pos = close.index.get_loc(d_signal)
        except KeyError:
            continue
        if pos + 1 >= len(close.index):
            break
        fill_day = close.index[pos + 1]
        # Next-period fill day = first trading day AFTER d_next.
        try:
            pos_next = close.index.get_loc(d_next)
        except KeyError:
            continue
        if pos_next + 1 >= len(close.index):
            break
        exit_day = close.index[pos_next + 1]

        def _ret(names: list[str]) -> float | None:
            if not names:
                return None
            px_in = fill_tbl.loc[fill_day, names]
            px_out = fill_tbl.loc[exit_day, names]
            valid = (~px_in.isna()) & (~px_out.isna()) & (px_in > 0)
            if not valid.any():
                return None
            return float((px_out[valid] / px_in[valid] - 1.0).mean())

        long_r = _ret(top)
        if long_r is None:
            continue
        if cfg.long_short and bot:
            short_r = _ret(bot) or 0.0
            gross = long_r - short_r
        else:
            gross = long_r

        new_set = set(top) | set(bot)
        turnover = len(new_set.symmetric_difference(prev_holdings)) / max(len(new_set) + len(prev_holdings), 1) * 2.0
        cost = turnover * cost_per_side
        net = gross - cost
        prev_holdings = new_set

        port_returns.append((d_next, gross, net, len(top)))

    if not port_returns:
        raise RuntimeError("no portfolio returns produced")

    pr = pd.DataFrame(port_returns, columns=["date", "gross", "net", "n"]).set_index("date")

    # Benchmarks — ALIGNED to strategy's fill_day → exit_day windows, same
    # fill_at convention, so the comparison is apples to apples.
    # Use unfiltered tables for 1321 so excluding ETFs doesn't also drop benchmark.
    fill_tbl_full = open_full if cfg.fill_at == "open" else close_full
    bench_1321 = _aligned_benchmark_1321(close_full, rebal_dates, pr.index, fill_tbl_full)
    bench_ew = _aligned_universe_ew(close, rebal_dates, pr.index, fill_tbl,
                                    liq_mask=liq)
    pr["bench_1321"] = bench_1321
    pr["bench_ew"] = bench_ew

    metrics = _metrics(pr)
    out = {
        "config": asdict(cfg),
        "n_periods": len(pr),
        "metrics": metrics,
        "monthly": pr.reset_index().assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
    }
    return out


def _aligned_benchmark_1321(close, rebal_dates, out_index, fill_tbl) -> pd.Series:
    """1321.T return over the strategy's fill_day(i)→exit_day(i+1) windows.

    Needs to be loaded separately because ETF exclusion removes 1321 from
    close/vol. So we re-read it from the DB if missing.
    """
    sym = "1321.T"
    src = close
    if sym not in close.columns:
        # Pull from full DB to cover exclusion.
        # Caller passes full close; for simplicity reuse fill_tbl index only.
        return pd.Series([np.nan] * len(out_index), index=out_index)
    out = {}
    for i in range(len(rebal_dates) - 1):
        d_sig = rebal_dates[i]
        d_next = rebal_dates[i + 1]
        try:
            pos_sig = src.index.get_loc(d_sig)
            pos_nxt = src.index.get_loc(d_next)
        except KeyError:
            continue
        if pos_sig + 1 >= len(src.index) or pos_nxt + 1 >= len(src.index):
            continue
        fill_day = src.index[pos_sig + 1]
        exit_day = src.index[pos_nxt + 1]
        try:
            p_in = fill_tbl.loc[fill_day, sym]
            p_out = fill_tbl.loc[exit_day, sym]
        except KeyError:
            continue
        if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
            continue
        out[d_next] = float(p_out / p_in - 1.0)
    return pd.Series(out).reindex(out_index)


def _aligned_universe_ew(close, rebal_dates, out_index, fill_tbl, liq_mask) -> pd.Series:
    """Tradable equal-weight benchmark using same fill windows as strategy.

    Only averages names that are (a) liquid at signal date and (b) have
    valid fill/exit prices — i.e., the same candidate set the strategy
    could have drawn from.
    """
    out = {}
    for i in range(len(rebal_dates) - 1):
        d_sig = rebal_dates[i]
        d_next = rebal_dates[i + 1]
        try:
            pos_sig = close.index.get_loc(d_sig)
            pos_nxt = close.index.get_loc(d_next)
        except KeyError:
            continue
        if pos_sig + 1 >= len(close.index) or pos_nxt + 1 >= len(close.index):
            continue
        fill_day = close.index[pos_sig + 1]
        exit_day = close.index[pos_nxt + 1]
        liquid = liq_mask.loc[d_sig]
        cand = liquid[liquid].index
        if len(cand) == 0:
            continue
        p_in = fill_tbl.loc[fill_day, cand]
        p_out = fill_tbl.loc[exit_day, cand]
        valid = (~p_in.isna()) & (~p_out.isna()) & (p_in > 0)
        if not valid.any():
            continue
        r = (p_out[valid] / p_in[valid] - 1.0).mean()
        out[d_next] = float(r)
    return pd.Series(out).reindex(out_index)


def _benchmark_returns(close, sym, dates):
    """Legacy — kept for backward reference, not used in v1."""
    if sym not in close.columns:
        return pd.Series([np.nan] * len(dates), index=dates)
    s = close[sym].reindex(dates, method="ffill")
    return s.pct_change(fill_method=None)


def _metrics(pr: pd.DataFrame) -> dict:
    def stats(col: str) -> dict:
        r = pr[col].dropna()
        if r.empty:
            return {"mean": None, "std": None, "sharpe_ann": None, "cum": None, "maxdd": None}
        mean = r.mean()
        std = r.std()
        # monthly → annualize with √12
        sharpe = mean / std * np.sqrt(12) if std and std > 0 else None
        cum = float((1 + r).prod() - 1)
        roll = (1 + r).cumprod()
        maxdd = float(((roll / roll.cummax()) - 1).min())
        return {
            "mean_monthly": float(mean),
            "std_monthly": float(std) if std else None,
            "sharpe_ann": float(sharpe) if sharpe is not None else None,
            "cum": cum,
            "maxdd": maxdd,
        }
    return {
        "gross": stats("gross"),
        "net": stats("net"),
        "bench_1321": stats("bench_1321"),
        "bench_ew": stats("bench_ew"),
        "excess_net_vs_1321": float((pr["net"] - pr["bench_1321"]).dropna().mean())
            if "bench_1321" in pr else None,
        "excess_net_vs_ew": float((pr["net"] - pr["bench_ew"]).dropna().mean())
            if "bench_ew" in pr else None,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-04-14")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--fee_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=15.0)
    ap.add_argument("--min_adv", type=float, default=5_000_000)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--factors", default="mom_consist,high52w,vol_z",
                    help="comma-sep subset")
    ap.add_argument("--vol_z_sign", type=float, default=1.0)
    ap.add_argument("--fill_at", default="open", choices=("open", "close"))
    ap.add_argument("--no_exclude_etf", action="store_true")
    ap.add_argument("--long_short", action="store_true")
    ap.add_argument("--alpha_factor", default=None,
                    help="if set, loads single factor from factors.registry; overrides --factors")
    ap.add_argument("--alpha_sign", type=float, default=1.0)
    ap.add_argument("--alpha_composite", default=None,
                    help="comma-sep list of 'name:sign', e.g. 'alpha_amihud_20:1,high52w:1'")
    args = ap.parse_args()

    cfg = RunConfig(
        db=args.db, start=args.start, end=args.end, top_k=args.top_k,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        min_adv=args.min_adv, tag=args.tag,
        use_factors=tuple(f.strip() for f in args.factors.split(",") if f.strip()),
        vol_z_sign=args.vol_z_sign,
        fill_at=args.fill_at,
        exclude_etf_reit=not args.no_exclude_etf,
        long_short=args.long_short,
        alpha_factor=args.alpha_factor,
        alpha_sign=args.alpha_sign,
        alpha_composite=tuple(x.strip() for x in args.alpha_composite.split(",")) if args.alpha_composite else (),
    )
    out = run_walk_forward(cfg)

    out_path = Path(args.out) if args.out else Path("reports") / f"walk_forward_{cfg.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    m = out["metrics"]
    print(f"periods={out['n_periods']}")
    for k in ("gross", "net", "bench_1321", "bench_ew"):
        s = m[k]
        print(f"  {k:>12} cum={s['cum']:+.1%} sharpe={s['sharpe_ann']} maxdd={s['maxdd']}")
    print(f"  excess_net_vs_1321_monthly={m['excess_net_vs_1321']}")
    print(f"  excess_net_vs_ew_monthly  ={m['excess_net_vs_ew']}")
    print(f"→ {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
