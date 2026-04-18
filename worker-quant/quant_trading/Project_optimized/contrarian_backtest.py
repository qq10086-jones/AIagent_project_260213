"""
contrarian_backtest.py — 反转策略独立回测 (v4.0 根据 factor_diagnostics_batch 发现)

基于 alphalens 体检发现：日股 ADV>50M 流动池在 2025-06~2026-04 期间
短期动量全部是反向信号（ksft2 t=-12.66, ma_gap_3 t=-6.74 等）。

本脚本测试反转策略 composite 相对当前 sprint + 日经 的表现。

Usage:
    python contrarian_backtest.py --start 2025-06-01 --end 2026-04-15 --top_k 10
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Contrarian factor composite ─────────────────────────────────

CONTRARIAN_FACTORS = {
    "ksft2":     -1.0,   # 收盘在日内区间位置
    "ma_gap_3":  -1.0,   # 3日均线偏离
    "resi10":    -1.0,   # 10日趋势残差
    "high52w":   +1.0,   # 52周高点距离（正向长期质量）
    "imin5":     +1.0,   # 5日最低位（超卖反弹）
}


def xs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score 每一行."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def load_panel(conn, start, end):
    lookback = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        "SELECT date, symbol, close, volume, open, high, low FROM daily_prices "
        "WHERE date BETWEEN ? AND ?",
        conn, params=(lookback, end)
    )
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    high = df.pivot(index="date", columns="symbol", values="high").sort_index()
    low = df.pivot(index="date", columns="symbol", values="low").sort_index()
    open_ = df.pivot(index="date", columns="symbol", values="open").sort_index()
    return close, volume, high, low, open_


def compute_contrarian_panels(close, volume, high, low, open_, min_days=60):
    from factor_library import compute_short_term_factors

    factor_per_symbol = {}
    for sym in close.columns:
        c = close[sym].dropna()
        if len(c) < min_days:
            continue
        v = volume[sym].dropna() if sym in volume.columns else None
        h = high[sym].dropna() if sym in high.columns else None
        l = low[sym].dropna() if sym in low.columns else None
        o = open_[sym].dropna() if sym in open_.columns else None
        factor_per_symbol[sym] = compute_short_term_factors(c, h, l, o, v)

    # high52w 单独算（factor_library 没有）
    rmax = close.rolling(252, min_periods=126).max()
    high52w = close / rmax - 1.0

    panels = {}
    for fn in CONTRARIAN_FACTORS:
        if fn == "high52w":
            panels[fn] = high52w
            continue
        cols = {sym: d[fn] for sym, d in factor_per_symbol.items() if fn in d}
        if cols:
            panels[fn] = pd.DataFrame(cols)
    return panels


def composite_score(panels, weights=CONTRARIAN_FACTORS) -> pd.DataFrame:
    """对每个因子 z-score + 加权平均."""
    parts = []
    total_w = 0.0
    for fn, w in weights.items():
        if fn not in panels:
            continue
        z = xs_zscore(panels[fn]).fillna(0)
        parts.append(z * w)
        total_w += abs(w)
    if not parts:
        raise RuntimeError("no panels")
    # 面板并集
    all_cols = set()
    all_idx = None
    for p in parts:
        all_cols.update(p.columns)
        all_idx = p.index if all_idx is None else all_idx.union(p.index)
    parts = [p.reindex(index=all_idx, columns=sorted(all_cols)).fillna(0) for p in parts]
    return sum(parts) / max(total_w, 1e-12)


def liquidity_mask(close, volume, window=60, min_adv=50_000_000):
    adv = (close * volume).rolling(window, min_periods=window).mean()
    return adv >= min_adv


def month_ends(index):
    df = pd.DataFrame(index=index)
    df["ym"] = df.index.to_period("M")
    return list(df.groupby("ym").tail(1).index)


def sqrt_slippage_bps(participation=0.05, k=10.0):
    return k * np.sqrt(participation * 100.0)


def run_rebalance(score, close, open_, top_k=10, fee_bps=5.0, slip_bps=None):
    if slip_bps is None:
        slip_bps = sqrt_slippage_bps()
    cost_per_side = (fee_bps + slip_bps) / 10_000.0

    rebal_dates = month_ends(close.index)
    fill_tbl = open_

    prev_holdings = set()
    out = []
    for i in range(len(rebal_dates) - 1):
        d_sig = rebal_dates[i]
        d_next = rebal_dates[i + 1]
        if d_sig not in score.index:
            continue
        row = score.loc[d_sig].dropna()
        if row.empty:
            continue
        top = row.nlargest(top_k).index.tolist()
        try:
            fill_day = close.index[close.index.get_loc(d_sig) + 1]
            exit_day = close.index[close.index.get_loc(d_next) + 1]
        except (KeyError, IndexError):
            continue

        px_in = fill_tbl.loc[fill_day, top]
        px_out = fill_tbl.loc[exit_day, top]
        valid = (~px_in.isna()) & (~px_out.isna()) & (px_in > 0)
        if not valid.any():
            continue
        gross = float((px_out[valid] / px_in[valid] - 1.0).mean())
        new_set = set(top)
        turnover = len(new_set.symmetric_difference(prev_holdings)) / max(len(new_set) + len(prev_holdings), 1) * 2.0
        cost = turnover * cost_per_side
        net = gross - cost
        prev_holdings = new_set
        out.append((d_next, gross, net, len(top)))

    return pd.DataFrame(out, columns=["date", "gross", "net", "n"]).set_index("date")


def benchmark_return(close, rebal_dates, out_idx, fill_tbl, sym="1321.T"):
    if sym not in close.columns:
        return pd.Series(np.nan, index=out_idx)
    out = {}
    for i in range(len(rebal_dates) - 1):
        d_sig = rebal_dates[i]
        d_next = rebal_dates[i + 1]
        try:
            f = close.index[close.index.get_loc(d_sig) + 1]
            e = close.index[close.index.get_loc(d_next) + 1]
        except (KeyError, IndexError):
            continue
        p_in = fill_tbl.loc[f, sym]
        p_out = fill_tbl.loc[e, sym]
        if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
            continue
        out[d_next] = float(p_out / p_in - 1.0)
    return pd.Series(out).reindex(out_idx)


def metrics(pr, col="net"):
    r = pr[col].dropna()
    if r.empty:
        return {}
    m, s = r.mean(), r.std()
    cum = float((1 + r).prod() - 1)
    roll = (1 + r).cumprod()
    maxdd = float(((roll / roll.cummax()) - 1).min())
    sharpe = float(m / s * np.sqrt(12)) if s > 0 else 0
    return {"cum": cum, "sharpe_ann": sharpe, "maxdd": maxdd, "n_months": len(r)}


def block_bootstrap(r, block_len=3, n=500, seed=42):
    r = r.dropna().values
    if len(r) < block_len * 3:
        return {}
    rng = np.random.default_rng(seed)
    sh = []
    for _ in range(n):
        starts = rng.integers(0, len(r) - block_len + 1, len(r) // block_len)
        sample = np.concatenate([r[s:s + block_len] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(12))
    arr = np.array(sh)
    return {"sharpe_lo": float(np.percentile(arr, 2.5)),
            "sharpe_hi": float(np.percentile(arr, 97.5)),
            "sharpe_mean": float(arr.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--start", default="2025-06-01")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--min_adv", type=float, default=50_000_000)
    p.add_argument("--fee_bps", type=float, default=5.0)
    p.add_argument("--tag", default="contrarian_v1")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        print(f"[contrarian] loading panel {args.start}~{args.end}...")
        close, volume, high, low, open_ = load_panel(conn, args.start, args.end)

        # 流动性过滤
        liq = liquidity_mask(close, volume, min_adv=args.min_adv)
        liquid_syms = liq.iloc[-1]
        liquid_syms = liquid_syms[liquid_syms].index
        print(f"[contrarian] liquidity filter: {close.shape[1]} → {len(liquid_syms)}")
        close = close[liquid_syms]
        volume = volume[liquid_syms]
        high = high[[c for c in liquid_syms if c in high.columns]]
        low = low[[c for c in liquid_syms if c in low.columns]]
        open_ = open_[[c for c in liquid_syms if c in open_.columns]]

        # Keep full index to benchmark retrieval
        close_full, volume_full, high_full, low_full, open_full = load_panel(conn, args.start, args.end)
    finally:
        conn.close()

    print(f"[contrarian] computing factor panels...")
    panels = compute_contrarian_panels(close, volume, high, low, open_)
    print(f"[contrarian]   panels: {list(panels.keys())}")

    score = composite_score(panels)
    # 切 start~end
    score = score.loc[args.start:args.end]
    score = score.mask(~liq.reindex_like(score).fillna(False))

    print(f"[contrarian] rebalancing top {args.top_k} monthly...")
    pr = run_rebalance(score, close, open_, top_k=args.top_k, fee_bps=args.fee_bps)

    # 基准
    rebal = month_ends(close.index)
    pr["bench_1321"] = benchmark_return(close_full, rebal, pr.index, open_full, "1321.T")

    print(f"\n===== Contrarian v1 ({args.start} → {args.end}) =====")
    print(f"rebalance periods: {len(pr)}")
    print(f"cost: fee={args.fee_bps}bps + slip={sqrt_slippage_bps():.1f}bps (sqrt impact)")
    for c in ("gross", "net", "bench_1321"):
        m = metrics(pr, c)
        if m:
            print(f"  {c:>12}  cum={m['cum']:+.1%}  sharpe={m['sharpe_ann']:+.2f}  maxdd={m['maxdd']:+.1%}")

    # bootstrap
    for c in ("net", "bench_1321"):
        b = block_bootstrap(pr[c])
        if b:
            print(f"  [bootstrap] {c:>10}  Sharpe 95% CI: [{b['sharpe_lo']:+.2f}, {b['sharpe_hi']:+.2f}]  mean={b['sharpe_mean']:+.2f}")

    excess = (pr["net"] - pr["bench_1321"]).mean()
    print(f"  excess_net_vs_1321 monthly = {excess:+.4f}")

    # 保存
    out_path = Path("reports") / f"contrarian_backtest_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "config": vars(args),
            "factors": CONTRARIAN_FACTORS,
            "periods": len(pr),
            "monthly": pr.reset_index().assign(date=lambda d: d["date"].astype(str)).to_dict(orient="records"),
            "metrics": {c: metrics(pr, c) for c in ("gross", "net", "bench_1321")},
        }, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
