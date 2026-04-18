"""
factor_diagnostics_batch.py — 批量因子诊断 (v4.0)

和 factor_diagnostics.py 的区别：
  - factor_diagnostics.py 每因子独立调用，2500 股 × 每次算 41 因子 → 重复计算
  - factor_diagnostics_batch.py 每 symbol 只算一次 → 41 因子共享，速度 40x+

核心指标 (无 alphalens 依赖，减少开销):
  - IC mean / IC std / IC IR / t-stat (Spearman rank corr)
  - 多周期 IC: T+1 / T+5 / T+10 / T+21
  - 因子间相关性矩阵
  - 建议: KEEP / FLIP_SIGN / DROP / DOWNWEIGHT

Usage:
    python factor_diagnostics_batch.py --db japan_market.db --asof 2026-04-15 --start 2025-06-01
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
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

REPORTS = Path("reports")
PERIODS = (1, 5, 10, 21)
IC_SIGNIFICANT = 0.02


def load_ohlcv_panels(conn, start: str, end: str):
    lookback_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_prices)").fetchall()}
    has_ohl = {"open", "high", "low"}.issubset(cols)
    sel = "date, symbol, close, volume" + (", open, high, low" if has_ohl else "")
    print(f"[batch] Loading daily_prices {lookback_start} ~ {end}...")
    df = pd.read_sql_query(
        f"SELECT {sel} FROM daily_prices WHERE date BETWEEN ? AND ?",
        conn, params=(lookback_start, end)
    )
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    if has_ohl:
        high = df.pivot(index="date", columns="symbol", values="high").sort_index()
        low = df.pivot(index="date", columns="symbol", values="low").sort_index()
        open_ = df.pivot(index="date", columns="symbol", values="open").sort_index()
    else:
        high = low = open_ = None
    print(f"[batch]   {close.shape[1]} symbols, {len(close)} days")
    return close, volume, high, low, open_


def compute_all_factors_panel(close, volume, high, low, open_, min_days: int = 60):
    """为整个宇宙算所有 41 + 3 因子，返回 {factor_name: DataFrame (T × N)}.

    核心: 每只股票只调 compute_short_term_factors 一次.
    """
    from factor_library import compute_short_term_factors

    print(f"[batch] Computing factors for universe...")
    factor_per_symbol: dict[str, dict[str, pd.Series]] = {}
    total = len(close.columns)
    for i, sym in enumerate(close.columns):
        if i % 500 == 0:
            print(f"[batch]   progress {i}/{total} ({i*100//total}%)")
        c = close[sym].dropna()
        if len(c) < min_days:
            continue
        v = volume[sym].dropna() if sym in volume.columns else None
        h = high[sym].dropna() if (high is not None and sym in high.columns) else None
        l = low[sym].dropna() if (low is not None and sym in low.columns) else None
        o = open_[sym].dropna() if (open_ is not None and sym in open_.columns) else None
        factor_per_symbol[sym] = compute_short_term_factors(c, h, l, o, v)

    # 加上旧 3 因子
    ret = close.pct_change(fill_method=None)
    up = (ret > 0).astype(float)
    mom_consist = up.rolling(20, min_periods=20).mean()

    rmax = close.rolling(252, min_periods=126).max()
    high52w = close / rmax - 1.0

    v_ = volume.replace(0, np.nan)
    lv = np.log(v_.clip(lower=1.0))
    vol_z = (lv - lv.rolling(60, min_periods=60).mean()) / lv.rolling(60, min_periods=60).std().replace(0, np.nan)

    # Pivot to {factor: panel}
    factor_names = set()
    for d in factor_per_symbol.values():
        factor_names.update(d.keys())

    panels: dict[str, pd.DataFrame] = {}
    for fn in factor_names:
        cols = {sym: d[fn] for sym, d in factor_per_symbol.items() if fn in d}
        if cols:
            panels[fn] = pd.DataFrame(cols)

    panels["mom_consist"] = mom_consist
    panels["high52w"] = high52w
    panels["vol_z"] = vol_z

    print(f"[batch] Computed {len(panels)} factor panels")
    return panels


def xs_spearman_ic(factor_panel: pd.DataFrame, fwd_returns: pd.DataFrame,
                   min_symbols: int = 30) -> pd.Series:
    """每日横截面 Spearman IC."""
    common_dates = factor_panel.index.intersection(fwd_returns.index)
    common_syms = factor_panel.columns.intersection(fwd_returns.columns)
    if len(common_dates) == 0 or len(common_syms) < min_symbols:
        return pd.Series(dtype=float)

    f = factor_panel.loc[common_dates, common_syms]
    r = fwd_returns.loc[common_dates, common_syms]

    # Vectorized rank across rows (each row = one date)
    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)

    ics = []
    dates = []
    for d in common_dates:
        fr = f_rank.loc[d].dropna()
        rr = r_rank.loc[d].dropna()
        common = fr.index.intersection(rr.index)
        if len(common) < min_symbols:
            continue
        ic = fr.loc[common].corr(rr.loc[common])
        if pd.notna(ic):
            ics.append(ic)
            dates.append(d)
    return pd.Series(ics, index=dates)


def analyze_factor_batch(factor_name: str, factor_panel: pd.DataFrame,
                         close_prices: pd.DataFrame,
                         start: str, end: str,
                         periods=PERIODS) -> dict:
    """针对单因子跑多周期 IC 统计."""
    # 切 start~end 范围
    factor_panel = factor_panel.loc[start:end] if not factor_panel.empty else factor_panel
    if factor_panel.empty or len(factor_panel) < 20:
        return {"factor": factor_name, "status": "insufficient_data"}

    result = {"factor": factor_name, "status": "ok", "ic": {}}

    for p in periods:
        fwd = close_prices.pct_change(p, fill_method=None).shift(-p)
        ic = xs_spearman_ic(factor_panel, fwd)
        if len(ic) < 5:
            result["ic"][f"{p}D"] = {"mean": None, "std": None, "ic_ir": None, "t_stat": None, "n": len(ic)}
            continue
        m, s = ic.mean(), ic.std()
        result["ic"][f"{p}D"] = {
            "mean": float(m),
            "std": float(s),
            "ic_ir": float(m / s) if s > 0 else 0.0,
            "t_stat": float(m / (s / np.sqrt(len(ic)))) if s > 0 else 0.0,
            "n": int(len(ic)),
        }

    # 建议
    ic1 = result["ic"].get("1D", {})
    m1 = ic1.get("mean") or 0
    ir1 = ic1.get("ic_ir") or 0
    t1 = ic1.get("t_stat") or 0

    if abs(t1) < 1.5 and abs(m1) < 0.005:
        action = "DROP"
    elif m1 < -IC_SIGNIFICANT and abs(t1) > 2.0:
        action = "FLIP_SIGN"
    elif abs(m1) < IC_SIGNIFICANT and ir1 < 0.10:
        action = "DOWNWEIGHT"
    else:
        action = "KEEP"

    result["recommended_action"] = action
    return result


def build_correlation(panels: dict, start: str, end: str, factors: list) -> pd.DataFrame:
    """跨因子 Pearson 相关（全宇宙全期 stack）."""
    print(f"[batch] Correlation on {len(factors)} factors...")
    stacked = {}
    for f in factors:
        if f not in panels:
            continue
        df = panels[f].loc[start:end].stack(dropna=True)
        stacked[f] = df
    if len(stacked) < 2:
        return pd.DataFrame()
    combined = pd.DataFrame(stacked).dropna()
    return combined.corr()


def write_report(results: list, corr: pd.DataFrame, asof: str, start: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "factor_tearsheet_batch.md"

    # 按 |IC(1D)| 排序
    def ic1_abs(r):
        ic1 = r.get("ic", {}).get("1D", {})
        m = ic1.get("mean")
        return abs(m) if m is not None else -1
    results_sorted = sorted(results, key=ic1_abs, reverse=True)

    lines = [
        f"# Factor Tearsheet (Batch) — {asof}",
        f"_Period: {start} → {asof}_",
        "",
        "## IC 排序（按 |IC(1D)| 降序）",
        "",
        "| Rank | 因子 | IC(1D) | IC IR | t-stat | IC(5D) | IC(21D) | 样本 | 建议 |",
        "|------|------|--------|-------|--------|--------|---------|------|------|",
    ]
    for i, r in enumerate(results_sorted, 1):
        if r.get("status") != "ok":
            lines.append(f"| {i} | {r['factor']} | — | — | — | — | — | — | {r.get('status')} |")
            continue
        ic = r["ic"]
        ic1 = ic.get("1D", {}) or {}
        ic5 = ic.get("5D", {}) or {}
        ic21 = ic.get("21D", {}) or {}
        emoji = {"KEEP": "✓", "FLIP_SIGN": "🔴", "DROP": "⚫", "DOWNWEIGHT": "🟡"}.get(r["recommended_action"], "?")
        lines.append(
            f"| {i} | {r['factor']} | {ic1.get('mean', 0):+.4f} | {ic1.get('ic_ir', 0):+.2f} "
            f"| {ic1.get('t_stat', 0):+.2f} | {ic5.get('mean', 0):+.4f} | {ic21.get('mean', 0):+.4f} "
            f"| {ic1.get('n', 0)} | {emoji} **{r['recommended_action']}** |"
        )

    # Top 候选 (建议 KEEP 且 |t-stat| > 2)
    top = [r for r in results_sorted if r.get("status") == "ok"
           and r["recommended_action"] in ("KEEP", "FLIP_SIGN")
           and abs((r["ic"].get("1D") or {}).get("t_stat", 0)) > 2.0]
    lines.append("")
    lines.append(f"## 🎯 Top 候选（|t-stat| > 2, {len(top)} 个）")
    lines.append("")
    if top:
        for r in top:
            ic1 = r["ic"]["1D"]
            action_note = " (翻转方向后使用)" if r["recommended_action"] == "FLIP_SIGN" else ""
            lines.append(f"- **{r['factor']}**: IC {ic1['mean']:+.4f}, t-stat {ic1['t_stat']:+.2f}{action_note}")
    else:
        lines.append("_无因子 |t-stat| > 2.0 —— 需要更长样本或更多数据_")

    if not corr.empty:
        lines.append("")
        lines.append("## 因子相关矩阵 (Top 10 IC)")
        top10 = [r["factor"] for r in results_sorted[:10] if r.get("status") == "ok"]
        if top10:
            sub = corr.loc[corr.index.intersection(top10), corr.columns.intersection(top10)]
            if not sub.empty:
                lines.append("```")
                lines.append(sub.round(2).to_string())
                lines.append("```")
                # 高相关对
                hc = []
                cols = sub.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        c = sub.iloc[i, j]
                        if abs(c) > 0.7:
                            hc.append((cols[i], cols[j], c))
                if hc:
                    lines.append("")
                    lines.append("🔴 **高相关对 (|ρ|>0.7)**：")
                    for a, b, c in hc:
                        lines.append(f"  - {a} ↔ {b}: ρ = {c:+.2f}")

    path.write_text("\n".join(lines), encoding="utf-8")

    (REPORTS / "factor_tearsheet_batch.json").write_text(
        json.dumps({"asof": asof, "start": start, "factors": results_sorted,
                    "correlation": corr.round(3).to_dict() if not corr.empty else {}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--asof", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--start", default=None)
    p.add_argument("--min_adv", type=float, default=50_000_000, help="过滤 universe: ADV 下限")
    args = p.parse_args()

    start = args.start or (datetime.strptime(args.asof, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")

    conn = sqlite3.connect(args.db)
    try:
        close, volume, high, low, open_ = load_ohlcv_panels(conn, start, args.asof)

        # 流动性过滤
        if args.min_adv > 0:
            adv = (close * volume).rolling(60, min_periods=20).mean()
            last_adv = adv.iloc[-1]
            liquid = last_adv[last_adv > args.min_adv].index.tolist()
            print(f"[batch] Liquidity filter ADV>{args.min_adv/1e6:.0f}M: {close.shape[1]} → {len(liquid)} symbols")
            close = close[liquid]
            volume = volume[liquid]
            if high is not None:
                high = high[[c for c in liquid if c in high.columns]]
                low = low[[c for c in liquid if c in low.columns]]
                open_ = open_[[c for c in liquid if c in open_.columns]]

        panels = compute_all_factors_panel(close, volume, high, low, open_)
    finally:
        conn.close()

    # 分析
    print(f"[batch] Analyzing IC per factor...")
    results = []
    for fname, fpanel in panels.items():
        r = analyze_factor_batch(fname, fpanel, close, start, args.asof)
        results.append(r)

    corr = build_correlation(panels, start, args.asof, list(panels.keys()))
    path = write_report(results, corr, args.asof, start)

    print(f"\nReport → {path}")
    ok = [r for r in results if r.get("status") == "ok"]
    ok_sorted = sorted(ok, key=lambda r: abs((r["ic"].get("1D") or {}).get("mean") or 0), reverse=True)
    print(f"\nTop 10 by |IC(1D)|:")
    for r in ok_sorted[:10]:
        ic1 = r["ic"]["1D"]
        print(f"  {r['factor']:20s}  IC={ic1['mean']:+.4f}  IR={ic1['ic_ir']:+.2f}  t={ic1['t_stat']:+.2f}  → {r['recommended_action']}")


if __name__ == "__main__":
    main()
