"""
factor_diagnostics.py — alphalens-reloaded 包装 (v4.0)

给每个生产因子打一份完整体检报告：
  - IC 时间序列 + 月度衰减曲线
  - 分层回测单调性
  - 双向组合累计曲线（long Q5 / short Q1）
  - Turnover 估计
  - 和其他因子的相关性矩阵
  - 自动给出 KEEP/FLIP/DROP 三选一建议

输出: reports/factor_tearsheet_{factor}.md + png (若 matplotlib OK)

用法:
    python factor_diagnostics.py --db japan_market.db --asof 2026-04-16 \
           --factors mom_consist,high52w,vol_z --start 2025-01-01
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPORTS = Path("reports")
DEFAULT_PERIODS = (1, 5, 10, 21)   # T+1, T+5, T+10, T+21
DEFAULT_QUANTILES = 5


def load_factor_panel(conn: sqlite3.Connection, factor: str,
                      start: str, end: str) -> pd.Series:
    """从 feature_daily 读因子面板，返回 (date, asset) MultiIndex Series."""
    df = pd.read_sql_query(
        "SELECT asof AS date, symbol AS asset, value AS factor "
        "FROM feature_daily WHERE feature_name=? AND asof BETWEEN ? AND ?",
        conn, params=(factor, start, end)
    )
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "asset"])["factor"].dropna()


def compute_factor_from_prices(conn: sqlite3.Connection, factor: str,
                               start: str, end: str) -> pd.Series:
    """当 feature_daily 数据不足时，直接从 daily_prices 实时算因子.

    支持：
    - 旧 3 因子 (mom_consist, high52w, vol_z) — 直接 panel 实现
    - factor_library 里的 41 个短期因子 — 每 symbol 调 compute_short_term_factors
    """
    # 读 OHLCV（新因子需要 open/high/low）
    lookback_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=300)).strftime("%Y-%m-%d")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_prices)").fetchall()}
    has_ohl = {"open", "high", "low"}.issubset(cols)
    sel = "date, symbol, close, volume" + (", open, high, low" if has_ohl else "")
    df = pd.read_sql_query(
        f"SELECT {sel} FROM daily_prices WHERE date BETWEEN ? AND ?",
        conn, params=(lookback_start, end)
    )
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    high = df.pivot(index="date", columns="symbol", values="high").sort_index() if has_ohl else None
    low = df.pivot(index="date", columns="symbol", values="low").sort_index() if has_ohl else None
    open_ = df.pivot(index="date", columns="symbol", values="open").sort_index() if has_ohl else None

    # 路径 1: 旧 3 因子
    if factor in ("mom_consist", "high52w", "vol_z"):
        if factor == "mom_consist":
            ret = close.pct_change(fill_method=None)
            up = (ret > 0).astype(float)
            panel = up.rolling(20, min_periods=20).mean()
        elif factor == "high52w":
            rmax = close.rolling(252, min_periods=126).max()
            panel = close / rmax - 1.0
        else:  # vol_z
            v = volume.replace(0, np.nan)
            lv = np.log(v.clip(lower=1.0))
            mu = lv.rolling(60, min_periods=60).mean()
            sd = lv.rolling(60, min_periods=60).std()
            panel = (lv - mu) / sd.replace(0, np.nan)
        panel = panel.loc[start:end]
        long = panel.stack(dropna=True)
        long.index.names = ["date", "asset"]
        return long

    # 路径 2: factor_library 里的短期因子
    try:
        from factor_library import compute_short_term_factors, SHORT_TERM_FACTOR_LIST
    except ImportError:
        return pd.Series(dtype=float)

    if factor not in SHORT_TERM_FACTOR_LIST:
        return pd.Series(dtype=float)

    # 为每个 symbol 算，合并成 panel
    symbol_panels = {}
    for sym in close.columns:
        c = close[sym].dropna()
        if len(c) < 60:
            continue
        v = volume[sym].dropna() if sym in volume.columns else None
        h = high[sym].dropna() if (high is not None and sym in high.columns) else None
        l = low[sym].dropna() if (low is not None and sym in low.columns) else None
        o = open_[sym].dropna() if (open_ is not None and sym in open_.columns) else None
        factors = compute_short_term_factors(c, h, l, o, v)
        if factor in factors:
            s = factors[factor].dropna()
            if not s.empty:
                symbol_panels[sym] = s

    if not symbol_panels:
        return pd.Series(dtype=float)

    panel = pd.DataFrame(symbol_panels)
    panel = panel.loc[start:end] if not panel.empty else panel
    long = panel.stack(dropna=True)
    long.index.names = ["date", "asset"]
    return long


def load_price_panel(conn: sqlite3.Connection, start: str, end_buffer: str) -> pd.DataFrame:
    """读 daily_prices 宽表（date × asset）."""
    df = pd.read_sql_query(
        "SELECT date, symbol AS asset, close FROM daily_prices "
        "WHERE date BETWEEN ? AND ?",
        conn, params=(start, end_buffer)
    )
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="asset", values="close").sort_index()


def run_single_factor(
    conn: sqlite3.Connection,
    factor: str,
    start: str,
    end: str,
    periods=DEFAULT_PERIODS,
    quantiles: int = DEFAULT_QUANTILES,
) -> dict:
    from alphalens.utils import get_clean_factor_and_forward_returns
    from alphalens.performance import (
        factor_information_coefficient,
        mean_information_coefficient,
        mean_return_by_quantile,
        quantile_turnover,
    )

    fac = load_factor_panel(conn, factor, start, end)
    # 数据稀疏时从 daily_prices 实时算（保证回测可用）
    if fac.empty or len(fac.index.get_level_values(0).unique()) < 30:
        print(f"  [fallback] {factor}: feature_daily 不足 30 天，从日线实时计算")
        fac = compute_factor_from_prices(conn, factor, start, end)
    if fac.empty or len(fac) < 100:
        return {"factor": factor, "status": "insufficient_data", "n": len(fac)}

    end_buf = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=max(periods) + 5)).strftime("%Y-%m-%d")
    px = load_price_panel(conn, start, end_buf)
    if px.empty:
        return {"factor": factor, "status": "no_prices"}

    try:
        factor_data = get_clean_factor_and_forward_returns(
            factor=fac,
            prices=px,
            periods=periods,
            quantiles=quantiles,
            max_loss=0.5,
        )
    except Exception as e:
        return {"factor": factor, "status": f"alphalens_error: {e}"}

    ic = factor_information_coefficient(factor_data)
    ic_summary = {
        f"{p}D": {
            "mean":  float(ic[f"{p}D"].mean()),
            "std":   float(ic[f"{p}D"].std()),
            "ic_ir": float(ic[f"{p}D"].mean() / ic[f"{p}D"].std()) if ic[f"{p}D"].std() > 0 else 0.0,
            "t_stat": float(ic[f"{p}D"].mean() / (ic[f"{p}D"].std() / np.sqrt(len(ic))))
                      if ic[f"{p}D"].std() > 0 else 0.0,
        } for p in periods
    }

    qret, qret_err = mean_return_by_quantile(factor_data, by_group=False)
    q5_vs_q1 = {}
    try:
        q5_mean = qret.xs(quantiles, level="factor_quantile")["1D"].mean()
        q1_mean = qret.xs(1, level="factor_quantile")["1D"].mean()
        spread = (q5_mean - q1_mean) * 252
        monotonic = bool(qret["1D"].groupby("factor_quantile").mean().is_monotonic_increasing)
        q5_vs_q1 = {
            "q5_mean_1d": float(q5_mean),
            "q1_mean_1d": float(q1_mean),
            "long_short_ann": float(spread),
            "monotonic_increasing": monotonic,
        }
    except Exception:
        pass

    turnover = {}
    try:
        for p in periods:
            t = quantile_turnover(factor_data["factor_quantile"], quantile=quantiles, period=p)
            turnover[f"{p}D_top_q"] = float(t.mean())
    except Exception:
        pass

    # 分析建议
    ic_1d_mean = ic_summary["1D"]["mean"]
    ic_1d_ir = ic_summary["1D"]["ic_ir"]
    action = "KEEP"
    notes = []
    if abs(ic_1d_mean) < 0.005 and abs(ic_1d_ir) < 0.1:
        action = "DROP"
        notes.append(f"IC 均值 {ic_1d_mean:+.4f}，IR {ic_1d_ir:+.2f}，几乎无 alpha")
    elif ic_1d_mean < -0.02:
        action = "FLIP_SIGN"
        notes.append(f"IC 均值 {ic_1d_mean:+.3f} 显著为负，建议翻转 sign")
    elif q5_vs_q1.get("monotonic_increasing") is False and abs(ic_1d_mean) < 0.015:
        action = "DOWNWEIGHT"
        notes.append("分层非单调且 IC 较弱")

    return {
        "factor": factor,
        "status": "ok",
        "n_observations": int(len(factor_data)),
        "ic": ic_summary,
        "q5_vs_q1": q5_vs_q1,
        "turnover": turnover,
        "recommended_action": action,
        "notes": notes,
    }


def factor_correlation_matrix(conn: sqlite3.Connection, factors: list,
                              start: str, end: str) -> pd.DataFrame:
    """计算因子间横截面相关性矩阵 (Pearson over full panel)."""
    panels = {}
    for f in factors:
        fac = load_factor_panel(conn, f, start, end)
        if not fac.empty:
            panels[f] = fac
    if len(panels) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(panels).dropna()
    return df.corr()


def write_markdown_report(results: list, corr: pd.DataFrame, asof: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "factor_tearsheet_summary.md"

    lines = [
        f"# Factor Tearsheet Summary — {asof}",
        "",
        "## 每因子概览",
        "",
        "| 因子 | 样本 | IC(1D) | IC IR(1D) | t-stat | Q5-Q1 年化 | 单调 | Turnover(1D) | 建议 |",
        "|------|------|--------|-----------|--------|-----------|------|--------------|------|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['factor']} | — | — | — | — | — | — | — | {r.get('status','?')} |")
            continue
        ic1 = r["ic"]["1D"]
        qv = r.get("q5_vs_q1", {})
        tv = r.get("turnover", {})
        emoji = {"KEEP": "✓", "FLIP_SIGN": "🔴", "DROP": "⚫", "DOWNWEIGHT": "🟡"}.get(r["recommended_action"], "?")
        lines.append(
            f"| {r['factor']} | {r['n_observations']} | {ic1['mean']:+.4f} | {ic1['ic_ir']:+.2f} "
            f"| {ic1['t_stat']:+.2f} | {qv.get('long_short_ann', 0)*100:+.1f}% "
            f"| {'✓' if qv.get('monotonic_increasing') else '✗'} "
            f"| {tv.get('1D_top_q', 0)*100:.1f}% "
            f"| {emoji} **{r['recommended_action']}** |"
        )

    lines.append("")
    lines.append("## 多周期 IC 衰减")
    lines.append("")
    lines.append("| 因子 | T+1 | T+5 | T+10 | T+21 |")
    lines.append("|------|-----|-----|------|------|")
    for r in results:
        if r.get("status") != "ok":
            continue
        row = [f"| {r['factor']} "]
        for p in DEFAULT_PERIODS:
            key = f"{p}D"
            row.append(f"| {r['ic'][key]['mean']:+.4f} ")
        row.append("|")
        lines.append("".join(row))

    if not corr.empty:
        lines.append("")
        lines.append("## 因子相关矩阵")
        lines.append("")
        lines.append("```")
        lines.append(corr.round(3).to_string())
        lines.append("```")
        lines.append("")
        high_corr = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if abs(c) > 0.7:
                    high_corr.append((cols[i], cols[j], c))
        if high_corr:
            lines.append("🔴 **高相关对 (|ρ|>0.7)**：")
            for a, b, c in high_corr:
                lines.append(f"  - {a} <-> {b}: ρ = {c:+.3f} → 保留 ICIR 更高的一个")

    lines.append("")
    lines.append("## 详细说明")
    for r in results:
        if r.get("status") != "ok":
            continue
        lines.append(f"### {r['factor']}")
        for n in r.get("notes", []):
            lines.append(f"- {n}")
        if not r.get("notes"):
            lines.append("- 无特别告警")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    # JSON 版本
    (REPORTS / "factor_tearsheet_summary.json").write_text(
        json.dumps({"asof": asof, "factors": results,
                    "correlation": corr.round(3).to_dict() if not corr.empty else {}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--asof", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--factors", default="mom_consist,high52w,vol_z")
    p.add_argument("--start", default=None, help="默认 asof - 180d")
    p.add_argument("--periods", default="1,5,10,21")
    p.add_argument("--quantiles", type=int, default=5)
    args = p.parse_args()

    start = args.start or (datetime.strptime(args.asof, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    periods = tuple(int(x) for x in args.periods.split(","))
    factors = [s.strip() for s in args.factors.split(",") if s.strip()]

    conn = sqlite3.connect(args.db)
    try:
        results = []
        for f in factors:
            print(f"[factor_diagnostics] {f}...")
            r = run_single_factor(conn, f, start, args.asof, periods=periods, quantiles=args.quantiles)
            results.append(r)
        print("[factor_diagnostics] correlation matrix...")
        corr = factor_correlation_matrix(conn, factors, start, args.asof)
    finally:
        conn.close()

    path = write_markdown_report(results, corr, args.asof)
    print(f"\nReport → {path}")
    for r in results:
        if r.get("status") == "ok":
            print(f"  {r['factor']}: IC(1D)={r['ic']['1D']['mean']:+.4f}  IR={r['ic']['1D']['ic_ir']:+.2f}  → {r['recommended_action']}")
        else:
            print(f"  {r['factor']}: {r['status']}")


if __name__ == "__main__":
    main()
