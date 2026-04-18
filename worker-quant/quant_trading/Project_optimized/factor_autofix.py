"""
factor_autofix.py — 因子 IC 反向检测 (v4.0)

每周/每日扫描每个生产因子的 rolling IC（Spearman）。
如果 |IC| > 0.03 但符号和配置的 weight 符号相反超过 window_days 天，
flag 到 reports/factor_ic_alerts.md 并建议翻转 sign。

目的: 防止 vol_z 符号搞反 6 个月都没人发现的事再发生。

用法:
    python factor_autofix.py --db japan_market.db --asof 2026-04-16 --window 30

输出:
    reports/factor_ic_alerts.md       — 人可读报告
    reports/factor_ic_history.json    — 时间序列，供 briefing 引用
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path("reports")

# 生产因子 + 配置的期望 sign (+1 = 高值看多, -1 = 高值看空)
# 必须和 sprint_signal.sprint_score 的 weights 保持一致
FACTOR_SIGNS = {
    "mom_consist": +1.0,
    "high52w":     +1.0,
    "vol_z":       -1.0,   # 2026-04-16 修正后
}

IC_ALERT_THRESHOLD = 0.03     # |IC| > 0.03 才算显著
IC_REVERSAL_WINDOW = 30       # 回看天数
IC_REVERSAL_MIN_DAYS = 10     # 至少 10 天持续反向才 flag


def _fetch_factor_panel(conn: sqlite3.Connection, factor: str, start: str, end: str) -> pd.DataFrame:
    """从 feature_daily 读取因子面板 (T × N)."""
    df = pd.read_sql_query(
        "SELECT asof, symbol, value FROM feature_daily "
        "WHERE feature_name=? AND asof BETWEEN ? AND ?",
        conn, params=(factor, start, end)
    )
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="asof", columns="symbol", values="value")


def _fetch_fwd_returns(conn: sqlite3.Connection, start: str, end: str, horizon: int = 1) -> pd.DataFrame:
    """计算 T+1 收益面板."""
    df = pd.read_sql_query(
        "SELECT date, symbol, close FROM daily_prices WHERE date BETWEEN ? AND ?",
        conn, params=(start, (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=horizon + 5)).strftime("%Y-%m-%d"))
    )
    if df.empty:
        return pd.DataFrame()
    px = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return px.pct_change(horizon, fill_method=None).shift(-horizon)


def _daily_ic(factor_panel: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """每日横截面 Spearman IC."""
    common_dates = factor_panel.index.intersection(fwd.index)
    common_syms = factor_panel.columns.intersection(fwd.columns)
    ic_series = {}
    for d in common_dates:
        f = factor_panel.loc[d, common_syms].dropna()
        r = fwd.loc[d, common_syms].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 20:
            continue
        ic = f.loc[common].rank().corr(r.loc[common].rank())
        if pd.notna(ic):
            ic_series[d] = ic
    return pd.Series(ic_series).sort_index()


def analyze_factor(conn: sqlite3.Connection, factor: str, expected_sign: float,
                   asof: str, window: int = IC_REVERSAL_WINDOW) -> dict:
    """对单个因子跑 IC 诊断."""
    end = asof
    start = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=window * 2)).strftime("%Y-%m-%d")

    fac = _fetch_factor_panel(conn, factor, start, end)
    if fac.empty:
        return {"factor": factor, "status": "no_data", "expected_sign": expected_sign}

    fwd = _fetch_fwd_returns(conn, start, end)
    if fwd.empty:
        return {"factor": factor, "status": "no_returns", "expected_sign": expected_sign}

    ic = _daily_ic(fac, fwd)
    if len(ic) < 5:
        return {"factor": factor, "status": "insufficient_ic", "expected_sign": expected_sign, "ic_days": len(ic)}

    recent = ic.iloc[-window:] if len(ic) >= window else ic
    ic_mean = float(recent.mean())
    ic_std = float(recent.std())
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

    # 反向检测: 当 expected_sign > 0 时，IC 若为显著负值（连续多天）→ 反向
    reversal_days = 0
    for v in recent.values:
        if pd.notna(v) and abs(v) > IC_ALERT_THRESHOLD:
            if np.sign(v) != np.sign(expected_sign):
                reversal_days += 1

    is_reversed = reversal_days >= IC_REVERSAL_MIN_DAYS
    action = "FLIP_SIGN" if is_reversed else ("KEEP" if abs(ic_mean) > 0.01 else "MONITOR")
    if abs(ic_mean) < 0.005 and len(recent) >= 20:
        action = "DROP_LOW_IC"

    return {
        "factor": factor,
        "expected_sign": expected_sign,
        "ic_mean_recent": round(ic_mean, 4),
        "ic_std_recent": round(ic_std, 4),
        "ic_ir_recent": round(ic_ir, 3),
        "ic_days": len(recent),
        "reversal_days": reversal_days,
        "is_reversed": is_reversed,
        "action": action,
        "status": "ok",
    }


def run_autofix(db_path: str, asof: str, window: int = IC_REVERSAL_WINDOW) -> dict:
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        return {"error": str(e)}

    results = []
    try:
        for factor, sign in FACTOR_SIGNS.items():
            results.append(analyze_factor(conn, factor, sign, asof, window))
    finally:
        conn.close()

    report = {"asof": asof, "window": window, "factors": results,
              "alerts": [r for r in results if r.get("action") == "FLIP_SIGN"]}

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "factor_ic_history.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # MD 报告
    lines = [
        f"# Factor IC Autofix Report — {asof}",
        f"_Window: last {window} days, threshold |IC| > {IC_ALERT_THRESHOLD}_",
        "",
        "| 因子 | 期望sign | IC均值 | IC IR | 反向天数 | 状态 | 建议 |",
        "|------|---------|-------|-------|---------|------|------|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['factor']} | {r['expected_sign']:+.0f} | — | — | — | {r['status']} | 数据不足 |")
            continue
        emoji = "🔴" if r["action"] == "FLIP_SIGN" else ("🟡" if r["action"] == "DROP_LOW_IC" else "✓")
        lines.append(
            f"| {r['factor']} | {r['expected_sign']:+.0f} | {r['ic_mean_recent']:+.3f} "
            f"| {r['ic_ir_recent']:+.2f} | {r['reversal_days']}/{r['ic_days']} "
            f"| {emoji} | **{r['action']}** |"
        )

    if report["alerts"]:
        lines.append("")
        lines.append("## 🔴 反向告警")
        for a in report["alerts"]:
            lines.append(f"- **{a['factor']}**: 期望 sign={a['expected_sign']:+.0f}, 实际 IC 均值 {a['ic_mean_recent']:+.3f} "
                         f"(反向 {a['reversal_days']}/{a['ic_days']} 天) → **建议翻转 sign**")

    (REPORTS / "factor_ic_alerts.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="japan_market.db")
    p.add_argument("--asof", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--window", type=int, default=IC_REVERSAL_WINDOW)
    args = p.parse_args()

    report = run_autofix(args.db, args.asof, args.window)
    if "error" in report:
        print(f"ERROR: {report['error']}")
        raise SystemExit(1)
    alerts = report.get("alerts", [])
    print(f"Factors analyzed: {len(report['factors'])}")
    print(f"Alerts: {len(alerts)}")
    for a in alerts:
        print(f"  🔴 {a['factor']}: IC mean {a['ic_mean_recent']:+.3f}, 反向 {a['reversal_days']} 天 → FLIP_SIGN")
    print(f"Report: reports/factor_ic_alerts.md")
