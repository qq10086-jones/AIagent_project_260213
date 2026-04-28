"""ETF Buy-Hold Monthly Check — Path A (2026-04-28 decision)

Reads etf_buyhold positions, compares NAV vs benchmark, checks rebalance need,
and emails the report.

Run monthly via Windows Task Scheduler (1st of each month, 09:30 JST).

Required environment variables for email:
  EMAIL_USER : SMTP login (e.g. lwyssq@gmail.com)
  EMAIL_PASS : SMTP app password (Gmail: https://myaccount.google.com/apppasswords)

If env vars unset, report is written to reports/etf_monthly_<asof>.md only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import smtplib
import sqlite3
import ssl
import sys
import textwrap
from email.message import EmailMessage
from pathlib import Path

import yaml

JST_OFFSET = dt.timedelta(hours=9)
ETF_STRATEGY_ID = "etf_buyhold"


def jst_now() -> dt.datetime:
    return dt.datetime.utcnow() + JST_OFFSET


def load_etf_config(config_path: str = "config.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    profile = (cfg.get("strategy_profiles") or {}).get(ETF_STRATEGY_ID)
    if not profile:
        raise RuntimeError(
            f"strategy_profiles.{ETF_STRATEGY_ID} not in config.yaml. "
            f"Did you skip the Path A migration?"
        )
    if not profile.get("enabled", False):
        raise RuntimeError(f"{ETF_STRATEGY_ID} is disabled in config.yaml.")
    return profile


def get_latest_close(conn: sqlite3.Connection, symbol: str, asof: str) -> float | None:
    row = conn.execute(
        "SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
        (symbol, asof),
    ).fetchone()
    return float(row[0]) if row else None


def get_close_at_or_before(conn: sqlite3.Connection, symbol: str, target_date: str) -> float | None:
    return get_latest_close(conn, symbol, target_date)


def get_etf_positions(conn: sqlite3.Connection, asof: str) -> list[dict]:
    """Return {symbol, qty, avg_cost, market_price, market_value} for etf_buyhold."""
    rows = conn.execute(
        """
        SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl
        FROM positions
        WHERE strategy_id=? AND qty>0
          AND asof = (SELECT MAX(asof) FROM positions WHERE strategy_id=? AND qty>0)
        """,
        (ETF_STRATEGY_ID, ETF_STRATEGY_ID),
    ).fetchall()
    return [
        {
            "symbol": r[0], "qty": float(r[1]), "avg_cost": float(r[2] or 0),
            "market_price": float(r[3] or 0), "market_value": float(r[4] or 0),
            "unrealized_pnl": float(r[5] or 0),
        }
        for r in rows
    ]


def get_etf_nav(conn: sqlite3.Connection, asof: str) -> dict | None:
    """Latest account_snapshots entry for etf_buyhold (or None if not yet bootstrapped)."""
    row = conn.execute(
        """
        SELECT asof, cash, nav FROM account_snapshots
        WHERE strategy_id=? AND asof<=? ORDER BY asof DESC LIMIT 1
        """,
        (ETF_STRATEGY_ID, asof),
    ).fetchone()
    return {"asof": row[0], "cash": float(row[1]), "nav": float(row[2])} if row else None


def compute_rebalance_drift(positions: list[dict], targets: list[dict]) -> dict:
    """Return {symbol: {current_weight, target_weight, drift_pct}}.

    Codex 2026-04-28 review fix: include target symbols that are NOT yet in
    positions table (pre-bootstrap state) so user sees the gap explicitly.
    """
    total_value = sum(p["market_value"] for p in positions)
    target_map = {t["symbol"]: float(t["weight"]) for t in targets}
    pos_map = {p["symbol"]: p for p in positions}
    out = {}
    all_symbols = set(target_map.keys()) | set(pos_map.keys())
    for sym in sorted(all_symbols):
        p = pos_map.get(sym)
        cur_w = (p["market_value"] / total_value) if (p and total_value > 0) else 0.0
        tgt_w = target_map.get(sym, 0.0)
        drift = cur_w - tgt_w
        out[sym] = {
            "current_weight": cur_w,
            "target_weight": tgt_w,
            "drift_pct": drift,
            "abs_drift_pct": abs(drift),
            "in_positions": p is not None,
        }
    return out


def compute_monthly_return(conn: sqlite3.Connection, asof: str) -> dict:
    """Compare current NAV to NAV approx 1 month ago for etf_buyhold."""
    asof_dt = dt.datetime.strptime(asof, "%Y-%m-%d").date()
    one_mo_ago = asof_dt - dt.timedelta(days=30)
    row_now = conn.execute(
        "SELECT nav FROM account_snapshots WHERE strategy_id=? AND asof<=? ORDER BY asof DESC LIMIT 1",
        (ETF_STRATEGY_ID, asof),
    ).fetchone()
    row_then = conn.execute(
        "SELECT nav, asof FROM account_snapshots WHERE strategy_id=? AND asof<=? ORDER BY asof DESC LIMIT 1",
        (ETF_STRATEGY_ID, str(one_mo_ago)),
    ).fetchone()
    if not row_now or not row_then:
        return {"available": False}
    nav_now = float(row_now[0])
    nav_then = float(row_then[0])
    if nav_then <= 0:
        return {"available": False}
    pct_change = (nav_now - nav_then) / nav_then
    return {
        "available": True,
        "nav_now": nav_now,
        "nav_then": nav_then,
        "asof_then": row_then[1],
        "pct_change": pct_change,
    }


def compute_benchmark_return(conn: sqlite3.Connection, symbol: str, asof: str) -> dict:
    """Same window as compute_monthly_return but for a benchmark symbol."""
    asof_dt = dt.datetime.strptime(asof, "%Y-%m-%d").date()
    one_mo_ago = asof_dt - dt.timedelta(days=30)
    p_now = get_close_at_or_before(conn, symbol, asof)
    p_then = get_close_at_or_before(conn, symbol, str(one_mo_ago))
    if p_now is None or p_then is None or p_then <= 0:
        return {"available": False}
    return {
        "available": True,
        "p_now": p_now,
        "p_then": p_then,
        "pct_change": (p_now - p_then) / p_then,
    }


def render_markdown_report(
    *, asof: str, profile: dict, positions: list[dict], nav_info: dict | None,
    drift: dict, monthly: dict, bench: dict, rebalance_needed: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# ETF 月度报告 — {asof}")
    lines.append(f"_自动生成 by etf_monthly_check.py — {jst_now().strftime('%Y-%m-%d %H:%M JST')}_")
    lines.append("")
    lines.append("## 一、账户状态")
    if nav_info:
        lines.append(f"- 最新 NAV：**¥{nav_info['nav']:,.0f}**（asof {nav_info['asof']}）")
        lines.append(f"- 现金余额：¥{nav_info['cash']:,.0f}")
    else:
        lines.append(f"- ⚠️ etf_buyhold 账户尚未初始化（未找到 account_snapshots 记录）")
    lines.append("")
    lines.append("## 二、持仓明细")
    if positions:
        lines.append("| 代码 | 数量 | 成本 | 现价 | 市值 | 浮盈亏 |")
        lines.append("|------|------|------|------|------|--------|")
        for p in positions:
            lines.append(
                f"| {p['symbol']} | {p['qty']:.0f} | ¥{p['avg_cost']:.1f} | "
                f"¥{p['market_price']:.1f} | ¥{p['market_value']:,.0f} | "
                f"¥{p['unrealized_pnl']:+,.0f} |"
            )
    else:
        lines.append("- ⚠️ 无 etf_buyhold 持仓（如刚迁移完成尚未建仓属正常）")
    lines.append("")
    lines.append("## 三、收益率（30 日窗口）")
    if monthly.get("available"):
        lines.append(f"- ETF 组合：**{monthly['pct_change']*100:+.2f}%**（vs {monthly['asof_then']}）")
    else:
        lines.append("- 数据不足，无法计算")
    if bench.get("available"):
        lines.append(f"- 1321.T 基准：{bench['pct_change']*100:+.2f}%")
        if monthly.get("available"):
            excess = (monthly['pct_change'] - bench['pct_change']) * 100
            lines.append(f"- 超额：{excess:+.2f}%（vs 1321.T）")
    lines.append("")
    lines.append("## 四、再平衡检查")
    threshold = float(profile.get("rebalance", {}).get("drift_threshold", 0.05))
    if drift:
        lines.append(f"- 触发阈值：{threshold*100:.0f}% drift")
        lines.append("| 代码 | 当前权重 | 目标权重 | 偏离 |")
        lines.append("|------|---------|---------|------|")
        for sym, d in sorted(drift.items()):
            lines.append(
                f"| {sym} | {d['current_weight']*100:.1f}% | "
                f"{d['target_weight']*100:.1f}% | {d['drift_pct']*100:+.1f}% |"
            )
        lines.append("")
        if rebalance_needed:
            lines.append("**⚠️ 建议再平衡**：偏离已超阈值。")
        else:
            lines.append("**✅ 无需再平衡**：偏离在阈值内。")
    else:
        lines.append("- 无持仓，跳过")
    lines.append("")
    lines.append("## 五、心理检查 (HALT)")
    lines.append("")
    lines.append("> 你这个月有没有想过：")
    lines.append("> - `重新启用 sprint` （实测无 alpha，禁止）")
    lines.append("> - `试一下激进档` （实测 MaxDD -35%，禁止）")
    lines.append("> - `再加点单股` （违反 Path A 决议）")
    lines.append("> - `做 T 一下` （违反 Path A 决议）")
    lines.append(">")
    lines.append("> 如果有，重读 docs/STRATEGY_DECISION_2026-04-28.md")
    lines.append("> 解锁条件清单。任何冲动 → 看条件 → 不满足 → 不动。")
    lines.append("")
    lines.append("## 六、下次检查")
    lines.append(f"- 下次月报：{(jst_now() + dt.timedelta(days=30)).strftime('%Y-%m-01')}")
    return "\n".join(lines)


def send_email(subject: str, body_md: str, to_addr: str) -> bool:
    user = os.environ.get("EMAIL_USER", "").strip()
    password = os.environ.get("EMAIL_PASS", "").strip()
    if not user or not password:
        print("[etf_monthly_check] EMAIL_USER/EMAIL_PASS not set; skipping email.")
        return False
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body_md)
    msg.add_alternative(f"<pre>{body_md}</pre>", subtype="html")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as s:
            s.login(user, password)
            s.send_message(msg)
        print(f"[etf_monthly_check] Email sent to {to_addr}.")
        return True
    except Exception as e:
        print(f"[etf_monthly_check] Email failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asof", default=None, help="Override asof (default: today JST)")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--no_email", action="store_true", help="Skip email; write file only")
    args = ap.parse_args()

    asof = args.asof or jst_now().strftime("%Y-%m-%d")
    profile = load_etf_config(args.config)
    targets = profile.get("holdings", [])
    threshold = float(profile.get("rebalance", {}).get("drift_threshold", 0.05))
    to_email = (
        profile.get("monitoring", {}).get("report_email")
        or "lwyssq@gmail.com"
    )

    conn = sqlite3.connect(args.db)
    try:
        positions = get_etf_positions(conn, asof)
        nav_info = get_etf_nav(conn, asof)
        drift = compute_rebalance_drift(positions, targets)
        monthly = compute_monthly_return(conn, asof)
        bench = compute_benchmark_return(conn, "1321.T", asof)
        rebalance_needed = any(d["abs_drift_pct"] > threshold for d in drift.values())

        report_md = render_markdown_report(
            asof=asof, profile=profile, positions=positions, nav_info=nav_info,
            drift=drift, monthly=monthly, bench=bench, rebalance_needed=rebalance_needed,
        )

        out_dir = Path(args.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"etf_monthly_{asof}.md"
        out_path.write_text(report_md, encoding="utf-8")
        print(f"[etf_monthly_check] Report written: {out_path}")

        email_sent = True
        if not args.no_email:
            tag = "⚠️ REBALANCE" if rebalance_needed else "monthly"
            email_sent = send_email(
                subject=f"[ETF buy-hold] {tag} report {asof}",
                body_md=report_md,
                to_addr=to_email,
            )
    finally:
        conn.close()
    # Codex 2026-04-28 review: surface email failure to scheduler so monitoring
    # is not blind. Exit 2 if email was attempted but failed (report still on disk).
    if not args.no_email and not email_sent:
        print("[etf_monthly_check] Exit code 2: email send failed (report file written).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
