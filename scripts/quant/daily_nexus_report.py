import os
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

from paper_trader_bridge import (
    get_paper_db_path,
    load_paper_account,
    load_paper_config,
    sync_account_to_db,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_OPT_DIR = PROJECT_ROOT / "worker-quant" / "quant_trading" / "Project_optimized"
if str(PROJECT_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_OPT_DIR))

from trade_schema import connect, ensure_trade_tables


def _load_db_snapshot():
    db_path = get_paper_db_path()
    if not Path(db_path).exists():
        return None
    conn = connect(str(db_path))
    ensure_trade_tables(conn)
    try:
        row = conn.execute(
            """
            SELECT asof, cash, positions_value, nav, run_id
            FROM account_snapshots
            ORDER BY asof DESC, ts DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        asof, cash, positions_value, nav, run_id = row
        pos_rows = conn.execute(
            """
            SELECT symbol, qty, COALESCE(avg_cost,0), COALESCE(market_price,0), COALESCE(unrealized_pnl,0)
            FROM positions
            WHERE asof=?
            ORDER BY symbol
            """,
            (asof,),
        ).fetchall()
        return {
            "db_path": str(db_path),
            "asof": str(asof),
            "cash": float(cash or 0.0),
            "positions_value": float(positions_value or 0.0),
            "nav": float(nav or 0.0),
            "run_id": run_id,
            "positions": [
                {
                    "symbol": symbol,
                    "qty": float(qty),
                    "avg": float(avg_cost or 0.0),
                    "current": float(market_price or 0.0),
                    "pnl": float(unrealized_pnl or 0.0),
                    "pnl_pct": ((float(market_price or 0.0) / float(avg_cost) - 1) * 100) if avg_cost else 0.0,
                }
                for symbol, qty, avg_cost, market_price, unrealized_pnl in pos_rows
            ],
        }
    finally:
        conn.close()


def _load_snapshot_history(limit=120):
    db_path = get_paper_db_path()
    if not Path(db_path).exists():
        return []
    conn = connect(str(db_path))
    ensure_trade_tables(conn)
    try:
        rows = conn.execute(
            """
            SELECT asof, nav
            FROM account_snapshots
            WHERE nav IS NOT NULL
            ORDER BY asof DESC, ts DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        dedup = {}
        for asof, nav in rows:
            dedup.setdefault(str(asof), float(nav or 0.0))
        return [{"asof": k, "nav": dedup[k]} for k in sorted(dedup)]
    finally:
        conn.close()


def _weekly_compound_rate(annual_return_pct):
    annual_multiple = 1.0 + float(annual_return_pct) / 100.0
    return annual_multiple ** (1.0 / 52.0) - 1.0


def _window_target_return(weekly_rate, weeks):
    return (1.0 + weekly_rate) ** int(weeks) - 1.0


def _rolling_return(history, weeks):
    if len(history) < weeks + 1:
        return None
    end_nav = float(history[-1]["nav"])
    start_nav = float(history[-(weeks + 1)]["nav"])
    if start_nav <= 0:
        return None
    return end_nav / start_nav - 1.0


def _status_label(actual, floor_target, target):
    if actual is None:
        return "N/A"
    if actual >= target:
        return "达标"
    if actual >= floor_target:
        return "达保底"
    return "未达标"


def _daily_trade_summary(account, current_total, initial, weekly_floor, weekly_target):
    history = list(account.get("trade_history") or [])
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in history if str(t.get("date", "")).startswith(today)]
    buys = [t for t in today_trades if str(t.get("action", "")).upper() == "BUY"]
    sells = [t for t in today_trades if str(t.get("action", "")).upper() == "SELL"]
    turnover = sum(float(t.get("total", 0.0) or 0.0) for t in today_trades)
    total_return_pct = (current_total / initial - 1.0) * 100 if initial else 0.0

    lines = []
    if today_trades:
        buy_syms = ", ".join(f"{t['symbol']}x{int(t['qty'])}" for t in buys) if buys else "无"
        sell_syms = ", ".join(f"{t['symbol']}x{int(t['qty'])}" for t in sells) if sells else "无"
        lines.append(f"- 当日成交: {len(today_trades)} 笔 | 买入: {buy_syms} | 卖出: {sell_syms}")
        lines.append(f"- 当日成交额: {turnover:,.0f} JPY")
    else:
        lines.append("- 当日成交: 无新增成交，维持现有仓位。")

    if total_return_pct >= weekly_target:
        lines.append("- 目标状态: 已达到本周标准收益目标，优先控制回撤并锁定利润。")
    elif total_return_pct >= weekly_floor:
        lines.append("- 目标状态: 已达到本周保底目标，但仍低于标准目标。")
    else:
        lines.append("- 目标状态: 低于本周保底目标，优先复盘和检查信号质量，不追收益。")
    return lines


def _position_summary(pos_details, cash_balance, current_total):
    invest_ratio = 0.0 if current_total <= 0 else (current_total - cash_balance) / current_total * 100.0
    best = max(pos_details, key=lambda x: x["pnl_pct"], default=None)
    worst = min(pos_details, key=lambda x: x["pnl_pct"], default=None)
    lines = [f"- 当前权益仓位: {invest_ratio:.1f}%"]
    if best is not None:
        lines.append(f"- 最强持仓: {best['symbol']} {best['pnl_pct']:+.2f}%")
    if worst is not None:
        lines.append(f"- 最弱持仓: {worst['symbol']} {worst['pnl_pct']:+.2f}%")
    return lines


def _next_step_lines(cash_balance, current_total):
    invest_ratio = 0.0 if current_total <= 0 else (current_total - cash_balance) / current_total * 100.0
    lines = [f"- 当前权益仓位: {invest_ratio:.1f}% | 可用现金: {cash_balance:,.0f} JPY"]
    if invest_ratio <= 1e-9:
        lines.append("- 当前正式量化目标为空仓，后续以等待新鲜可执行信号为主。")
    else:
        lines.append("- 若滚动收益低于保底目标，优先触发复盘与风控检查，而不是被动追收益加仓。")
    return lines


def _pending_order_lines(account):
    pending = list(account.get("pending_orders") or [])
    if not pending:
        return ["- 待执行订单: 无"]
    lines = [f"- 待执行订单: {len(pending)} 笔"]
    for order in pending[:10]:
        lines.append(
            f"- {order.get('created_at', '')} | {order.get('action', '')} "
            f"{int(order.get('qty', 0))} {order.get('symbol', '')} | {order.get('status', '')}"
        )
    return lines


def generate_daily_report():
    if not os.path.exists("paper_trading_account.json"):
        return "未找到 paper_trading_account.json"

    account = load_paper_account()
    config = load_paper_config()
    goals = config.get("goals", {})

    print("正在同步正式账本并拉取估值...")
    sync_account_to_db(account)
    account = load_paper_account()
    db_snapshot = _load_db_snapshot()
    history = _load_snapshot_history()

    pos_details = []
    if db_snapshot:
        current_total = db_snapshot["nav"]
        cash_balance = db_snapshot["cash"]
        initial = float(account["account_info"]["initial_capital"])
        total_return_pct = (current_total / initial - 1) * 100
        for p in db_snapshot["positions"]:
            pos_details.append(
                {
                    "symbol": p["symbol"],
                    "qty": p["qty"],
                    "avg": p["avg"],
                    "current": round(p["current"], 2),
                    "pnl": round(p["pnl"], 0),
                    "pnl_pct": round(p["pnl_pct"], 2),
                }
            )
        report_date = db_snapshot["asof"]
        data_source_line = f"- 数据来源: 正式账本 {db_snapshot['db_path']} | run_id: {db_snapshot['run_id']}"
    else:
        print("正式账本不可用，回退到 JSON 账户估值。")
        total_market_value = 0.0
        for symbol, pos in account["positions"].items():
            ticker = yf.Ticker(symbol)
            current_price = ticker.fast_info.last_price
            val = current_price * pos["qty"]
            pnl = (current_price - pos["avg_price"]) * pos["qty"]
            pnl_pct = (current_price / pos["avg_price"] - 1) * 100
            total_market_value += val
            pos_details.append(
                {
                    "symbol": symbol,
                    "qty": pos["qty"],
                    "avg": pos["avg_price"],
                    "current": round(current_price, 2),
                    "pnl": round(pnl, 0),
                    "pnl_pct": round(pnl_pct, 2),
                }
            )
        current_total = total_market_value + account["account_info"]["current_cash"]
        cash_balance = account["account_info"]["current_cash"]
        initial = account["account_info"]["initial_capital"]
        total_return_pct = (current_total / initial - 1) * 100
        report_date = datetime.now().strftime("%Y-%m-%d")
        data_source_line = "- 数据来源: JSON 沙盘账户（回退模式）"

    weekly_floor = float(goals.get("weekly_floor_return_pct", 1.0))
    weekly_target = float(goals.get("weekly_target_return_pct", 2.0))
    annual_low = float(goals.get("annual_target_return_low_pct", 300.0))
    annual_high = float(goals.get("annual_target_return_high_pct", 400.0))
    rolling_windows = list(goals.get("rolling_windows_weeks", [2, 4, 8]))
    progress = (total_return_pct / weekly_target) * 100 if weekly_target else 0.0
    annual_low_weekly = _weekly_compound_rate(annual_low)
    annual_high_weekly = _weekly_compound_rate(annual_high)

    rolling_lines = []
    for weeks in rolling_windows:
        actual = _rolling_return(history, int(weeks))
        floor_target = _window_target_return(weekly_floor / 100.0, int(weeks))
        target = _window_target_return(weekly_target / 100.0, int(weeks))
        annual_low_target = _window_target_return(annual_low_weekly, int(weeks))
        annual_high_target = _window_target_return(annual_high_weekly, int(weeks))
        actual_text = "N/A" if actual is None else f"{actual * 100:+.2f}%"
        status = _status_label(actual, floor_target, target)
        rolling_lines.append(
            f"- rolling_{weeks}w: 实际 {actual_text} | 保底 {floor_target * 100:.2f}% | "
            f"目标 {target * 100:.2f}% | 复利轨道 {annual_low_target * 100:.2f}%~{annual_high_target * 100:.2f}% | {status}"
        )

    trade_summary_lines = _daily_trade_summary(account, current_total, initial, weekly_floor, weekly_target)
    position_summary_lines = _position_summary(pos_details, cash_balance, current_total)
    next_step_lines = _next_step_lines(cash_balance, current_total)
    pending_order_lines = _pending_order_lines(account)

    report_lines = [
        "==========================================",
        f" NEXUS QUANT 每日交易及收益分析报告",
        f" 日期: {report_date}",
        "==========================================",
        "",
        "[账户概览]",
        f"- 初始资金: {initial:,.0f} JPY",
        f"- 当前总资产: {current_total:,.0f} JPY",
        f"- 现金余额: {cash_balance:,.0f} JPY",
        f"- 总收益率: {total_return_pct:+.2f}%",
        f"- 周保底收益目标: {weekly_floor:.2f}%",
        f"- 周标准收益目标: {weekly_target:.2f}%",
        f"- 相对周标准目标完成度: {progress:.1f}%",
        f"- 年化复利轨道: {annual_low:.0f}% ~ {annual_high:.0f}%",
        f"- 对应周复利基线: {annual_low_weekly * 100:.2f}% ~ {annual_high_weekly * 100:.2f}%",
        data_source_line,
        "",
        "[持仓明细]",
    ]

    if pos_details:
        for p in pos_details:
            report_lines.append(
                f"- {p['symbol']}: {int(p['qty'])}股 | 现价: {p['current']} | "
                f"盈亏: {p['pnl']:+,.0f} ({p['pnl_pct']:+.2f}%)"
            )
    else:
        report_lines.append("- 当前无持仓")

    report_lines.extend(["", "[滚动目标跟踪]"])
    report_lines.extend(rolling_lines)
    report_lines.extend(["", "[今日量化分析与交易总结]"])
    report_lines.extend(trade_summary_lines)
    report_lines.extend(position_summary_lines)
    report_lines.extend(["", "[后续策略]"])
    report_lines.extend(next_step_lines)
    report_lines.extend(["", "[待执行订单]"])
    report_lines.extend(pending_order_lines)
    report_lines.append("==========================================")
    return "\n".join(report_lines) + "\n"


if __name__ == "__main__":
    report = generate_daily_report()
    print(report)
    with open("LATEST_TRADING_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
