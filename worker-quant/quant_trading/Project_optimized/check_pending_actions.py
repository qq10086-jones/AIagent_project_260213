"""Check if action plan items have been executed; emit alerts for pending items."""
from __future__ import annotations
import argparse, json, sys, os, yaml
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from trade_schema import connect, ensure_trade_tables

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TOKYO_TZ = ZoneInfo("Asia/Tokyo")

# Import alert infra from daily_run
def _import_alert_infra():
    """Import alert functions from daily_run.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("daily_run", Path(__file__).parent / "daily_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_action_plan(reports_dir: Path) -> dict:
    path = reports_dir / "action_plan_today.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _get_executed_fills(conn, asof: str, strategy_id: str) -> list[dict]:
    """Get today's fills."""
    rows = conn.execute(
        "SELECT symbol, side, qty, price, ts FROM fills WHERE asof=? AND strategy_id=?",
        (asof, strategy_id)
    ).fetchall()
    return [{"symbol": r[0], "side": r[1].upper(), "qty": float(r[2]), "price": float(r[3]), "ts": r[4]} for r in rows]

def _get_current_price_from_db(conn, symbol: str, asof: str) -> float | None:
    """Get latest close price."""
    row = conn.execute(
        "SELECT close FROM daily_prices WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 1",
        (symbol, asof)
    ).fetchone()
    return float(row[0]) if row else None

def check_pending_actions(
    conn,
    asof: str,
    strategy_id: str = "sprint",
    reports_dir: Path = Path("reports"),
    alert_level: str = "info",
    check_label: str = "09:30",
) -> list[dict]:
    """Compare action plan vs fills, return list of unexecuted items."""
    plan = _load_action_plan(reports_dir)
    if not plan:
        return []

    fills = _get_executed_fills(conn, asof, strategy_id)
    fill_set = {(f["symbol"], f["side"]) for f in fills}

    pending = []

    # Check sells
    for sell in plan.get("pending_sells", []):
        sym = sell["symbol"]
        if (sym, "SELL") not in fill_set:
            current_price = _get_current_price_from_db(conn, sym, asof)
            signal_price = sell.get("est_notional", 0) / max(sell.get("qty", 1), 1) if sell.get("qty") else None
            est_loss = None
            if current_price and signal_price and sell.get("qty"):
                est_loss = round((current_price - signal_price) * sell["qty"], 0)
            pending.append({
                "type": "pending_sell",
                "symbol": sym,
                "qty": sell.get("qty"),
                "signal_price": round(signal_price, 1) if signal_price else None,
                "current_price": current_price,
                "estimated_extra_loss": est_loss,
                "urgency": "HIGH",
                "check_time": check_label,
            })

    # Check buys
    for buy in plan.get("pending_buys", []):
        sym = buy["symbol"]
        if (sym, "BUY") not in fill_set:
            current_price = _get_current_price_from_db(conn, sym, asof)
            pending.append({
                "type": "pending_buy",
                "symbol": sym,
                "qty": buy.get("qty"),
                "current_price": current_price,
                "urgency": "MEDIUM",
                "check_time": check_label,
            })

    return pending

def run_check_and_alert(
    conn,
    asof: str,
    strategy_id: str,
    reports_dir: Path,
    alert_level: str = "warning",
    check_label: str = "09:30",
    cfg: dict | None = None,
):
    """Run check and emit alerts for pending items."""
    # Configure alerts
    if cfg:
        from daily_run import configure_alert_env, emit_runtime_event
        configure_alert_env(cfg)
    else:
        from daily_run import emit_runtime_event

    pending = check_pending_actions(conn, asof, strategy_id, reports_dir, alert_level, check_label)

    if not pending:
        print(f"[{check_label}] All action plan items executed. No alerts.")
        emit_runtime_event(
            reports_dir, "pending_action_check",
            level="info",
            check_time=check_label,
            pending_count=0,
            status="all_executed",
        )
        return pending

    # Emit alert for each pending item
    for item in pending:
        level = alert_level
        if check_label == "14:00":
            level = "error"  # CRITICAL at pre-close

        msg_parts = [f"[{check_label}] {item['symbol']} {item['type'].replace('pending_', '').upper()} not executed"]
        if item.get("current_price"):
            msg_parts.append(f"Current: {item['current_price']}")
        if item.get("signal_price"):
            msg_parts.append(f"Signal: {item['signal_price']}")
        if item.get("estimated_extra_loss") and item["estimated_extra_loss"] != 0:
            msg_parts.append(f"Est extra loss: {item['estimated_extra_loss']:+,.0f} JPY")

        if check_label == "14:00":
            msg_parts.append("LAST EXECUTION WINDOW - signal expires at close")

        emit_runtime_event(
            reports_dir, "pending_action_alert",
            level=level,
            **item,
            message=" | ".join(msg_parts),
        )
        print(" | ".join(msg_parts))

    return pending

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=datetime.now(TOKYO_TZ).strftime("%Y-%m-%d"))
    ap.add_argument("--strategy_id", default="sprint")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--check_label", default="09:30", help="Time label: 09:30, 11:30, 14:00")
    ap.add_argument("--alert_level", default="warning", help="info, warning, error")
    args = ap.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        run_check_and_alert(
            conn, args.asof, args.strategy_id,
            Path(args.reports_dir),
            alert_level=args.alert_level,
            check_label=args.check_label,
            cfg=cfg,
        )
    finally:
        conn.close()

if __name__ == "__main__":
    main()
