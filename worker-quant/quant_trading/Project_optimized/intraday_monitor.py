"""Intraday monitoring: stop-loss checks, pending action escalation, regime updates."""
from __future__ import annotations
import argparse, json, sys, yaml
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

def _load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _get_positions(conn, asof: str, strategy_id: str) -> list[dict]:
    row = conn.execute(
        "SELECT MAX(asof) FROM positions WHERE asof<=? AND strategy_id=?",
        (asof, strategy_id)
    ).fetchone()
    pos_asof = row[0] if row and row[0] else asof
    rows = conn.execute(
        """SELECT symbol, qty, avg_cost, market_price, high_since_entry
           FROM positions WHERE asof=? AND strategy_id=? AND qty > 0""",
        (pos_asof, strategy_id)
    ).fetchall()
    return [{"symbol": r[0], "qty": float(r[1]), "avg_cost": float(r[2] or 0),
             "market_price": float(r[3] or 0), "high_since_entry": float(r[4] or 0)} for r in rows]

def _fetch_live_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch near-realtime prices via yfinance (15min delay on free tier)."""
    if not symbols:
        return {}
    try:
        import yfinance as yf
        data = yf.download(symbols if len(symbols) > 1 else symbols[0],
                           period="1d", interval="1m", progress=False, prepost=False)
        prices = {}
        if len(symbols) == 1:
            if not data.empty:
                _ser = data["Close"].dropna()
                if len(_ser) > 0:
                    prices[symbols[0]] = float(_ser.iloc[-1])
        else:
            for sym in symbols:
                try:
                    col = data["Close"][sym] if "Close" in data.columns.get_level_values(0) else data[sym]["Close"]
                    val = col.dropna()
                    if len(val) > 0:
                        prices[sym] = float(val.iloc[-1])
                except Exception:
                    pass
        return prices
    except Exception as e:
        print(f"[WARN] Failed to fetch live prices: {e}")
        return {}

def check_stop_loss(positions: list[dict], live_prices: dict, config: dict) -> list[dict]:
    """Check positions against stop-loss levels."""
    sp = config.get("strategy_profiles", {}).get("sprint", {})
    vol_mult = sp.get("stop_loss_vol_mult", 3.0)
    min_pct = sp.get("stop_loss_min_pct", 0.04)
    max_pct = sp.get("stop_loss_max_pct", 0.12)
    trailing_activate = sp.get("trailing_activate_pct", 0.03)
    trailing_stop = sp.get("trailing_stop_pct", 0.02)

    alerts = []
    for pos in positions:
        sym = pos["symbol"]
        avg_cost = pos["avg_cost"]
        current = live_prices.get(sym, pos["market_price"])
        high = max(pos["high_since_entry"], current)

        if avg_cost <= 0:
            continue

        # Simple stop-loss check (use max_pct as conservative estimate without ATR)
        stop_pct = max_pct  # conservative; ATR calculation needs daily_prices
        stop_price = avg_cost * (1 - stop_pct)

        if current <= stop_price:
            alerts.append({
                "type": "stop_loss_triggered",
                "symbol": sym,
                "avg_cost": avg_cost,
                "current_price": current,
                "stop_price": round(stop_price, 1),
                "loss_pct": round((current - avg_cost) / avg_cost * 100, 2),
                "urgency": "CRITICAL",
            })
            continue

        # Trailing protect check
        if high >= avg_cost * (1 + trailing_activate):
            trail_price = high * (1 - trailing_stop)
            if current < trail_price:
                alerts.append({
                    "type": "trailing_protect_triggered",
                    "symbol": sym,
                    "avg_cost": avg_cost,
                    "current_price": current,
                    "high_since_entry": round(high, 1),
                    "trail_price": round(trail_price, 1),
                    "urgency": "HIGH",
                })
                continue

        # Position status (no trigger)
        pnl_pct = (current - avg_cost) / avg_cost * 100
        alerts.append({
            "type": "position_status",
            "symbol": sym,
            "avg_cost": avg_cost,
            "current_price": current,
            "pnl_pct": round(pnl_pct, 2),
            "stop_price": round(stop_price, 1),
            "distance_to_stop_pct": round((current - stop_price) / current * 100, 2),
            "urgency": "INFO",
        })

    return alerts

def run_monitor(
    mode: str,
    db_path: str = "japan_market.db",
    config_path: str = "config.yaml",
    asof: str | None = None,
    strategy_id: str = "sprint",
    reports_dir_str: str = "reports",
    fetch_live: bool = True,
):
    """Run intraday monitoring in specified mode."""
    cfg = _load_config(config_path)
    reports_dir = Path(reports_dir_str)

    from daily_run import configure_alert_env, emit_runtime_event
    configure_alert_env(cfg)

    if asof is None:
        asof = datetime.now(TOKYO_TZ).strftime("%Y-%m-%d")

    conn = connect(db_path)
    ensure_trade_tables(conn)

    try:
        positions = _get_positions(conn, asof, strategy_id)
        if not positions:
            print(f"[{mode}] No positions to monitor.")
            return

        # Fetch live prices
        symbols = [p["symbol"] for p in positions]
        live_prices = _fetch_live_prices(symbols) if fetch_live else {}

        # 1. Stop-loss / trailing checks
        sl_alerts = check_stop_loss(positions, live_prices, cfg)
        triggered = [a for a in sl_alerts if a["urgency"] in ("CRITICAL", "HIGH")]

        for alert in triggered:
            emit_runtime_event(
                reports_dir, f"intraday_{alert['type']}",
                level="error" if alert["urgency"] == "CRITICAL" else "warning",
                mode=mode,
                **{k: v for k, v in alert.items() if k != "type"},
            )
            print(f"[{mode}] {alert['urgency']}: {alert['symbol']} {alert['type']} @ {alert.get('current_price')}")

        # 2. Pending action escalation
        from check_pending_actions import run_check_and_alert
        label_map = {"midday": "11:30", "pre_close": "14:00", "open_watch": "09:00"}
        check_label = label_map.get(mode, mode)
        level_map = {"midday": "warning", "pre_close": "error", "open_watch": "info"}
        alert_level = level_map.get(mode, "warning")

        run_check_and_alert(
            conn, asof, strategy_id, reports_dir,
            alert_level=alert_level,
            check_label=check_label,
            cfg=cfg,
        )

        # 3. Summary
        status_items = [a for a in sl_alerts if a["urgency"] == "INFO"]
        for s in status_items:
            print(f"  {s['symbol']}: {s.get('pnl_pct', 0):+.2f}% | stop distance: {s.get('distance_to_stop_pct', 0):.1f}%")

        # Write monitor snapshot
        snapshot = {
            "mode": mode,
            "asof": asof,
            "checked_at": datetime.now(TOKYO_TZ).isoformat(),
            "positions_checked": len(positions),
            "stop_loss_triggered": len([a for a in sl_alerts if a["type"] == "stop_loss_triggered"]),
            "trailing_triggered": len([a for a in sl_alerts if a["type"] == "trailing_protect_triggered"]),
            "live_prices_fetched": bool(live_prices),
            "alerts": sl_alerts,
        }
        (reports_dir / "intraday_monitor_latest.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    finally:
        conn.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["open_watch", "midday", "pre_close"],
                    help="Monitor mode: open_watch (09:00), midday (11:30), pre_close (14:00)")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--strategy_id", default="sprint")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--no-live", action="store_true", help="Skip live price fetch (use DB prices)")
    args = ap.parse_args()

    run_monitor(
        mode=args.mode,
        db_path=args.db,
        config_path=args.config,
        asof=args.asof,
        strategy_id=args.strategy_id,
        reports_dir_str=args.reports_dir,
        fetch_live=not args.no_live,
    )

if __name__ == "__main__":
    main()
