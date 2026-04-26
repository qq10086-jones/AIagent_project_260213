"""Build daily action plan from latest decision artifacts + current positions."""
from __future__ import annotations
import argparse, json, csv, sys
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

def _now_jst():
    return datetime.now(TOKYO_TZ)

def _load_latest_orders_proposal(reports_dir: Path, artifact_root: Path, asof: str) -> list[dict]:
    """Find latest orders_proposal.csv from decision artifacts."""
    # Try artifact dir first
    candidates = list(artifact_root.glob(f"{asof}/*/orders_proposal.csv")) if artifact_root.exists() else []
    if candidates:
        # Sort by modification time (not path) to get the truly latest run
        candidates.sort(key=lambda p: p.stat().st_mtime)
        path = candidates[-1]  # latest run by mtime
    else:
        # Fallback: reports dir
        path = reports_dir / "orders_proposal.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "symbol": row.get("symbol", "").strip(),
                "side": row.get("side", "").strip().upper(),
                "qty": float(row.get("qty", 0)),
                "suggested_type": row.get("suggested_type", "MKT"),
                "suggested_limit": float(row["suggested_limit"]) if row.get("suggested_limit") else None,
                "est_notional": float(row.get("est_notional", 0)),
                "comment": row.get("comment", ""),
            })
    return rows

def _load_positions(conn, asof: str, strategy_id: str) -> list[dict]:
    """Load current positions with stop/take-profit from latest asof <= given date."""
    row = conn.execute(
        "SELECT MAX(asof) FROM positions WHERE asof<=? AND strategy_id=?",
        (asof, strategy_id)
    ).fetchone()
    pos_asof = row[0] if row and row[0] else asof
    rows = conn.execute(
        """SELECT symbol, qty, avg_cost, market_price, market_value, unrealized_pnl,
                  high_since_entry, entry_date
           FROM positions WHERE asof=? AND strategy_id=? AND qty > 0""",
        (pos_asof, strategy_id)
    ).fetchall()
    positions = []
    for r in rows:
        positions.append({
            "symbol": r[0], "qty": float(r[1]), "avg_cost": float(r[2] or 0),
            "market_price": float(r[3] or 0), "market_value": float(r[4] or 0),
            "unrealized_pnl": float(r[5] or 0),
            "high_since_entry": float(r[6] or 0), "entry_date": r[7] or "",
        })
    return positions

def _load_regime(reports_dir: Path) -> dict:
    """Load latest regime diagnosis."""
    path = reports_dir / "regime_diagnosis.json"
    if not path.exists():
        return {"sprint_final_state": "unknown"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"sprint_final_state": "unknown"}

def _load_stop_loss_info(artifact_root: Path, asof: str) -> list[dict]:
    """Load stop-loss diagnostics from latest decision snapshot."""
    candidates = list(artifact_root.glob(f"{asof}/*/decision_snapshot.json")) if artifact_root.exists() else []
    if not candidates:
        return []
    candidates.sort(key=lambda p: p.stat().st_mtime)
    try:
        snap = json.loads(candidates[-1].read_text(encoding="utf-8"))
        return snap.get("risk_management", {}).get("stop_loss_diagnostics", [])
    except Exception:
        return []

def build_action_plan(
    conn,
    asof: str,
    strategy_id: str = "sprint",
    reports_dir: Path = Path("reports"),
    artifact_root: Path = Path("artifacts/decision"),
) -> dict:
    """Build comprehensive action plan for today."""
    now = _now_jst()

    # Load all data sources
    orders = _load_latest_orders_proposal(reports_dir, artifact_root, asof)
    positions = _load_positions(conn, asof, strategy_id)
    regime = _load_regime(reports_dir)
    stop_diags = _load_stop_loss_info(artifact_root, asof)

    # Classify pending actions
    pending_sells = [o for o in orders if o["side"] == "SELL"]
    pending_buys = [o for o in orders if o["side"] == "BUY"]

    # 2026-04-25 (Codex #3): current_status must reconcile with pending
    # orders. Previously a symbol could appear as HOLD while simultaneously
    # showing up in pending_sells — downstream readers then got contradictory
    # guidance. Mark symbols with pending SELLs as SELL_PENDING so the JSON
    # is internally consistent.
    pending_sell_syms = {o["symbol"] for o in pending_sells}

    # Enrich positions with stop-loss info
    stop_map = {d["symbol"]: d for d in stop_diags}
    for pos in positions:
        sd = stop_map.get(pos["symbol"], {})
        pos["stop_loss_price"] = sd.get("stop_loss_price")
        pos["stop_loss_pct"] = sd.get("stop_loss_pct_effective")
        pos["stop_triggered"] = sd.get("triggered", False)
        if sd.get("triggered"):
            pos["current_status"] = "STOP_LOSS"
        elif pos["symbol"] in pending_sell_syms:
            pos["current_status"] = "SELL_PENDING"
        else:
            pos["current_status"] = "HOLD"

    # Risk alerts
    risk_alerts = []
    for pos in positions:
        if pos.get("stop_triggered"):
            risk_alerts.append({
                "type": "stop_loss_triggered",
                "symbol": pos["symbol"],
                "urgency": "CRITICAL",
                "message": f"{pos['symbol']} has hit stop-loss. Sell immediately.",
            })

    # Build action summary
    summary_parts = []
    if pending_sells:
        syms = ", ".join(s["symbol"] for s in pending_sells)
        summary_parts.append(f"SELL: {syms}")
    if pending_buys:
        syms = ", ".join(b["symbol"] for b in pending_buys)
        summary_parts.append(f"BUY: {syms}")
    if not pending_sells and not pending_buys:
        summary_parts.append("No trades required")
    sell_syms = {s["symbol"] for s in pending_sells}
    held_syms = [p["symbol"] for p in positions if p["current_status"] == "HOLD" and p["symbol"] not in sell_syms]
    if held_syms:
        summary_parts.append(f"HOLD: {', '.join(held_syms)}")
    regime_state = regime.get("sprint_final_state", "unknown")
    summary_parts.append(f"Regime: {regime_state}")

    plan = {
        "asof": asof,
        "generated_at": now.isoformat(),
        "strategy_id": strategy_id,
        "regime": regime_state,
        "regime_detail": {
            "ma_gap_pct": round((regime.get("fast_ma", 0) - regime.get("slow_ma", 1)) / max(regime.get("slow_ma", 1), 1) * 100, 2) if regime.get("slow_ma") else None,
            "enter_line": regime.get("enter_line"),
            "fast_ma": regime.get("fast_ma"),
            "slow_ma": regime.get("slow_ma"),
        },
        "pending_sells": pending_sells,
        "pending_buys": pending_buys,
        "held_positions": positions,
        "risk_alerts": risk_alerts,
        "action_summary": " | ".join(summary_parts),
    }

    # Write to file
    out_path = reports_dir / "action_plan_today.json"
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--asof", default=datetime.now(TOKYO_TZ).strftime("%Y-%m-%d"))
    ap.add_argument("--strategy_id", default="sprint")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--artifact_root", default="artifacts/decision")
    args = ap.parse_args()

    conn = connect(args.db)
    ensure_trade_tables(conn)
    try:
        plan = build_action_plan(
            conn, args.asof, args.strategy_id,
            Path(args.reports_dir), Path(args.artifact_root),
        )
        print(f"Action plan generated: {plan['action_summary']}")
        print(f"  Pending sells: {len(plan['pending_sells'])}")
        print(f"  Pending buys: {len(plan['pending_buys'])}")
        print(f"  Held positions: {len(plan['held_positions'])}")
        print(f"  Risk alerts: {len(plan['risk_alerts'])}")
        print(f"  Output: reports/action_plan_today.json")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
