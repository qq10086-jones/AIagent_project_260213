"""Decision journal and compliance tracking for execution discipline."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from trade_schema import connect, ensure_trade_tables

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TOKYO_TZ = ZoneInfo("Asia/Tokyo")


def ensure_journal_table(conn) -> None:
    """Create decision_journal table if missing. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_journal (
          journal_id TEXT PRIMARY KEY,
          asof TEXT NOT NULL,
          ts TEXT NOT NULL,
          strategy_id TEXT DEFAULT 'sprint',
          action_type TEXT NOT NULL,
          model_signal TEXT,
          actual_action TEXT NOT NULL,
          override_reason TEXT,
          outcome_pnl REAL,
          outcome_filled_at TEXT,
          compliance_score REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_asof ON decision_journal(asof);")


def _load_orders_proposal(artifact_root: Path, asof: str) -> list[dict]:
    """Load orders_proposal.csv from latest decision artifact."""
    candidates = sorted(artifact_root.glob(f"{asof}/*/orders_proposal.csv")) if artifact_root.exists() else []
    if not candidates:
        return []
    rows = []
    with open(candidates[-1], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "symbol": row.get("symbol", "").strip(),
                "side": row.get("side", "").strip().upper(),
                "qty": float(row.get("qty", 0)),
            })
    return rows


def _get_fills(conn, asof: str, strategy_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT symbol, side, qty, price FROM fills WHERE asof=? AND strategy_id=?",
        (asof, strategy_id),
    ).fetchall()
    return [{"symbol": r[0], "side": r[1].upper(), "qty": float(r[2]), "price": float(r[3])} for r in rows]


def check_model_deviation(
    conn,
    asof: str,
    strategy_id: str = "sprint",
    artifact_root: Path = Path("artifacts/decision"),
) -> list[dict]:
    """Compare model orders_proposal vs actual fills. Return list of deviations."""
    proposals = _load_orders_proposal(artifact_root, asof)
    fills = _get_fills(conn, asof, strategy_id)

    fill_set = {(f["symbol"], f["side"]) for f in fills}
    proposal_set = {(p["symbol"], p["side"]) for p in proposals}

    deviations = []

    # Model said to trade but user didn't
    for p in proposals:
        key = (p["symbol"], p["side"])
        if key not in fill_set:
            deviations.append({
                "symbol": p["symbol"],
                "deviation": "model_signal_not_executed",
                "model_side": p["side"],
                "model_qty": p["qty"],
                "actual_side": None,
                "actual_qty": 0,
            })

    # User traded something model didn't suggest
    for f in fills:
        key = (f["symbol"], f["side"])
        if key not in proposal_set:
            deviations.append({
                "symbol": f["symbol"],
                "deviation": "unplanned_trade",
                "model_side": None,
                "model_qty": 0,
                "actual_side": f["side"],
                "actual_qty": f["qty"],
                "actual_price": f["price"],
            })

    return deviations


def record_action(
    conn,
    asof: str,
    strategy_id: str,
    model_signal: Optional[dict],
    actual_action: dict,
    override_reason: Optional[str] = None,
) -> str:
    """Write a single entry to decision_journal. Returns journal_id."""
    ensure_journal_table(conn)
    ts = datetime.now(TOKYO_TZ).isoformat(timespec="seconds")
    raw = f"{asof}{ts}{json.dumps(actual_action, sort_keys=True)}"
    journal_id = hashlib.md5(raw.encode()).hexdigest()[:16]

    # Determine action_type
    if model_signal is None:
        action_type = "manual_entry"
    elif override_reason:
        action_type = "model_override"
    else:
        action_type = "model_follow"

    conn.execute(
        """
        INSERT OR REPLACE INTO decision_journal
        (journal_id, asof, ts, strategy_id, action_type, model_signal, actual_action, override_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            journal_id,
            asof,
            ts,
            strategy_id,
            action_type,
            json.dumps(model_signal, ensure_ascii=False) if model_signal else None,
            json.dumps(actual_action, ensure_ascii=False),
            override_reason,
        ),
    )
    conn.commit()
    return journal_id


def compute_compliance_score(conn, asof_from: str, asof_to: str) -> dict:
    """Compute compliance metrics over a date range."""
    ensure_journal_table(conn)
    rows = conn.execute(
        "SELECT action_type FROM decision_journal WHERE asof>=? AND asof<=?",
        (asof_from, asof_to),
    ).fetchall()
    if not rows:
        return {"total": 0, "follow": 0, "override": 0, "manual": 0, "compliance_rate": None}
    total = len(rows)
    follow = sum(1 for r in rows if r[0] == "model_follow")
    override = sum(1 for r in rows if r[0] == "model_override")
    manual = sum(1 for r in rows if r[0] == "manual_entry")
    rate = follow / max(follow + override, 1)
    return {
        "total": total,
        "follow": follow,
        "override": override,
        "manual": manual,
        "compliance_rate": round(rate, 4),
    }


def record_daily_compliance(
    conn,
    asof: str,
    strategy_id: str = "sprint",
    artifact_root: Path = Path("artifacts/decision"),
) -> dict:
    """End-of-day: record all deviations into journal. Returns summary.
    Idempotent — skips if entries already exist for this asof+strategy_id."""
    ensure_journal_table(conn)

    # Idempotency guard: skip if already recorded today
    existing = conn.execute(
        "SELECT COUNT(*) FROM decision_journal WHERE asof=? AND strategy_id=?",
        (asof, strategy_id),
    ).fetchone()[0]
    if existing > 0:
        return {"asof": asof, "entries_recorded": 0, "deviations": 0,
                "details": [], "skipped": f"already {existing} entries for {asof}"}

    deviations = check_model_deviation(conn, asof, strategy_id, artifact_root)
    fills = _get_fills(conn, asof, strategy_id)
    proposals = _load_orders_proposal(artifact_root, asof)

    proposal_set = {(p["symbol"], p["side"]) for p in proposals}
    recorded = []

    # Record follows
    for f in fills:
        key = (f["symbol"], f["side"])
        if key in proposal_set:
            jid = record_action(
                conn, asof, strategy_id,
                model_signal={"side": f["side"], "symbol": f["symbol"]},
                actual_action=f,
                override_reason=None,
            )
            recorded.append({"journal_id": jid, "action_type": "model_follow", "symbol": f["symbol"]})

    # Record overrides / unplanned
    for dev in deviations:
        if dev["deviation"] == "unplanned_trade":
            jid = record_action(
                conn, asof, strategy_id,
                model_signal=None,
                actual_action={
                    "symbol": dev["symbol"],
                    "side": dev["actual_side"],
                    "qty": dev["actual_qty"],
                },
                override_reason="unplanned_trade",
            )
            recorded.append({"journal_id": jid, "action_type": "model_override", "symbol": dev["symbol"]})
        elif dev["deviation"] == "model_signal_not_executed":
            jid = record_action(
                conn, asof, strategy_id,
                model_signal={"side": dev["model_side"], "symbol": dev["symbol"], "qty": dev["model_qty"]},
                actual_action={"symbol": dev["symbol"], "side": "NONE", "qty": 0},
                override_reason="signal_not_executed",
            )
            recorded.append({"journal_id": jid, "action_type": "model_override", "symbol": dev["symbol"]})

    return {
        "asof": asof,
        "entries_recorded": len(recorded),
        "deviations": len(deviations),
        "details": recorded,
    }
