"""T0.1 — paper execution approval gate.

Verifies the default two-step workflow: orders get preserved in `proposed`
state, artifacts land in reports/, decision_runs flips to `awaiting_approval`,
and NO fills are inserted until --approve is passed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from trade_schema import connect, ensure_trade_tables


SCRIPT = Path(__file__).resolve().parent.parent / "paper_execute.py"


def _seed_run(db: Path, run_id: str, asof: str) -> None:
    conn = connect(str(db))
    ensure_trade_tables(conn)
    try:
        conn.execute(
            "INSERT INTO decision_runs(run_id, ts, asof, strategy_id, status) VALUES (?,?,?,?,?)",
            (run_id, f"{asof}T16:00:00", asof, "sprint", "proposed"),
        )
        conn.execute(
            """
            INSERT INTO orders(order_id, run_id, asof, symbol, side, qty,
                               order_type, limit_price, tif, reason,
                               expected_fee, expected_slippage, expected_value,
                               status, created_ts, strategy_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"{run_id}__000", run_id, asof, "3041.T", "BUY", 100.0,
                "MKT", None, "DAY", "test",
                0.0, 0.0, 58500.0,
                "proposed", "2026-04-13T16:00:00", "sprint",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _run_paper(db: Path, run_id: str, asof: str, reports_dir: Path, extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(SCRIPT),
        "--db", str(db),
        "--run_id", run_id,
        "--asof", asof,
        "--reports_dir", str(reports_dir),
    ] + extra
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=SCRIPT.parent,
        encoding="utf-8", errors="replace",
    )


def test_default_requires_approval_emits_artifacts_and_pauses(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    run_id = "2026-04-13__testrun01"
    asof = "2026-04-13"
    _seed_run(db, run_id, asof)

    reports = tmp_path / "reports"
    result = _run_paper(db, run_id, asof, reports, extra=[])

    assert result.returncode == 0, result.stderr
    assert "approval required" in result.stdout.lower()

    # per-run artifact is the source of truth; latest_* are convenience pointers
    per_run_path = reports / f"pending_approval_{run_id}.json"
    assert per_run_path.exists(), "per-run-id pending artifact missing"
    pending = json.loads(per_run_path.read_text(encoding="utf-8"))
    assert pending["status"] == "awaiting_approval"
    assert pending["run_id"] == run_id
    assert "order_fingerprint" in pending and len(pending["order_fingerprint"]) == 16
    assert len(pending["orders"]) == 1
    assert pending["orders"][0]["symbol"] == "3041.T"

    # convenience latest pointers also written
    assert (reports / "pending_approval.json").exists()
    assert (reports / "PENDING_APPROVAL.md").exists()

    md = (reports / f"PENDING_APPROVAL_{run_id}.md").read_text(encoding="utf-8")
    assert "3041.T" in md and "approve" in md.lower()
    assert pending["order_fingerprint"] in md

    conn = connect(str(db))
    try:
        (status,) = conn.execute(
            "SELECT status FROM decision_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        assert status == "awaiting_approval"

        order_status = conn.execute(
            "SELECT status FROM orders WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert order_status == "proposed"

        fills = conn.execute(
            "SELECT COUNT(*) FROM fills WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert fills == 0
    finally:
        conn.close()


def test_approve_without_pending_artifact_aborts(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    run_id = "2026-04-13__testrun_noart"
    asof = "2026-04-13"
    _seed_run(db, run_id, asof)
    # Put run in awaiting_approval manually but do NOT create the artifact
    conn = connect(str(db))
    conn.execute("UPDATE decision_runs SET status='awaiting_approval' WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()

    reports = tmp_path / "reports"
    reports.mkdir()
    result = _run_paper(db, run_id, asof, reports, extra=["--approve"])
    assert result.returncode != 0
    assert "no pending_approval artifact" in (result.stdout + result.stderr).lower()


def test_approve_detects_order_drift(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    run_id = "2026-04-13__testrun_drift"
    asof = "2026-04-13"
    _seed_run(db, run_id, asof)
    reports = tmp_path / "reports"

    # Generate pending artifact
    pre = _run_paper(db, run_id, asof, reports, extra=[])
    assert pre.returncode == 0

    # Mutate orders in DB (simulate drift)
    conn = connect(str(db))
    conn.execute(
        "UPDATE orders SET qty=200 WHERE run_id=?", (run_id,),
    )
    conn.commit()
    conn.close()

    result = _run_paper(db, run_id, asof, reports, extra=["--approve"])
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "order-set drift" in combined or "drift" in combined


def test_approve_cas_rejects_second_attempt(tmp_path: Path) -> None:
    """CAS: once a run is not in awaiting_approval, --approve must refuse."""
    db = tmp_path / "m.db"
    run_id = "2026-04-13__testrun_cas"
    asof = "2026-04-13"
    _seed_run(db, run_id, asof)
    reports = tmp_path / "reports"

    # Generate pending artifact (status -> awaiting_approval)
    _run_paper(db, run_id, asof, reports, extra=[])

    # Simulate first-approve consumed the CAS by flipping status already.
    conn = connect(str(db))
    conn.execute(
        "UPDATE decision_runs SET status='approved_in_progress' WHERE run_id=?", (run_id,),
    )
    conn.commit()
    conn.close()

    result = _run_paper(db, run_id, asof, reports, extra=["--approve"])
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "not in awaiting_approval" in combined or "approved_in_progress" in combined


def test_no_require_approval_flag_bypasses_gate(tmp_path: Path) -> None:
    """Legacy path must still be reachable for backfills.

    We cannot drive full simulate_fills here without a seeded daily_prices
    row, so we only assert that the approval gate does NOT short-circuit —
    i.e. the run proceeds far enough to fail on missing market data, not on
    the approval artifact.
    """
    db = tmp_path / "m.db"
    run_id = "2026-04-13__testrun02"
    asof = "2026-04-13"
    _seed_run(db, run_id, asof)

    reports = tmp_path / "reports"
    result = _run_paper(db, run_id, asof, reports, extra=["--no_require_approval"])

    assert not (reports / "pending_approval.json").exists(), \
        "approval artifact should NOT be written when gate is disabled"
    combined = (result.stdout + result.stderr).lower()
    assert "approval required" not in combined
