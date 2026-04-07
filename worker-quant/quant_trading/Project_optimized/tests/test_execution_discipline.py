"""Tests for execution discipline system: action_plan, pending checks, compliance."""
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from trade_schema import connect, ensure_trade_tables, ensure_decision_journal_table
from action_plan_builder import build_action_plan
from check_pending_actions import check_pending_actions
from compliance_tracker import (
    check_model_deviation,
    compute_compliance_score,
    ensure_journal_table,
    record_action,
    record_daily_compliance,
)


@pytest.fixture
def tmp_env(tmp_path):
    """Create temp DB + reports dir + artifacts dir with test data."""
    db_path = str(tmp_path / "test.db")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    artifact_root = tmp_path / "artifacts" / "decision"
    artifact_root.mkdir(parents=True)

    conn = connect(db_path)
    ensure_trade_tables(conn)
    ensure_decision_journal_table(conn)

    # Insert test positions
    conn.execute(
        """INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
           VALUES ('2026-04-07', 'sprint', '4005.T', 100, 528.8, 533.7, 53370, 490)"""
    )
    conn.execute(
        """INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
           VALUES ('2026-04-07', 'sprint', '7267.T', 100, 1259.5, 1252.0, 125200, -750)"""
    )
    conn.commit()

    # Create daily_prices table for price lookups
    conn.execute(
        """CREATE TABLE IF NOT EXISTS daily_prices (
          date TEXT, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
          PRIMARY KEY (date, symbol))"""
    )
    conn.execute("INSERT INTO daily_prices VALUES ('2026-04-07','4005.T',533,541,530,533.7,100000)")
    conn.execute("INSERT INTO daily_prices VALUES ('2026-04-07','7267.T',1260,1267,1246,1252,200000)")
    conn.commit()

    # Write regime_diagnosis.json
    regime = {
        "asof": "2026-04-07",
        "sprint_final_state": "off",
        "fast_ma": 55559.0,
        "slow_ma": 56638.0,
        "enter_line": 56072.0,
    }
    (reports_dir / "regime_diagnosis.json").write_text(json.dumps(regime), encoding="utf-8")

    # Write orders_proposal.csv in artifact
    run_dir = artifact_root / "2026-04-07" / "2026-04-07__test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "orders_proposal.csv").write_text(
        "symbol,side,qty,suggested_type,suggested_limit,est_notional,comment\n"
        "7267.T,SELL,100,MKT,,125200,rebalance\n",
        encoding="utf-8",
    )

    # Write decision_snapshot.json
    snap = {
        "risk_management": {
            "stop_loss_diagnostics": [
                {"symbol": "4005.T", "avg_cost": 528.8, "current_price": 533.7,
                 "atr_pct": 0.04, "triggered": False, "stop_loss_price": 507.6},
                {"symbol": "7267.T", "avg_cost": 1259.5, "current_price": 1252.0,
                 "atr_pct": 0.05, "triggered": False, "stop_loss_price": 1196.5},
            ]
        }
    }
    (run_dir / "decision_snapshot.json").write_text(json.dumps(snap), encoding="utf-8")

    yield {
        "conn": conn,
        "db_path": db_path,
        "reports_dir": reports_dir,
        "artifact_root": artifact_root,
        "asof": "2026-04-07",
    }
    conn.close()


# ── Action Plan Builder ─────────────────────────────────────────────

class TestActionPlanBuilder:
    def test_generates_plan_with_pending_sell(self, tmp_env):
        plan = build_action_plan(
            tmp_env["conn"], tmp_env["asof"], "sprint",
            tmp_env["reports_dir"], tmp_env["artifact_root"],
        )
        assert plan["asof"] == "2026-04-07"
        assert plan["regime"] == "off"
        assert len(plan["pending_sells"]) == 1
        assert plan["pending_sells"][0]["symbol"] == "7267.T"
        assert len(plan["pending_buys"]) == 0
        assert len(plan["held_positions"]) == 2
        assert "SELL" in plan["action_summary"]

    def test_writes_json_file(self, tmp_env):
        build_action_plan(
            tmp_env["conn"], tmp_env["asof"], "sprint",
            tmp_env["reports_dir"], tmp_env["artifact_root"],
        )
        path = tmp_env["reports_dir"] / "action_plan_today.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "action_summary" in data

    def test_no_orders_plan(self, tmp_env):
        # Remove orders_proposal
        for f in tmp_env["artifact_root"].rglob("orders_proposal.csv"):
            f.unlink()
        plan = build_action_plan(
            tmp_env["conn"], tmp_env["asof"], "sprint",
            tmp_env["reports_dir"], tmp_env["artifact_root"],
        )
        assert len(plan["pending_sells"]) == 0
        assert len(plan["pending_buys"]) == 0
        assert "No trades" in plan["action_summary"]


# ── Check Pending Actions ────────────────────────────────────────────

class TestCheckPendingActions:
    def test_detects_unexecuted_sell(self, tmp_env):
        # Write action plan first
        build_action_plan(
            tmp_env["conn"], tmp_env["asof"], "sprint",
            tmp_env["reports_dir"], tmp_env["artifact_root"],
        )
        pending = check_pending_actions(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["reports_dir"],
        )
        assert len(pending) == 1
        assert pending[0]["type"] == "pending_sell"
        assert pending[0]["symbol"] == "7267.T"

    def test_no_alert_when_fill_exists(self, tmp_env):
        # Insert matching fill
        tmp_env["conn"].execute(
            """INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price, fee, tax)
               VALUES ('test001', 'run1', '2026-04-07', 'sprint', '2026-04-07 09:00', '7267.T', 'SELL', 100, 1250, 0, 0)"""
        )
        tmp_env["conn"].commit()
        build_action_plan(
            tmp_env["conn"], tmp_env["asof"], "sprint",
            tmp_env["reports_dir"], tmp_env["artifact_root"],
        )
        pending = check_pending_actions(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["reports_dir"],
        )
        assert len(pending) == 0

    def test_no_alert_when_no_plan(self, tmp_env):
        # Don't build action plan → no file
        pending = check_pending_actions(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["reports_dir"],
        )
        assert len(pending) == 0


# ── Compliance Tracker ───────────────────────────────────────────────

class TestComplianceTracker:
    def test_detect_unplanned_trade(self, tmp_env):
        # Insert a BUY fill that model didn't suggest (model only has SELL 7267.T)
        tmp_env["conn"].execute(
            """INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price, fee, tax)
               VALUES ('test_buy', 'run1', '2026-04-07', 'sprint', '2026-04-07 09:00', '9432.T', 'BUY', 400, 158, 0, 0)"""
        )
        tmp_env["conn"].commit()
        devs = check_model_deviation(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["artifact_root"],
        )
        unplanned = [d for d in devs if d["deviation"] == "unplanned_trade"]
        assert len(unplanned) >= 1
        assert unplanned[0]["symbol"] == "9432.T"

    def test_detect_signal_not_executed(self, tmp_env):
        devs = check_model_deviation(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["artifact_root"],
        )
        not_exec = [d for d in devs if d["deviation"] == "model_signal_not_executed"]
        assert len(not_exec) == 1
        assert not_exec[0]["symbol"] == "7267.T"

    def test_no_deviation_when_follow(self, tmp_env):
        tmp_env["conn"].execute(
            """INSERT INTO fills(fill_id, run_id, asof, strategy_id, ts, symbol, side, qty, price, fee, tax)
               VALUES ('test_sell', 'run1', '2026-04-07', 'sprint', '2026-04-07 09:00', '7267.T', 'SELL', 100, 1250, 0, 0)"""
        )
        tmp_env["conn"].commit()
        devs = check_model_deviation(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["artifact_root"],
        )
        assert len(devs) == 0

    def test_record_action_and_score(self, tmp_env):
        conn = tmp_env["conn"]
        # Record 3 follows and 2 overrides
        for i in range(3):
            record_action(conn, "2026-04-07", "sprint",
                          model_signal={"side": "SELL", "symbol": f"TEST{i}.T"},
                          actual_action={"side": "SELL", "symbol": f"TEST{i}.T", "qty": 100})
        for i in range(2):
            record_action(conn, "2026-04-07", "sprint",
                          model_signal={"side": "BUY", "symbol": f"OVR{i}.T"},
                          actual_action={"side": "SELL", "symbol": f"OVR{i}.T", "qty": 100},
                          override_reason="disagree with model")
        score = compute_compliance_score(conn, "2026-04-07", "2026-04-07")
        assert score["total"] == 5
        assert score["follow"] == 3
        assert score["override"] == 2
        assert score["compliance_rate"] == 0.6

    def test_record_daily_compliance(self, tmp_env):
        # No fills → everything is "not executed"
        result = record_daily_compliance(
            tmp_env["conn"], tmp_env["asof"], "sprint", tmp_env["artifact_root"],
        )
        assert result["deviations"] == 1  # 7267.T SELL not executed
        assert result["entries_recorded"] >= 1


# ── Decision Journal Table ───────────────────────────────────────────

class TestDecisionJournalTable:
    def test_table_created(self, tmp_env):
        rows = tmp_env["conn"].execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='decision_journal'"
        ).fetchall()
        assert len(rows) == 1

    def test_insert_and_query(self, tmp_env):
        conn = tmp_env["conn"]
        record_action(conn, "2026-04-07", "sprint",
                      model_signal=None,
                      actual_action={"side": "BUY", "symbol": "TEST.T", "qty": 100})
        rows = conn.execute("SELECT action_type FROM decision_journal").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "manual_entry"
