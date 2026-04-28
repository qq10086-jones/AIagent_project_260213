"""Tests for capital_gate strategy_locks override (Path A 2026-04-28).

Verifies that capital_gate.evaluate honors `strategy_locks` in
capital_gate_config.yaml and forces `recommended_state` regardless of
underlying gate evaluation. This is the load-bearing protection that
prevents sprint from ever re-entering live trading without explicit signoff.
"""
from __future__ import annotations

import sys
import pathlib
import sqlite3
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import capital_gate
from trade_schema import ensure_trade_tables


def _setup_db_with_sprint_evidence(conn: sqlite3.Connection, monthly_excess: float = 0.05):
    """Seed minimal data so capital_gate.evaluate has something to chew on."""
    ensure_trade_tables(conn)
    # Add a few snapshots so retention/kill checks can run
    for i, asof in enumerate(["2026-02-01", "2026-03-01", "2026-04-01", "2026-04-28"]):
        nav = 400_000 * (1 + 0.02 * i)
        conn.execute(
            """INSERT INTO account_snapshots(asof, strategy_id, ts, run_id, cash, positions_value, nav)
               VALUES(?, 'sprint', ?, ?, 200000, 200000, ?)""",
            (asof, asof + " 16:00:00", f"R{i}", nav),
        )
    conn.commit()


class TestStrategyLocks(unittest.TestCase):
    def test_lock_overrides_recommended_state(self):
        """A locked strategy must have recommended_state forced regardless of gate eval."""
        conn = sqlite3.connect(":memory:")
        _setup_db_with_sprint_evidence(conn, monthly_excess=0.05)  # would pass G1 normally
        cfg_with_lock = {
            "version": "test",
            "strategy_locks": {
                "sprint": {
                    "locked_state": "paused_sunk_only",
                    "reason": "test lock",
                    "unlock_requires_signoff": True,
                },
            },
            "g1_entry": {"min_monthly_excess_vs_ew": 0.0},
            "g2_retention": {"trailing_months": 3, "min_trailing_pnl_pct": 0.0},
            "g3_kill": {"max_drawdown_pct": 0.15, "trailing_months": 3,
                        "min_trailing_pnl_pct": -0.05, "early_inception_min_pnl_pct": -0.05},
            "g4_promotion": {"min_monthly_returns": 6, "min_sharpe_ann": 0.5, "min_dsr_score": 0.10},
        }
        decision = capital_gate.evaluate(conn, "sprint", gate_config=cfg_with_lock)
        self.assertEqual(decision.recommended_state, "paused_sunk_only",
            "strategy_locks must force recommended_state to locked value")
        # Reason should mention the lock so humans can debug
        self.assertTrue(
            any("STRATEGY LOCK" in str(r) for r in decision.reasons),
            f"Lock reason must be in decision.reasons; got {decision.reasons}",
        )
        conn.close()

    def test_no_lock_means_normal_evaluation(self):
        """Strategies not in strategy_locks evaluate normally."""
        conn = sqlite3.connect(":memory:")
        _setup_db_with_sprint_evidence(conn)
        cfg_no_lock = {
            "version": "test",
            "strategy_locks": {
                "some_other_strategy": {"locked_state": "retired", "reason": "x"},
            },
            "g1_entry": {"min_monthly_excess_vs_ew": 0.0},
            "g2_retention": {"trailing_months": 3, "min_trailing_pnl_pct": 0.0},
            "g3_kill": {"max_drawdown_pct": 0.15, "trailing_months": 3,
                        "min_trailing_pnl_pct": -0.05, "early_inception_min_pnl_pct": -0.05},
            "g4_promotion": {"min_monthly_returns": 6, "min_sharpe_ann": 0.5, "min_dsr_score": 0.10},
        }
        decision = capital_gate.evaluate(conn, "sprint", gate_config=cfg_no_lock)
        # No lock for sprint → normal flow (whatever it computes is fine, just not forced)
        self.assertNotIn("STRATEGY LOCK ACTIVE: sprint",
            " ".join(str(r) for r in decision.reasons))
        conn.close()

    def test_shipped_capital_gate_config_has_sprint_lock(self):
        """The shipped capital_gate_config.yaml MUST lock sprint per Path A 2026-04-28."""
        cfg = capital_gate.load_gate_config()
        locks = cfg.get("strategy_locks") or {}
        self.assertIn("sprint", locks,
            "Path A: capital_gate_config.yaml must contain strategy_locks.sprint")
        self.assertEqual(locks["sprint"]["locked_state"], "paused_sunk_only",
            "Path A: sprint must be locked to paused_sunk_only")
        self.assertIn("sprint_aggressive", locks,
            "Path A: sprint_aggressive must also be locked")


if __name__ == "__main__":
    unittest.main()
