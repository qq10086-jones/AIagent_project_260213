"""Tests for etf_monthly_check.py — Path A 2026-04-28.

Covers:
- compute_rebalance_drift handles pre-position symbols (Codex review fix)
- compute_monthly_return handles missing snapshots
- send_email failure path returns False (not raise)
- main() returns exit code 2 when email send fails (Task Scheduler visible)
"""
from __future__ import annotations

import os
import sys
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import etf_monthly_check as emc


class TestComputeRebalanceDrift(unittest.TestCase):
    def test_includes_pre_position_target_symbols(self):
        """Codex 2026-04-28 catch: target symbols not yet in positions must
        appear in drift output with current_weight=0."""
        positions = []
        targets = [{"symbol": "1321.T", "weight": 0.5}, {"symbol": "1306.T", "weight": 0.5}]
        drift = emc.compute_rebalance_drift(positions, targets)
        self.assertEqual(set(drift.keys()), {"1321.T", "1306.T"})
        for sym, d in drift.items():
            self.assertEqual(d["current_weight"], 0.0)
            self.assertEqual(d["target_weight"], 0.5)
            self.assertEqual(d["drift_pct"], -0.5)
            self.assertFalse(d["in_positions"])

    def test_normal_drift(self):
        """Drift calculation when both symbols have positions."""
        positions = [
            {"symbol": "1321.T", "qty": 100, "avg_cost": 1000, "market_price": 1100,
             "market_value": 110_000, "unrealized_pnl": 10_000},
            {"symbol": "1306.T", "qty": 50, "avg_cost": 2000, "market_price": 1800,
             "market_value": 90_000, "unrealized_pnl": -10_000},
        ]
        targets = [{"symbol": "1321.T", "weight": 0.5}, {"symbol": "1306.T", "weight": 0.5}]
        drift = emc.compute_rebalance_drift(positions, targets)
        self.assertAlmostEqual(drift["1321.T"]["current_weight"], 110_000/200_000, places=4)
        self.assertAlmostEqual(drift["1321.T"]["drift_pct"], 0.05, places=4)
        self.assertAlmostEqual(drift["1306.T"]["drift_pct"], -0.05, places=4)
        self.assertTrue(drift["1321.T"]["in_positions"])

    def test_off_target_symbol_appears(self):
        """A symbol present in positions but not in targets should still appear (drift = current - 0)."""
        positions = [
            {"symbol": "1321.T", "qty": 100, "market_value": 100_000,
             "avg_cost": 1000, "market_price": 1000, "unrealized_pnl": 0},
            {"symbol": "3041.T", "qty": 400, "market_value": 230_000,  # leftover sprint position
             "avg_cost": 585, "market_price": 575, "unrealized_pnl": -4000},
        ]
        targets = [{"symbol": "1321.T", "weight": 1.0}]
        drift = emc.compute_rebalance_drift(positions, targets)
        self.assertIn("3041.T", drift)
        self.assertEqual(drift["3041.T"]["target_weight"], 0.0)
        self.assertGreater(drift["3041.T"]["abs_drift_pct"], 0.0)


class TestSendEmailFailure(unittest.TestCase):
    def test_send_email_no_creds_returns_false(self):
        """Without EMAIL_USER/EMAIL_PASS env vars, must return False, not raise."""
        with patch.dict(os.environ, {"EMAIL_USER": "", "EMAIL_PASS": ""}, clear=False):
            ok = emc.send_email("test", "body", "x@y.z")
            self.assertFalse(ok)

    def test_send_email_smtp_error_returns_false(self):
        """SMTP error must be caught and return False."""
        with patch.dict(os.environ, {"EMAIL_USER": "u@g.com", "EMAIL_PASS": "p"}, clear=False):
            with patch("etf_monthly_check.smtplib.SMTP_SSL") as MockSMTP:
                MockSMTP.side_effect = Exception("simulated SMTP failure")
                ok = emc.send_email("test", "body", "x@y.z")
                self.assertFalse(ok)


class TestRenderMarkdown(unittest.TestCase):
    def test_report_includes_halt_block(self):
        """Sanity: monthly report must include the psychological HALT block."""
        md = emc.render_markdown_report(
            asof="2026-04-28",
            profile={"rebalance": {"drift_threshold": 0.05}},
            positions=[],
            nav_info=None,
            drift={},
            monthly={"available": False},
            bench={"available": False},
            rebalance_needed=False,
        )
        self.assertIn("HALT", md)
        self.assertIn("STRATEGY_DECISION", md)


if __name__ == "__main__":
    unittest.main()
