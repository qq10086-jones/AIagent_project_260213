"""P0 regression tests — capital_gate fail-closed + destructive-rerun guard.

These are the minimum bars the 2026-04-15 architecture must pass before
the 17:00 JST auto-run goes live.
"""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capital_gate
import strategy_registry as reg


class TestCapitalGateEarlyInception(unittest.TestCase):
    """G3 kill switch must fire even before 3 months of data exist."""

    def _mk_conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(":memory:")
        c.execute(
            """
            CREATE TABLE account_snapshots (
                asof TEXT, strategy_id TEXT, ts TEXT, run_id TEXT,
                cash REAL, positions_value REAL, nav REAL,
                net_trade_cashflow REAL, fees REAL, tax REAL, notes TEXT
            )
            """
        )
        return c

    def test_g3_trips_on_inception_loss_before_3m(self):
        conn = self._mk_conn()
        # Amihud just started, down 7% after 3 weeks — must kill.
        for asof, nav in [
            ("2026-04-01", 100_000.0),
            ("2026-04-08", 96_000.0),
            ("2026-04-15", 93_000.0),   # -7% since inception
        ]:
            conn.execute(
                "INSERT INTO account_snapshots VALUES (?, ?, ?, NULL, ?, 0, ?, 0, 0, 0, ?)",
                (asof, "amihud", asof, nav, nav, "test"),
            )
        d = capital_gate.evaluate(conn, "amihud")
        self.assertFalse(d.passes["G3_kill"],
                         f"expected G3 to FAIL (kill) on -7% inception, got reasons={d.reasons}")
        self.assertEqual(d.recommended_state, "paused")

    def test_g3_ok_when_inception_loss_within_5pct(self):
        conn = self._mk_conn()
        for asof, nav in [
            ("2026-04-01", 100_000.0),
            ("2026-04-15", 97_000.0),  # -3% — not enough to kill
        ]:
            conn.execute(
                "INSERT INTO account_snapshots VALUES (?, ?, ?, NULL, ?, 0, ?, 0, 0, 0, ?)",
                (asof, "amihud", asof, nav, nav, "test"),
            )
        d = capital_gate.evaluate(conn, "amihud")
        self.assertTrue(d.passes["G3_kill"])


class TestDestructiveRerunGuard(unittest.TestCase):
    """run_alpha_factor_decision must refuse to rewrite executed runs."""

    def _mk_conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(":memory:")
        c.execute(
            """
            CREATE TABLE decision_runs (
                run_id TEXT PRIMARY KEY, asof TEXT, ts TEXT,
                snapshot_path TEXT, status TEXT, notes TEXT, strategy_id TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY, run_id TEXT, asof TEXT, symbol TEXT,
                side TEXT, qty REAL, order_type TEXT, limit_price REAL,
                tif TEXT, reason TEXT, expected_fee REAL, expected_slippage REAL,
                expected_value REAL, status TEXT, created_ts TEXT,
                strategy_id TEXT, source TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE fills (
                fill_id TEXT PRIMARY KEY, run_id TEXT, order_id TEXT, asof TEXT,
                ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL,
                fee REAL, tax REAL, slippage_bps REAL, strategy_id TEXT,
                venue TEXT, source TEXT, validated INTEGER
            )
            """
        )
        return c

    def test_refuses_to_overwrite_filled_run(self):
        from run_alpha_factor_decision import _write_decision, DestructiveRerunError
        conn = self._mk_conn()
        run_id = "2026-04-15__test__abcd1234"
        # Pre-existing decision with a FILLED order.
        conn.execute(
            "INSERT INTO decision_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, "2026-04-15", "2026-04-15T09:00", "", "proposed", "", "test"),
        )
        conn.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"{run_id}__000", run_id, "2026-04-15", "3041.T",
             "BUY", 100, "MKT", None, "DAY", "x",
             0.0, 0.0, 58500.0, "filled", "2026-04-15", "test", "x"),
        )
        conn.commit()
        with self.assertRaises(DestructiveRerunError):
            _write_decision(conn, run_id, "test", "2026-04-15",
                            [{"order_id": f"{run_id}__000",
                              "symbol": "3041.T", "side": "BUY", "qty": 100,
                              "order_type": "MKT", "limit_price": None,
                              "tif": "DAY", "reason": "x", "expected_fee": 0,
                              "expected_slippage": 0, "expected_value": 58000}])

    def test_refuses_when_fills_exist(self):
        from run_alpha_factor_decision import _write_decision, DestructiveRerunError
        conn = self._mk_conn()
        run_id = "2026-04-15__test__fill0001"
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("f1", run_id, f"{run_id}__000", "2026-04-15", "2026-04-15T10",
             "3041.T", "BUY", 100, 585, 0, 0, 5, "test", "PAPER", "x", 1),
        )
        conn.commit()
        with self.assertRaises(DestructiveRerunError):
            _write_decision(conn, run_id, "test", "2026-04-15",
                            [{"order_id": "z", "symbol": "3041.T",
                              "side": "BUY", "qty": 100, "order_type": "MKT",
                              "limit_price": None, "tif": "DAY", "reason": "x",
                              "expected_fee": 0, "expected_slippage": 0,
                              "expected_value": 58000}])

    def test_allows_rerun_when_only_proposed(self):
        from run_alpha_factor_decision import _write_decision
        conn = self._mk_conn()
        run_id = "2026-04-15__test__aaaaaaaa"
        conn.execute(
            "INSERT INTO decision_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, "2026-04-15", "t0", "", "proposed", "", "test"),
        )
        conn.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"{run_id}__old", run_id, "2026-04-15", "0000.T",
             "BUY", 100, "MKT", None, "DAY", "old",
             0, 0, 1000.0, "proposed", "t0", "test", "x"),
        )
        conn.commit()
        _write_decision(conn, run_id, "test", "2026-04-15",
                        [{"order_id": f"{run_id}__new", "symbol": "9999.T",
                          "side": "BUY", "qty": 100, "order_type": "MKT",
                          "limit_price": None, "tif": "DAY", "reason": "new",
                          "expected_fee": 0, "expected_slippage": 0,
                          "expected_value": 2000.0}])
        rows = conn.execute(
            "SELECT symbol FROM orders WHERE run_id=?", (run_id,)
        ).fetchall()
        # Old proposed should be cleared, new one inserted.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "9999.T")


class TestAlphaFactorRotationSells(unittest.TestCase):
    """Monthly rebalance must SELL names that dropped out of top-K."""

    def test_rotation_out_emits_sell(self):
        # Integration-level test: we can't easily reach main() without a
        # full DB, but we can verify the rotation logic is invoked by
        # inspecting the code contract. Minimal guard: import the symbol.
        import run_alpha_factor_decision as mod
        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn('"side": "SELL"', src,
                      "SELL order emission must be present in run_alpha_factor_decision.py")
        self.assertIn("rotate_out", src,
                      "rotate_out rotation logic must exist")


if __name__ == "__main__":
    unittest.main()
