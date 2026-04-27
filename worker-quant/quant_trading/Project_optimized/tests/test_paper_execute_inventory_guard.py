"""Tests for paper_execute SELL inventory guard.

Background (2026-04-28 audit): paper simulator phantom NAV bug — SELL fills
were created without verifying inventory, inflating cash by ¥1M+ when the
strategy emitted repeated SELLs after position was already 0.

Codex review (2026-04-28) caught a follow-up bug: the guard was reading
the SOURCE strategy's positions (e.g. `sprint` real lane) when validating
the PAPER lane (e.g. `sprint_paper`) inventory. This test verifies the
guard uses the correct lane via `inventory_strategy_id`.
"""
from __future__ import annotations

import os
import sys
import pathlib
import sqlite3
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from paper_execute import simulate_fills, _current_paper_position
from trade_schema import ensure_trade_tables


def _setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_trade_tables(conn)
    # Daily prices for quote
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            PRIMARY KEY(symbol, date)
        )
    """)
    conn.execute("""
        INSERT INTO daily_prices VALUES('3041.T','2026-04-28',570.0,580.0,565.0,575.0,1000)
    """)
    return conn


def _add_order(conn, run_id, strategy_id, symbol, side, qty):
    conn.execute("""
        INSERT INTO orders(order_id, run_id, asof, symbol, side, qty, order_type, status, strategy_id)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (f"{run_id}_{symbol}_{side}", run_id, "2026-04-28", symbol, side, qty, "MKT", "proposed", strategy_id))
    conn.commit()


def _add_position(conn, asof, strategy_id, symbol, qty):
    conn.execute("""
        INSERT INTO positions(asof, strategy_id, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
    """, (asof, strategy_id, symbol, qty, 580.0, 575.0, qty*575.0, qty*(575.0-580.0)))
    conn.commit()


class TestPaperInventoryGuard(unittest.TestCase):
    def test_sell_within_inventory_passes(self):
        conn = _setup_db()
        _add_order(conn, "RUN1", "sprint", "3041.T", "SELL", 200)
        _add_position(conn, "2026-04-28", "sprint_paper", "3041.T", 200)
        df, _ = simulate_fills(
            conn, run_id="RUN1", asof="2026-04-28", price_mode="latest",
            slippage_bps=5.0, fee_bps=0.0, fill_ratio=1.0,
            strategy_id="sprint", inventory_strategy_id="sprint_paper",
        )
        self.assertEqual(len(df), 1, "SELL within inventory should produce one fill")
        self.assertEqual(int(df.iloc[0]["qty"]), 200)

    def test_sell_exceeding_inventory_rejected(self):
        conn = _setup_db()
        _add_order(conn, "RUN2", "sprint", "3041.T", "SELL", 400)
        _add_position(conn, "2026-04-28", "sprint_paper", "3041.T", 100)  # only 100 in paper
        df, _ = simulate_fills(
            conn, run_id="RUN2", asof="2026-04-28", price_mode="latest",
            slippage_bps=5.0, fee_bps=0.0, fill_ratio=1.0,
            strategy_id="sprint", inventory_strategy_id="sprint_paper",
        )
        self.assertEqual(len(df), 0, "SELL exceeding paper inventory must be rejected")

    def test_guard_reads_paper_lane_not_source_lane(self):
        """Codex 2026-04-28 review: guard must validate against paper lane,
        not the source (real) lane. If real has 400 but paper has 0, SELL must reject."""
        conn = _setup_db()
        _add_order(conn, "RUN3", "sprint", "3041.T", "SELL", 400)
        _add_position(conn, "2026-04-28", "sprint", "3041.T", 400)        # real has 400
        # paper lane has NO position
        df, _ = simulate_fills(
            conn, run_id="RUN3", asof="2026-04-28", price_mode="latest",
            slippage_bps=5.0, fee_bps=0.0, fill_ratio=1.0,
            strategy_id="sprint", inventory_strategy_id="sprint_paper",
        )
        self.assertEqual(len(df), 0,
            "Guard MUST check sprint_paper lane (empty), not sprint (400 real). "
            "If this passes >0 fills, the lane bug from 2026-04-27 is back.")

    def test_intra_run_position_delta(self):
        """Multiple SELLs in same run should accumulate against same inventory."""
        conn = _setup_db()
        _add_order(conn, "RUN4", "sprint", "3041.T", "SELL", 100)
        # Add second SELL via direct insert to control order_id ordering
        conn.execute("""
            INSERT INTO orders(order_id, run_id, asof, symbol, side, qty, order_type, status, strategy_id)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("RUN4_3041.T_SELL_2", "RUN4", "2026-04-28", "3041.T", "SELL", 100, "MKT", "proposed", "sprint"))
        conn.commit()
        _add_position(conn, "2026-04-28", "sprint_paper", "3041.T", 150)  # only 150 paper
        df, _ = simulate_fills(
            conn, run_id="RUN4", asof="2026-04-28", price_mode="latest",
            slippage_bps=5.0, fee_bps=0.0, fill_ratio=1.0,
            strategy_id="sprint", inventory_strategy_id="sprint_paper",
        )
        # First SELL 100 OK, second SELL 100 should reject (50 left)
        self.assertEqual(len(df), 1, "Second SELL must reject when intra-run delta exhausts inventory")

    def test_default_inventory_strategy_falls_back_to_source(self):
        """Backward-compat: when inventory_strategy_id not provided, falls back to strategy_id."""
        conn = _setup_db()
        _add_order(conn, "RUN5", "sprint_paper", "3041.T", "SELL", 200)
        _add_position(conn, "2026-04-28", "sprint_paper", "3041.T", 200)
        df, _ = simulate_fills(
            conn, run_id="RUN5", asof="2026-04-28", price_mode="latest",
            slippage_bps=5.0, fee_bps=0.0, fill_ratio=1.0,
            strategy_id="sprint_paper",  # source == paper, no inventory_strategy_id
        )
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
