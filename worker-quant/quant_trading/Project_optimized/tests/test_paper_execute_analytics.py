import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_execute import post_trade_analytics


class TestPostTradeAnalytics(unittest.TestCase):
    def test_post_trade_analytics_filters_by_strategy_and_computes_metrics(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE fills (
                run_id TEXT,
                strategy_id TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                fee REAL,
                price_validated INTEGER,
                quote_close REAL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO fills(run_id, strategy_id, side, qty, price, fee, price_validated, quote_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("r1", "sprint", "BUY", 100, 101.0, 12.0, 1, 100.0),
                ("r1", "sprint", "SELL", 100, 98.0, 8.0, 0, 100.0),
                ("r1", "harvest", "BUY", 100, 150.0, 99.0, 1, 149.0),
            ],
        )

        analytics = post_trade_analytics(conn, "r1", strategy_id="sprint")

        self.assertEqual(analytics["fill_count"], 2)
        self.assertEqual(analytics["strategy_id"], "sprint")
        self.assertAlmostEqual(analytics["total_commission"], 20.0)
        self.assertAlmostEqual(analytics["fill_validation_rate"], 0.5)
        self.assertAlmostEqual(analytics["avg_slippage_bps"], 150.0)


if __name__ == "__main__":
    unittest.main()
