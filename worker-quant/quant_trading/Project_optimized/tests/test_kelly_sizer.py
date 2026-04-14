import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kelly_sizer import KellyPositionSizer, compute_kelly_params


class TestKellySizer(unittest.TestCase):
    def test_positive_edge_clamped_by_bounds(self):
        sizer = KellyPositionSizer(
            win_rate=0.60,
            avg_win=0.06,
            avg_loss=0.03,
            sample_count=60,
            kelly_fraction=0.5,
            min_position_pct=0.05,
            max_position_pct=0.50,
        )
        self.assertGreater(sizer.edge(), 0.0)
        self.assertGreaterEqual(sizer.suggested_weight(), 0.05)
        self.assertLessEqual(sizer.suggested_weight(), 0.50)

    def test_negative_edge_returns_zero(self):
        sizer = KellyPositionSizer(
            win_rate=0.40,
            avg_win=0.03,
            avg_loss=0.05,
            sample_count=60,
        )
        self.assertEqual(sizer.edge(), 0.0)
        self.assertEqual(sizer.suggested_weight(), 0.0)

    def test_low_sample_uses_fallback(self):
        sizer = KellyPositionSizer(
            win_rate=0.70,
            avg_win=0.08,
            avg_loss=0.03,
            sample_count=10,
            fallback_position_pct=0.10,
            min_samples=30,
        )
        self.assertEqual(sizer.suggested_weight(), 0.10)

    def test_low_sample_defaults_to_zero_no_evidence_no_position(self):
        sizer = KellyPositionSizer(
            win_rate=0.70,
            avg_win=0.08,
            avg_loss=0.03,
            sample_count=10,
            min_samples=30,
        )
        self.assertEqual(sizer.suggested_weight(), 0.0)

    def test_cooldown_forces_zero_weight(self):
        sizer = KellyPositionSizer(
            win_rate=0.70,
            avg_win=0.08,
            avg_loss=0.03,
            sample_count=60,
            cooldown_remaining_days=3,
        )
        self.assertEqual(sizer.suggested_weight(), 0.0)

    def test_compute_kelly_params_detects_loss_streak_cooldown(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE fills (
              fill_id TEXT,
              strategy_id TEXT,
              asof TEXT,
              ts TEXT,
              symbol TEXT,
              side TEXT,
              qty REAL,
              price REAL
            )
            """
        )
        rows = [
            ("1", "sprint", "2026-03-25", "2026-03-25 09:00:00", "AAA.T", "BUY", 100, 100.0),
            ("2", "sprint", "2026-03-26", "2026-03-26 15:00:00", "AAA.T", "SELL", 100, 95.0),
            ("3", "sprint", "2026-03-27", "2026-03-27 09:00:00", "AAA.T", "BUY", 100, 100.0),
            ("4", "sprint", "2026-03-28", "2026-03-28 15:00:00", "AAA.T", "SELL", 100, 94.0),
            ("5", "sprint", "2026-03-29", "2026-03-29 09:00:00", "AAA.T", "BUY", 100, 100.0),
            ("6", "sprint", "2026-03-30", "2026-03-30 15:00:00", "AAA.T", "SELL", 100, 93.0),
        ]
        conn.executemany("INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

        params = compute_kelly_params(conn, strategy_id="sprint", lookback_days=60, asof="2026-04-01")

        self.assertEqual(params["sample_count"], 3)
        self.assertGreater(params["cooldown_remaining_days"], 0)
        self.assertEqual(params["suggested_weight"], 0.0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
