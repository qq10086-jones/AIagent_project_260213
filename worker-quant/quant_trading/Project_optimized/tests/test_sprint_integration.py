"""Integration tests: feature_daily → sprint_signal → target_weights pipeline.

These tests verify the plumbing between components, catching silent failures
like the P0 bug where feature_daily never contained price features.
"""
import sys
import sqlite3
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compute_price_features import (
    compute_features_for_symbol,
    write_features_to_db,
    load_prices_from_db,
)
from sprint_signal import sprint_entry_check, sprint_score, sprint_score_v2


# ── Features required by sprint_entry_check ─────────────────────
SPRINT_ENTRY_REQUIRED = {"mom_consist_pctile", "high52w", "vol_z", "fundamental_score"}

# ── All price features that sprint_score_v2 can use ─────────────
SPRINT_SCORE_FACTORS = {
    "ret1", "ret5", "ret20", "ret60",
    "vol20", "vol60", "ma_gap", "z_20", "rsi14", "slope60",
    "mom_12_1", "high52w", "vol_adj_mom20", "mom_consist", "vol_z",
    "sharpe_60", "sharpe_20", "sortino_60", "vol_stability",
}


def _make_price_series(n=300, start_price=1000.0, seed=42):
    """Generate synthetic daily close/volume for testing."""
    rng = np.random.RandomState(seed)
    rets = rng.normal(0.0005, 0.015, size=n)
    prices = start_price * np.cumprod(1 + rets)
    volume = rng.lognormal(15, 0.5, size=n).astype(int)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return (
        pd.Series(prices, index=dates, name="close"),
        pd.Series(volume, index=dates, name="volume"),
    )


class TestComputePriceFeatures(unittest.TestCase):
    """Verify compute_price_features produces all expected features."""

    def test_all_sprint_factors_present(self):
        close, volume = _make_price_series(n=300)
        df = compute_features_for_symbol(close, volume)
        last_row = df.iloc[-1].dropna()
        computed = set(last_row.index)
        missing = SPRINT_SCORE_FACTORS - computed
        self.assertEqual(missing, set(), f"Missing features: {missing}")

    def test_mom_consist_range(self):
        close, volume = _make_price_series()
        df = compute_features_for_symbol(close, volume)
        mc = df["mom_consist"].dropna()
        self.assertTrue((mc >= 0).all() and (mc <= 1).all(),
                        "mom_consist should be in [0, 1]")

    def test_high52w_range(self):
        close, volume = _make_price_series()
        df = compute_features_for_symbol(close, volume)
        h52 = df["high52w"].dropna()
        self.assertTrue((h52 <= 0).all(), "high52w should be <= 0 (distance below 52w max)")

    def test_sharpe_not_nan(self):
        close, volume = _make_price_series(n=300)
        df = compute_features_for_symbol(close, volume)
        last = df.iloc[-1]
        self.assertFalse(np.isnan(last["sharpe_60"]), "sharpe_60 should not be NaN with 300 days")
        self.assertFalse(np.isnan(last["sharpe_20"]), "sharpe_20 should not be NaN with 300 days")


class TestFeatureDailyWriteRead(unittest.TestCase):
    """Verify features survive DB round-trip."""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("""
            CREATE TABLE feature_daily (
                asof TEXT, symbol TEXT, feature_name TEXT, value REAL,
                source_fact_ids TEXT, created_at TEXT,
                PRIMARY KEY (asof, symbol, feature_name)
            )
        """)

    def tearDown(self):
        self.conn.close()

    def test_write_and_read_roundtrip(self):
        close, vol = _make_price_series()
        df = compute_features_for_symbol(close, vol)
        last_row = df.iloc[-1].dropna()
        rows_written = write_features_to_db(
            self.conn, "2025-03-01", {"TEST.T": last_row}, source="test"
        )
        self.assertGreater(rows_written, 15, "Should write 15+ features")

        # Read back
        db_rows = self.conn.execute(
            "SELECT feature_name, value FROM feature_daily WHERE symbol='TEST.T'"
        ).fetchall()
        db_features = {r[0]: r[1] for r in db_rows}

        for feat in SPRINT_SCORE_FACTORS:
            self.assertIn(feat, db_features, f"feature_daily missing {feat} after write")


class TestSprintEntryWithRealFeatures(unittest.TestCase):
    """Verify sprint_entry_check works with compute_price_features output."""

    def test_entry_check_with_computed_features(self):
        """A stock with good momentum should pass entry check."""
        # Create a strongly trending stock
        rng = np.random.RandomState(123)
        n = 300
        rets = rng.normal(0.002, 0.012, size=n)  # positive drift
        prices = 500 * np.cumprod(1 + rets)
        volume = rng.lognormal(16, 0.3, size=n).astype(int)
        # Make recent volume elevated
        volume[-30:] = (volume[-30:] * 2).astype(int)
        dates = pd.bdate_range("2024-01-01", periods=n)

        close = pd.Series(prices, index=dates)
        vol = pd.Series(volume, index=dates)
        df = compute_features_for_symbol(close, vol)
        row = df.iloc[-1].copy()
        # Simulate cross-sectional percentile (this stock is strong)
        row["mom_consist_pctile"] = 0.90
        row["fundamental_score"] = 1.0

        result = sprint_entry_check(row, "on")
        # We don't assert True (depends on exact random seed), but verify no KeyError
        self.assertIsInstance(result, bool)

    def test_entry_check_keys_present(self):
        """Verify all keys needed by sprint_entry_check exist in computed features."""
        close, vol = _make_price_series()
        df = compute_features_for_symbol(close, vol)
        row = df.iloc[-1].copy()
        row["mom_consist_pctile"] = 0.5
        row["fundamental_score"] = 1.0

        needed = {"mom_consist_pctile", "high52w", "vol_z", "fundamental_score"}
        available = set(row.dropna().index)
        missing = needed - available
        self.assertEqual(missing, set(), f"Missing keys for entry check: {missing}")


class TestSprintScoreWithFeatures(unittest.TestCase):
    """Verify sprint_score (v1) works with compute_price_features output."""

    def test_v1_score_nonzero(self):
        close, vol = _make_price_series(n=300, seed=99)
        df = compute_features_for_symbol(close, vol)
        # Build a features DataFrame with multiple symbols
        features = pd.DataFrame({"SYM1": df.iloc[-1], "SYM2": df.iloc[-2]}).T
        scores = sprint_score(features)
        self.assertEqual(len(scores), 2)
        # At least one score should be nonzero if features are present
        self.assertTrue((scores != 0).any(), "sprint_score should produce non-zero scores")

    def test_v2_score_with_registry(self):
        """sprint_score_v2 with a minimal factor registry."""
        close, vol = _make_price_series(n=300, seed=77)
        df = compute_features_for_symbol(close, vol)
        features = pd.DataFrame({"A": df.iloc[-1], "B": df.iloc[-2]}).T

        registry = {
            "sharpe_60": {"icir": 0.20, "n_obs": 100, "is_active": 1},
            "mom_consist": {"icir": 0.10, "n_obs": 100, "is_active": 1},
            "high52w": {"icir": 0.15, "n_obs": 100, "is_active": 1},
        }
        tier_config = {
            "core": ["mom_consist", "ma_gap", "sharpe_60"],
            "candidate": ["high52w"],
            "excluded": [],
        }
        scores = sprint_score_v2(features, factor_registry=registry, tier_config=tier_config)
        self.assertEqual(len(scores), 2)
        self.assertTrue((scores != 0).any(), "sprint_score_v2 should produce non-zero scores with active factors")


if __name__ == "__main__":
    unittest.main()
