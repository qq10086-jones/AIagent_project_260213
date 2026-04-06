"""Tests for factor tier system (T4) and Ridge CV (T5)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compute_ic import (
    evaluate_tier_transitions,
    get_last_tier_review_date,
    is_tier_review_due,
    load_factor_tiers,
)
from trade_schema import ensure_learning_tables
from model_ridge import PanelRidge


class TestFactorTiers(unittest.TestCase):
    def test_load_factor_tiers_from_config(self):
        tiers = load_factor_tiers(str(ROOT / "config.yaml"))
        self.assertEqual(tiers.get("mom_consist"), "core")
        self.assertEqual(tiers.get("ma_gap"), "core")
        self.assertEqual(tiers.get("sharpe_60"), "core")
        self.assertEqual(tiers.get("high52w"), "candidate")
        self.assertEqual(tiers.get("ret20"), "excluded")
        self.assertEqual(tiers.get("roa_op"), "fundamental_pending")

    def test_candidate_promoted_when_criteria_met(self):
        rules = {"min_observations": 100, "min_t_stat": 1.5, "min_ic_mean": 0.01, "demotion_t_stat": 0.5}
        new_tier, event = evaluate_tier_transitions("high52w", "candidate", t_stat=2.0, ic_mean=0.03, n_obs=120, rules=rules)
        self.assertEqual(new_tier, "core")
        self.assertEqual(event, "TIER_PROMOTED")

    def test_candidate_not_promoted_below_threshold(self):
        rules = {"min_observations": 100, "min_t_stat": 1.5, "min_ic_mean": 0.01, "demotion_t_stat": 0.5}
        new_tier, event = evaluate_tier_transitions("high52w", "candidate", t_stat=1.0, ic_mean=0.03, n_obs=120, rules=rules)
        self.assertEqual(new_tier, "candidate")
        self.assertEqual(event, "")

    def test_core_demoted_when_t_stat_below_demotion(self):
        rules = {"min_observations": 100, "min_t_stat": 1.5, "min_ic_mean": 0.01, "demotion_t_stat": 0.5}
        new_tier, event = evaluate_tier_transitions("ma_gap", "core", t_stat=0.3, ic_mean=0.001, n_obs=150, rules=rules)
        self.assertEqual(new_tier, "candidate")
        self.assertEqual(event, "TIER_DEMOTED")

    def test_core_stays_when_t_stat_healthy(self):
        rules = {"min_observations": 100, "min_t_stat": 1.5, "min_ic_mean": 0.01, "demotion_t_stat": 0.5}
        new_tier, event = evaluate_tier_transitions("ma_gap", "core", t_stat=2.5, ic_mean=0.02, n_obs=200, rules=rules)
        self.assertEqual(new_tier, "core")
        self.assertEqual(event, "")

    def test_excluded_never_transitions(self):
        rules = {"min_observations": 100, "min_t_stat": 1.5, "min_ic_mean": 0.01, "demotion_t_stat": 0.5}
        new_tier, event = evaluate_tier_transitions("ret20", "excluded", t_stat=5.0, ic_mean=0.05, n_obs=500, rules=rules)
        self.assertEqual(new_tier, "excluded")
        self.assertEqual(event, "")

    def test_tier_review_due_without_prior_review(self):
        conn = sqlite3.connect(":memory:")
        ensure_learning_tables(conn)
        due, last_review, days_since = is_tier_review_due(conn, "ma_gap", "2026-04-04", 30)
        self.assertTrue(due)
        self.assertIsNone(last_review)
        self.assertIsNone(days_since)
        conn.close()

    def test_tier_review_skipped_inside_frequency_window(self):
        conn = sqlite3.connect(":memory:")
        ensure_learning_tables(conn)
        conn.execute(
            """
            INSERT INTO learning_audit(asof, event_type, factor_name, old_value, new_value, ic_value, notes)
            VALUES ('2026-03-20', 'TIER_REVIEW', 'ma_gap', NULL, NULL, NULL, '{}')
            """
        )
        conn.commit()
        self.assertEqual(get_last_tier_review_date(conn, "ma_gap"), "2026-03-20")
        due, last_review, days_since = is_tier_review_due(conn, "ma_gap", "2026-04-04", 30)
        self.assertFalse(due)
        self.assertEqual(last_review, "2026-03-20")
        self.assertEqual(days_since, 15)
        conn.close()


class TestRidgeCV(unittest.TestCase):
    def test_fit_with_cv_selects_alpha(self):
        np.random.seed(42)
        n, p = 200, 5
        X = np.random.randn(n, p)
        beta_true = np.array([0.5, -0.3, 0.2, 0.0, 0.1])
        y = X @ beta_true + np.random.randn(n) * 0.5

        model = PanelRidge(alpha=50.0)
        result = model.fit_with_cv(X, y, candidate_alphas=[1, 5, 10, 25, 50, 100])
        self.assertIn("best_alpha", result)
        self.assertIn("cv_results", result)
        self.assertIn("best_ic", result)
        self.assertTrue(np.isfinite(result["best_ic"]))
        # Model should be fitted after CV
        pred = model.predict(X[:5])
        self.assertEqual(len(pred), 5)
        # Alpha should have been updated
        self.assertEqual(model.alpha, result["best_alpha"])

    def test_fit_with_cv_small_sample_fallback(self):
        model = PanelRidge(alpha=50.0)
        X = np.random.randn(3, 2)
        y = np.random.randn(3)
        result = model.fit_with_cv(X, y)
        self.assertEqual(result.get("note"), "too_few_samples")
        # Should still have fitted the model
        self.assertIsNotNone(model.beta_)

    def test_default_fit_unchanged(self):
        np.random.seed(123)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model = PanelRidge(alpha=50.0)
        model.fit(X, y)
        pred1 = model.predict(X[:3])
        # Verify basic fit still works
        self.assertEqual(len(pred1), 3)
        self.assertTrue(np.all(np.isfinite(pred1)))


if __name__ == "__main__":
    unittest.main()
