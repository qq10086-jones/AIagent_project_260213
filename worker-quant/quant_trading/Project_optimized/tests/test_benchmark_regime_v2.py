import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_regime import benchmark_regime_scale_v2


class TestBenchmarkRegimeV2(unittest.TestCase):
    def test_ma_off_and_high_vix_stays_off(self):
        state, scale, diag = benchmark_regime_scale_v2(
            px_b=90.0,
            fast_ma_b=95.0,
            slow_ma_b=100.0,
            prev_state="on",
            enter_pct=0.01,
            exit_pct=0.01,
            off_scale=0.0,
            caution_scale=0.4,
            use_vix_confirmation=True,
            vix_value=35.0,
            vix_off_threshold=30.0,
        )
        self.assertEqual(state, "off")
        self.assertEqual(scale, 0.0)
        self.assertIn("stay off", diag["diagnosis"])

    def test_ma_off_and_low_vix_downgrades_to_caution(self):
        state, scale, diag = benchmark_regime_scale_v2(
            px_b=90.0,
            fast_ma_b=95.0,
            slow_ma_b=100.0,
            prev_state="on",
            enter_pct=0.01,
            exit_pct=0.01,
            off_scale=0.0,
            caution_scale=0.4,
            use_vix_confirmation=True,
            vix_value=20.0,
            vix_off_threshold=30.0,
        )
        self.assertEqual(state, "caution")
        self.assertEqual(scale, 0.4)
        self.assertIn("downgraded to caution", diag["diagnosis"])

    def test_missing_vix_fails_closed(self):
        state, scale, diag = benchmark_regime_scale_v2(
            px_b=90.0,
            fast_ma_b=95.0,
            slow_ma_b=100.0,
            prev_state="on",
            enter_pct=0.01,
            exit_pct=0.01,
            off_scale=0.0,
            caution_scale=0.4,
            use_vix_confirmation=True,
            vix_value=None,
            vix_off_threshold=30.0,
        )
        self.assertEqual(state, "off")
        self.assertEqual(scale, 0.0)
        self.assertIn("fail closed", diag["diagnosis"])

    def test_missing_vix_can_use_explicit_ma_only_policy(self):
        state, scale, diag = benchmark_regime_scale_v2(
            px_b=90.0,
            fast_ma_b=95.0,
            slow_ma_b=100.0,
            prev_state="on",
            enter_pct=0.01,
            exit_pct=0.01,
            off_scale=0.0,
            caution_scale=0.4,
            use_vix_confirmation=True,
            vix_value=None,
            vix_off_threshold=30.0,
            vix_missing_policy="ma_only",
        )
        self.assertEqual(state, "off")
        self.assertEqual(scale, 0.0)
        self.assertIn("MA-only", diag["diagnosis"])


if __name__ == "__main__":
    unittest.main()
