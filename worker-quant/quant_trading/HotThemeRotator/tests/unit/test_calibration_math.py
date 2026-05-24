"""Tests for calibration math primitives."""
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.calibration.calibrator import (  # noqa: E402
    compute_brier_score,
    compute_calibration_bins,
    compute_log_loss,
    derive_opportunity_ground_truth,
)
from hot_theme_rotator.decision_log.schema import OutcomeRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def test_brier_score_perfect_prediction_is_zero():
    assert compute_brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0


def test_brier_score_constant_half_against_all_ones_is_quarter():
    # all predicted 0.5, all actual 1 → (0.5-1)^2 = 0.25 per row
    assert compute_brier_score([0.5] * 4, [1, 1, 1, 1]) == pytest.approx(0.25)


def test_brier_score_known_mixed_sample():
    # 0.9 vs 1 → 0.01; 0.2 vs 0 → 0.04; 0.7 vs 0 → 0.49; mean = 0.18
    assert compute_brier_score([0.9, 0.2, 0.7], [1, 0, 0]) == pytest.approx(
        (0.01 + 0.04 + 0.49) / 3
    )


def test_brier_score_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_brier_score([0.1, 0.5], [1])


def test_brier_score_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_brier_score([], [])


def test_brier_score_rejects_out_of_range_prob():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        compute_brier_score([1.2], [1])


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


def test_log_loss_perfect_prediction_is_near_zero():
    # 0.999999 close to 1 with actual 1 → log(0.999999) ≈ 0
    ll = compute_log_loss([0.99999, 0.00001], [1, 0])
    assert ll == pytest.approx(-math.log(0.99999), abs=1e-6)


def test_log_loss_clamps_to_avoid_infinite():
    # Predict 0 for actual 1 would be log(0) = -inf without clamp
    ll = compute_log_loss([0.0], [1])
    assert math.isfinite(ll)
    # Clamped to eps=1e-15 → log(1e-15) ≈ -34.5
    assert ll == pytest.approx(-math.log(1e-15), rel=1e-6)


def test_log_loss_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_log_loss([0.5, 0.5], [1])


def test_log_loss_rejects_invalid_eps():
    with pytest.raises(ValueError, match="eps"):
        compute_log_loss([0.5], [1], eps=0.7)


# ---------------------------------------------------------------------------
# Calibration bins
# ---------------------------------------------------------------------------


def test_calibration_bins_default_ten_equal_width():
    bins = compute_calibration_bins([0.05, 0.15, 0.95], [0, 1, 1])
    assert len(bins) == 10
    # first bin (0.0-0.1) gets the 0.05 prediction
    assert bins[0].sample_count == 1
    assert bins[0].mean_predicted == pytest.approx(0.05)
    assert bins[0].mean_actual == pytest.approx(0.0)
    # last bin (0.9-1.0) gets the 0.95 prediction; 1.0 also belongs here
    assert bins[-1].sample_count == 1
    assert bins[-1].mean_predicted == pytest.approx(0.95)


def test_calibration_bins_empty_bins_carry_nan_means():
    bins = compute_calibration_bins([0.05], [1])
    assert bins[5].sample_count == 0
    assert math.isnan(bins[5].mean_predicted)
    assert math.isnan(bins[5].mean_actual)


def test_calibration_bins_last_bin_includes_one_point_oh():
    bins = compute_calibration_bins([1.0], [1])
    assert bins[-1].sample_count == 1
    assert bins[-1].mean_predicted == 1.0


def test_calibration_bins_rejects_non_positive_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        compute_calibration_bins([0.5], [1], n_bins=0)


# ---------------------------------------------------------------------------
# Ground truth derivation
# ---------------------------------------------------------------------------


def _outcome(**overrides):
    base = dict(
        prediction_id="pred-abcd1234ef567890",
        symbol="1306.T",
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        evaluated_as_of="2026-05-28",
        status="complete",
        realized_returns={"1D": 0.012, "3D": -0.005, "5D": 0.034},
        ladder_touches={},
    )
    base.update(overrides)
    return OutcomeRecord.build(**base)


def test_ground_truth_returns_one_for_positive_horizon_return():
    assert derive_opportunity_ground_truth(_outcome(), horizon_key="1D") == 1


def test_ground_truth_returns_zero_for_non_positive_horizon_return():
    assert derive_opportunity_ground_truth(_outcome(), horizon_key="3D") == 0


def test_ground_truth_returns_none_for_incomplete_outcome():
    incomplete = _outcome(
        status="insufficient_data",
        realized_returns={"1D": 0.01},  # missing 3D / 5D
    )
    assert derive_opportunity_ground_truth(incomplete, horizon_key="1D") is None


def test_ground_truth_returns_none_when_horizon_missing():
    only_5d = _outcome(
        status="insufficient_data",
        realized_returns={"5D": 0.04},
    )
    assert derive_opportunity_ground_truth(only_5d, horizon_key="3D") is None
