"""Tests for CalibrationReport + CalibrationBin (§10 gate 5)."""
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.calibration.schema import (  # noqa: E402
    CalibrationBin,
    CalibrationReport,
    CalibrationReportValidationError,
)


def _bin(**overrides):
    base = dict(
        lower=0.4,
        upper=0.5,
        sample_count=20,
        mean_predicted=0.45,
        mean_actual=0.50,
    )
    base.update(overrides)
    return CalibrationBin(**base)


def _empty_bin(lower=0.0, upper=0.1):
    return CalibrationBin(
        lower=lower,
        upper=upper,
        sample_count=0,
        mean_predicted=float("nan"),
        mean_actual=float("nan"),
    )


def test_calibration_bin_rejects_lower_above_one():
    with pytest.raises(CalibrationReportValidationError, match="lower"):
        _bin(lower=1.5)


def test_calibration_bin_rejects_upper_below_lower():
    with pytest.raises(CalibrationReportValidationError, match="upper must be >= lower"):
        _bin(lower=0.6, upper=0.4)


def test_calibration_bin_rejects_negative_sample_count():
    with pytest.raises(CalibrationReportValidationError, match="sample_count"):
        _bin(sample_count=-1)


def test_empty_bin_requires_nan_means():
    """A 0-sample bin with a real mean would silently fake calibration."""
    with pytest.raises(CalibrationReportValidationError, match="nan"):
        CalibrationBin(
            lower=0.0, upper=0.1, sample_count=0,
            mean_predicted=0.5, mean_actual=0.5,
        )


def test_populated_bin_rejects_nan_means():
    with pytest.raises(CalibrationReportValidationError, match="must be in"):
        CalibrationBin(
            lower=0.4, upper=0.5, sample_count=10,
            mean_predicted=float("nan"), mean_actual=0.5,
        )


def _calibrated_kwargs(**overrides):
    base = dict(
        source="opportunity",
        horizon_days=3,
        trade_date_range=("2026-04-01", "2026-05-23"),
        sample_count=150,
        status="calibrated",
        min_samples_required=100,
        brier_score=0.21,
        log_loss=0.59,
        bins=(_bin(),),
    )
    base.update(overrides)
    return base


def _insufficient_kwargs(**overrides):
    base = dict(
        source="opportunity",
        horizon_days=3,
        trade_date_range=("2026-05-20", "2026-05-23"),
        sample_count=12,
        status="insufficient_calibration",
        min_samples_required=100,
        brier_score=None,
        log_loss=None,
        bins=(),
    )
    base.update(overrides)
    return base


def test_calibration_report_accepts_valid_calibrated_payload():
    report = CalibrationReport(**_calibrated_kwargs())
    assert report.status == "calibrated"
    assert report.brier_score == 0.21


def test_calibration_report_accepts_valid_insufficient_payload():
    report = CalibrationReport(**_insufficient_kwargs())
    assert report.status == "insufficient_calibration"
    assert report.brier_score is None
    assert report.bins == ()


def test_calibration_report_rejects_invalid_source():
    with pytest.raises(CalibrationReportValidationError, match="source"):
        CalibrationReport(**_calibrated_kwargs(source="made_up"))


def test_calibration_report_rejects_invalid_status():
    with pytest.raises(CalibrationReportValidationError, match="status"):
        CalibrationReport(**_calibrated_kwargs(status="wishful_thinking"))


def test_calibration_report_rejects_calibrated_below_min_samples():
    """§9.4 — sample below threshold cannot wear the calibrated label."""
    with pytest.raises(CalibrationReportValidationError, match="sample_count"):
        CalibrationReport(**_calibrated_kwargs(sample_count=50))


def test_calibration_report_rejects_calibrated_without_metrics():
    with pytest.raises(CalibrationReportValidationError, match="brier_score"):
        CalibrationReport(**_calibrated_kwargs(brier_score=None))


def test_calibration_report_rejects_insufficient_with_smuggled_metrics():
    """§9.4 — insufficient samples must NOT publish brier/log_loss/bins."""
    with pytest.raises(CalibrationReportValidationError, match="brier"):
        CalibrationReport(**_insufficient_kwargs(brier_score=0.25))
    with pytest.raises(CalibrationReportValidationError, match="bins"):
        CalibrationReport(**_insufficient_kwargs(bins=(_empty_bin(),)))


def test_calibration_report_rejects_non_positive_horizon():
    with pytest.raises(CalibrationReportValidationError, match="horizon_days"):
        CalibrationReport(**_calibrated_kwargs(horizon_days=0))
