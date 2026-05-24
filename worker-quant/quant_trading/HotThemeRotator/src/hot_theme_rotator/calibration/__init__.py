"""Calibration engine — §10 gate 5.

Consumes `PredictionRecord` + `OutcomeRecord` from `decision_log/` and produces
`CalibrationReport` with Brier score, log loss, and reliability bins. Until the
report's sample count meets the configured `min_samples` threshold, output
remains `insufficient_calibration` and downstream UI must NOT label the
underlying scores as win rates (§9.4).

Calibration is the only place in the system where a score may legitimately be
called `calibrated_probability` (per §8.7 / §8.8). Even then, advice-only stays
in force and a human approval gate is required before any execution path opens.
"""
from hot_theme_rotator.calibration.calibrator import (
    compute_brier_score,
    compute_calibration_bins,
    compute_log_loss,
    derive_opportunity_ground_truth,
)
from hot_theme_rotator.calibration.reporter import (
    DEFAULT_MIN_SAMPLES,
    build_calibration_report,
)
from hot_theme_rotator.calibration.ladder_feedback import (
    LADDER_TIERS,
    LadderFeedbackError,
    LadderFeedbackReport,
    LadderTierFeedback,
    build_ladder_feedback_report,
)
from hot_theme_rotator.calibration.schema import (
    ALLOWED_CALIBRATION_SOURCES,
    ALLOWED_CALIBRATION_STATUSES,
    CalibrationBin,
    CalibrationReport,
    CalibrationReportValidationError,
)

__all__ = [
    "ALLOWED_CALIBRATION_SOURCES",
    "ALLOWED_CALIBRATION_STATUSES",
    "CalibrationBin",
    "CalibrationReport",
    "CalibrationReportValidationError",
    "DEFAULT_MIN_SAMPLES",
    "LADDER_TIERS",
    "LadderFeedbackError",
    "LadderFeedbackReport",
    "LadderTierFeedback",
    "build_calibration_report",
    "build_ladder_feedback_report",
    "compute_brier_score",
    "compute_calibration_bins",
    "compute_log_loss",
    "derive_opportunity_ground_truth",
]
