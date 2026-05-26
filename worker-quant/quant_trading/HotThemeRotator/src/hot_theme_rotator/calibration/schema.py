"""Calibration report schema (§10 gate 5 output)."""
from __future__ import annotations

import math
from dataclasses import dataclass


ALLOWED_CALIBRATION_STATUSES = frozenset(
    {
        "calibrated",
        "insufficient_calibration",
    }
)

ALLOWED_CALIBRATION_SOURCES = frozenset({"opportunity", "attribution"})

# evidence_origin (P10-13, ADR-0006): which sample types backed the calibration
ALLOWED_EVIDENCE_ORIGINS = frozenset({"live", "bootstrap", "mixed"})


class CalibrationReportValidationError(ValueError):
    """Raised when a calibration report would be unsafe to publish."""


@dataclass(frozen=True)
class CalibrationBin:
    """One reliability bin: predicted-probability range + realized rate."""

    lower: float
    upper: float
    sample_count: int
    mean_predicted: float  # nan when sample_count == 0
    mean_actual: float     # nan when sample_count == 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.lower) <= 1.0:
            raise CalibrationReportValidationError(
                f"lower must be in [0, 1]: {self.lower}"
            )
        if not 0.0 <= float(self.upper) <= 1.0:
            raise CalibrationReportValidationError(
                f"upper must be in [0, 1]: {self.upper}"
            )
        if float(self.upper) < float(self.lower):
            raise CalibrationReportValidationError(
                f"upper must be >= lower: {self.lower} -> {self.upper}"
            )
        if int(self.sample_count) < 0:
            raise CalibrationReportValidationError(
                "sample_count must be non-negative"
            )
        if self.sample_count == 0:
            # Empty bin: both means must be nan to be unambiguously "no data".
            if not (math.isnan(float(self.mean_predicted)) and math.isnan(float(self.mean_actual))):
                raise CalibrationReportValidationError(
                    "empty bin must carry nan for mean_predicted and mean_actual"
                )
        else:
            for name, val in (("mean_predicted", self.mean_predicted), ("mean_actual", self.mean_actual)):
                if math.isnan(float(val)) or not 0.0 <= float(val) <= 1.0:
                    raise CalibrationReportValidationError(
                        f"populated bin {name} must be in [0, 1]: {val}"
                    )


@dataclass(frozen=True)
class CalibrationReport:
    """A single calibration evaluation for one (source, horizon) pair.

    ``evidence_origin`` (P10-13, ADR-0006): backward-compatible field defaulting
    to ``"live"`` so existing callers stay valid. ``"bootstrap"`` means the
    sample is entirely historical-backfilled predictions (carries higher
    counterfactual-validity risk). ``"mixed"`` flags hybrid samples for the UI
    to surface a distinct evidence pill.
    """

    source: str
    horizon_days: int
    trade_date_range: tuple[str, str]
    sample_count: int
    status: str
    min_samples_required: int
    brier_score: float | None = None
    log_loss: float | None = None
    bins: tuple[CalibrationBin, ...] = ()
    evidence_origin: str = "live"

    def __post_init__(self) -> None:
        if self.source not in ALLOWED_CALIBRATION_SOURCES:
            raise CalibrationReportValidationError(
                f"source must be one of {sorted(ALLOWED_CALIBRATION_SOURCES)}"
            )
        if self.evidence_origin not in ALLOWED_EVIDENCE_ORIGINS:
            raise CalibrationReportValidationError(
                f"evidence_origin must be one of {sorted(ALLOWED_EVIDENCE_ORIGINS)}"
            )
        if self.status not in ALLOWED_CALIBRATION_STATUSES:
            raise CalibrationReportValidationError(
                f"status must be one of {sorted(ALLOWED_CALIBRATION_STATUSES)}"
            )
        if int(self.horizon_days) <= 0:
            raise CalibrationReportValidationError("horizon_days must be positive")
        if int(self.sample_count) < 0:
            raise CalibrationReportValidationError("sample_count must be non-negative")
        if int(self.min_samples_required) <= 0:
            raise CalibrationReportValidationError("min_samples_required must be positive")
        if self.status == "calibrated":
            if self.sample_count < self.min_samples_required:
                raise CalibrationReportValidationError(
                    f"status='calibrated' requires sample_count ({self.sample_count}) "
                    f">= min_samples_required ({self.min_samples_required})"
                )
            if self.brier_score is None or self.log_loss is None:
                raise CalibrationReportValidationError(
                    "status='calibrated' requires brier_score and log_loss"
                )
        else:
            # insufficient_calibration: NO numeric metrics may be reported,
            # per §9.4 — uncalibrated/insufficient samples cannot be labeled.
            if self.brier_score is not None or self.log_loss is not None:
                raise CalibrationReportValidationError(
                    "status='insufficient_calibration' must not carry brier/log_loss"
                )
            if self.bins:
                raise CalibrationReportValidationError(
                    "status='insufficient_calibration' must not carry bins"
                )
        if not isinstance(self.trade_date_range, tuple) or len(self.trade_date_range) != 2:
            raise CalibrationReportValidationError(
                "trade_date_range must be a (min, max) tuple"
            )
