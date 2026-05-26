"""Build a `CalibrationReport` from matched predictions + outcomes."""
from __future__ import annotations

from typing import Sequence

from hot_theme_rotator.calibration.calibrator import (
    compute_brier_score,
    compute_calibration_bins,
    compute_log_loss,
    derive_opportunity_ground_truth,
)
from hot_theme_rotator.calibration.schema import (
    ALLOWED_CALIBRATION_SOURCES,
    CalibrationReport,
)
from hot_theme_rotator.decision_log.schema import OutcomeRecord, PredictionRecord


DEFAULT_MIN_SAMPLES = 100
DEFAULT_N_BINS = 10


def derive_evidence_origin(predictions: Sequence[PredictionRecord]) -> str:
    """Return ``live`` / ``bootstrap`` / ``mixed`` based on predictions' extra.backdated.

    A prediction is considered backdated when ``extra.backdated=True`` OR its
    ``model_version`` ends with ``"-backdated"`` (defensive — both flags must
    be set by the bootstrap tool but we accept either at read time).
    """
    if not predictions:
        return "live"
    backdated_count = 0
    live_count = 0
    for pred in predictions:
        is_bd = bool(pred.extra.get("backdated", False)) or pred.model_version.endswith("-backdated")
        if is_bd:
            backdated_count += 1
        else:
            live_count += 1
    if backdated_count == 0:
        return "live"
    if live_count == 0:
        return "bootstrap"
    return "mixed"


def build_calibration_report(
    *,
    predictions: Sequence[PredictionRecord],
    outcomes: Sequence[OutcomeRecord],
    source: str,
    horizon_days: int = 3,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_bins: int = DEFAULT_N_BINS,
) -> CalibrationReport:
    """Pair predictions with outcomes; compute calibration if sample is large enough.

    Pairing key: `prediction_id`. Predictions without a matching outcome, and
    outcomes not in `status='complete'`, are skipped. Ground truth derives
    bullish-only: `realized_returns[f"{horizon_days}D"] > 0 → 1`, else `0`.

    Below `min_samples`, returns `status='insufficient_calibration'` with no
    metrics — per §9.4, uncalibrated samples must not be labeled as win rates.
    """
    if source not in ALLOWED_CALIBRATION_SOURCES:
        raise ValueError(
            f"source must be one of {sorted(ALLOWED_CALIBRATION_SOURCES)}"
        )
    if int(horizon_days) <= 0:
        raise ValueError("horizon_days must be positive")
    if int(min_samples) <= 0:
        raise ValueError("min_samples must be positive")

    horizon_key = f"{int(horizon_days)}D"
    outcome_by_pid = {o.prediction_id: o for o in outcomes}

    paired: list[tuple[float, int]] = []
    for pred in predictions:
        outcome = outcome_by_pid.get(pred.prediction_id)
        if outcome is None:
            continue
        gt = derive_opportunity_ground_truth(outcome, horizon_key=horizon_key)
        if gt is None:
            continue
        # Opportunity scanner is bullish-only: `pred.buy` is the long-side prob.
        # Attribution path uses the same field for buy-leg calibration; sell-leg
        # calibration is left to a future report (different source / horizon).
        paired.append((float(pred.buy), int(gt)))

    if predictions:
        trade_dates = sorted({p.trade_date for p in predictions})
        date_range = (trade_dates[0], trade_dates[-1])
    else:
        date_range = ("", "")

    sample_count = len(paired)
    evidence_origin = derive_evidence_origin(predictions)
    if sample_count < int(min_samples):
        return CalibrationReport(
            source=source,
            horizon_days=int(horizon_days),
            trade_date_range=date_range,
            sample_count=sample_count,
            status="insufficient_calibration",
            min_samples_required=int(min_samples),
            brier_score=None,
            log_loss=None,
            bins=(),
            evidence_origin=evidence_origin,
        )

    predicted = [p[0] for p in paired]
    actual = [p[1] for p in paired]
    return CalibrationReport(
        source=source,
        horizon_days=int(horizon_days),
        trade_date_range=date_range,
        sample_count=sample_count,
        status="calibrated",
        min_samples_required=int(min_samples),
        brier_score=compute_brier_score(predicted, actual),
        log_loss=compute_log_loss(predicted, actual),
        bins=compute_calibration_bins(predicted, actual, n_bins=int(n_bins)),
        evidence_origin=evidence_origin,
    )
