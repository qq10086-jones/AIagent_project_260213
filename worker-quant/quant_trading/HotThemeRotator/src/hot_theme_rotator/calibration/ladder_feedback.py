"""Opportunity ladder-tier feedback evidence for P8-05.

This module consumes existing P9-01 PredictionRecord and P9-02 OutcomeRecord
objects. It does not write storage, recompute outcomes, or publish execution
signals. Tier touch rates are level-touch evidence, not win rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hot_theme_rotator.calibration.reporter import build_calibration_report
from hot_theme_rotator.calibration.schema import CalibrationReport
from hot_theme_rotator.decision_log.schema import OutcomeRecord, PredictionRecord


LADDER_TIERS: tuple[str, ...] = (
    "aggressive_entry",
    "balanced_entry",
    "conservative_entry",
    "stop_price",
    "first_exit",
    "second_exit",
    "stretch_exit",
)

BELOW_TIERS = frozenset(
    {
        "aggressive_entry",
        "balanced_entry",
        "conservative_entry",
        "stop_price",
    }
)


class LadderFeedbackError(ValueError):
    """Raised when matched ladder feedback would be unsafe to summarize."""


@dataclass(frozen=True)
class LadderTierFeedback:
    """Feedback evidence for one Rule 9.3 ladder tier."""

    tier: str
    direction: str
    sample_count: int
    touched_count: int
    status: str
    touch_rate: float | None = None

    def __post_init__(self) -> None:
        if self.tier not in LADDER_TIERS:
            raise LadderFeedbackError(f"unknown ladder tier: {self.tier}")
        if self.direction not in {"below", "above"}:
            raise LadderFeedbackError("direction must be 'below' or 'above'")
        if int(self.sample_count) < 0:
            raise LadderFeedbackError("sample_count must be non-negative")
        if int(self.touched_count) < 0:
            raise LadderFeedbackError("touched_count must be non-negative")
        if self.touched_count > self.sample_count:
            raise LadderFeedbackError("touched_count cannot exceed sample_count")
        if self.status not in {"calibrated", "insufficient_calibration"}:
            raise LadderFeedbackError("invalid tier feedback status")
        if self.status == "insufficient_calibration" and self.touch_rate is not None:
            raise LadderFeedbackError("insufficient_calibration must not carry touch_rate")
        if self.touch_rate is not None and not 0.0 <= float(self.touch_rate) <= 1.0:
            raise LadderFeedbackError("touch_rate must be in [0, 1]")


@dataclass(frozen=True)
class LadderFeedbackReport:
    """Opportunity ladder feedback report for one evaluation batch."""

    source: str
    trade_date_range: tuple[str, str]
    complete_sample_count: int
    min_samples_required: int
    tiers: tuple[LadderTierFeedback, ...]
    bullish_calibration: CalibrationReport

    def tier(self, tier_name: str) -> LadderTierFeedback:
        for tier in self.tiers:
            if tier.tier == tier_name:
                return tier
        raise KeyError(tier_name)


def build_ladder_feedback_report(
    *,
    predictions: Sequence[PredictionRecord],
    outcomes: Sequence[OutcomeRecord],
    min_samples: int = 100,
    horizon_days: int = 3,
) -> LadderFeedbackReport:
    """Summarize seven-tier touch evidence from complete matched outcomes."""
    if int(min_samples) <= 0:
        raise ValueError("min_samples must be positive")

    opportunity_predictions = [p for p in predictions if _has_full_ladder(p)]
    prediction_by_id = {p.prediction_id: p for p in opportunity_predictions}
    complete_pairs: list[tuple[PredictionRecord, OutcomeRecord]] = []

    for outcome in outcomes:
        prediction = prediction_by_id.get(outcome.prediction_id)
        if prediction is None or outcome.status != "complete":
            continue
        _validate_complete_ladder_touches(outcome)
        complete_pairs.append((prediction, outcome))

    tiers: list[LadderTierFeedback] = []
    for tier in LADDER_TIERS:
        touched_count = sum(
            1 for _, outcome in complete_pairs if bool(outcome.ladder_touches[tier]["touched"])
        )
        sample_count = len(complete_pairs)
        if sample_count >= int(min_samples):
            status = "calibrated"
            touch_rate = round(touched_count / sample_count, 6) if sample_count else 0.0
        else:
            status = "insufficient_calibration"
            touch_rate = None
        tiers.append(
            LadderTierFeedback(
                tier=tier,
                direction="below" if tier in BELOW_TIERS else "above",
                sample_count=sample_count,
                touched_count=touched_count,
                status=status,
                touch_rate=touch_rate,
            )
        )

    trade_dates = sorted({p.trade_date for p, _ in complete_pairs})
    date_range = (trade_dates[0], trade_dates[-1]) if trade_dates else ("", "")

    bullish_calibration = build_calibration_report(
        predictions=opportunity_predictions,
        outcomes=outcomes,
        source="opportunity",
        horizon_days=int(horizon_days),
        min_samples=int(min_samples),
    )

    return LadderFeedbackReport(
        source="opportunity",
        trade_date_range=date_range,
        complete_sample_count=len(complete_pairs),
        min_samples_required=int(min_samples),
        tiers=tuple(tiers),
        bullish_calibration=bullish_calibration,
    )


def _has_full_ladder(prediction: PredictionRecord) -> bool:
    ladder = prediction.extra.get("ladder")
    if not isinstance(ladder, dict):
        return False
    return all(tier in ladder for tier in LADDER_TIERS)


def _validate_complete_ladder_touches(outcome: OutcomeRecord) -> None:
    for tier in LADDER_TIERS:
        if tier not in outcome.ladder_touches:
            raise LadderFeedbackError(
                f"complete outcome {outcome.outcome_id} missing ladder tier {tier}"
            )
        payload = outcome.ladder_touches[tier]
        if not isinstance(payload, dict) or "touched" not in payload:
            raise LadderFeedbackError(
                f"complete outcome {outcome.outcome_id} tier {tier} missing touched"
            )
        if not isinstance(payload["touched"], bool):
            raise LadderFeedbackError(
                f"complete outcome {outcome.outcome_id} tier {tier} touched must be bool"
            )
