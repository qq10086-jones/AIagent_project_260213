import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.calibration.ladder_feedback import (  # noqa: E402
    LADDER_TIERS,
    LadderFeedbackError,
    build_ladder_feedback_report,
)
from hot_theme_rotator.decision_log.schema import OutcomeRecord, PredictionRecord  # noqa: E402


def _prediction(symbol: str = "8035.T", trade_date: str = "2026-05-20") -> PredictionRecord:
    return PredictionRecord.build(
        symbol=symbol,
        trade_date=trade_date,
        decision_cutoff=f"{trade_date}T09:00:00+09:00",
        input_snapshot_id=f"snap-{symbol}-{trade_date}",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=0.7,
        sell=0.0,
        hold=0.3,
        extra={
            "reference_price": 100.0,
            "ladder": {
                "aggressive_entry": 98.0,
                "balanced_entry": 96.0,
                "conservative_entry": 94.0,
                "stop_price": 90.0,
                "first_exit": 104.0,
                "second_exit": 108.0,
                "stretch_exit": 112.0,
            },
        },
    )


def _touches(*, touched_tiers: set[str] | None = None) -> dict[str, dict[str, object]]:
    touched_tiers = touched_tiers or set()
    return {
        tier: {
            "touched": tier in touched_tiers,
            "touched_at": "2026-05-21" if tier in touched_tiers else None,
        }
        for tier in LADDER_TIERS
    }


def _outcome(
    prediction: PredictionRecord,
    *,
    status: str = "complete",
    touched_tiers: set[str] | None = None,
    ladder_touches: dict[str, dict[str, object]] | None = None,
) -> OutcomeRecord:
    realized_returns = {"1D": 0.01, "3D": 0.03, "5D": 0.04} if status == "complete" else {}
    return OutcomeRecord.build(
        prediction_id=prediction.prediction_id,
        symbol=prediction.symbol,
        trade_date=prediction.trade_date,
        decision_cutoff=prediction.decision_cutoff,
        evaluated_as_of="2026-05-30",
        status=status,
        realized_returns=realized_returns,
        ladder_touches=ladder_touches if ladder_touches is not None else _touches(touched_tiers=touched_tiers),
        failure_reason="" if status == "complete" else "not enough bars",
    )


def test_ladder_feedback_keeps_touch_rate_hidden_below_min_samples():
    prediction = _prediction()
    outcome = _outcome(prediction, touched_tiers={"aggressive_entry", "first_exit"})

    report = build_ladder_feedback_report(
        predictions=[prediction],
        outcomes=[outcome],
        min_samples=2,
    )

    assert report.complete_sample_count == 1
    aggressive = report.tier("aggressive_entry")
    assert aggressive.sample_count == 1
    assert aggressive.touched_count == 1
    assert aggressive.touch_rate is None
    assert aggressive.status == "insufficient_calibration"
    assert report.bullish_calibration.status == "insufficient_calibration"


def test_ladder_feedback_exposes_touch_rate_after_min_samples():
    predictions = [_prediction(symbol=f"80{i}.T") for i in range(3)]
    outcomes = [
        _outcome(predictions[0], touched_tiers={"first_exit"}),
        _outcome(predictions[1], touched_tiers={"first_exit"}),
        _outcome(predictions[2], touched_tiers=set()),
    ]

    report = build_ladder_feedback_report(
        predictions=predictions,
        outcomes=outcomes,
        min_samples=3,
    )

    first_exit = report.tier("first_exit")
    assert first_exit.direction == "above"
    assert first_exit.sample_count == 3
    assert first_exit.touched_count == 2
    assert first_exit.touch_rate == pytest.approx(2 / 3)
    assert first_exit.status == "calibrated"


def test_ladder_feedback_skips_non_complete_outcomes():
    prediction = _prediction()
    incomplete = _outcome(prediction, status="insufficient_data")

    report = build_ladder_feedback_report(
        predictions=[prediction],
        outcomes=[incomplete],
        min_samples=1,
    )

    assert report.complete_sample_count == 0
    assert all(tier.sample_count == 0 for tier in report.tiers)
    assert all(tier.touch_rate is None for tier in report.tiers)


def test_ladder_feedback_fails_closed_when_complete_outcome_missing_tier():
    prediction = _prediction()
    touches = _touches(touched_tiers={"first_exit"})
    touches.pop("stop_price")
    outcome = _outcome(prediction, ladder_touches=touches)

    with pytest.raises(LadderFeedbackError, match="stop_price"):
        build_ladder_feedback_report(
            predictions=[prediction],
            outcomes=[outcome],
            min_samples=1,
        )


def test_ladder_feedback_fails_closed_when_touched_is_not_boolean():
    prediction = _prediction()
    touches = _touches(touched_tiers={"first_exit"})
    touches["first_exit"]["touched"] = "yes"
    outcome = _outcome(prediction, ladder_touches=touches)

    with pytest.raises(LadderFeedbackError, match="touched"):
        build_ladder_feedback_report(
            predictions=[prediction],
            outcomes=[outcome],
            min_samples=1,
        )
