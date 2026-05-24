"""Tests for build_calibration_report (§10 gate 5 composition)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.calibration.reporter import (  # noqa: E402
    DEFAULT_MIN_SAMPLES,
    build_calibration_report,
)
from hot_theme_rotator.decision_log.schema import (  # noqa: E402
    OutcomeRecord,
    PredictionRecord,
)


def _prediction(symbol: str, *, buy: float, snapshot_seed: str = "snap") -> PredictionRecord:
    return PredictionRecord.build(
        symbol=symbol,
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        input_snapshot_id=f"{snapshot_seed}-{symbol}",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=buy,
        sell=0.0,
        hold=1.0 - buy,
        extra={"reference_price": 100.0},
    )


def _outcome_for(pred: PredictionRecord, *, returns: dict[str, float],
                 status: str = "complete") -> OutcomeRecord:
    return OutcomeRecord.build(
        prediction_id=pred.prediction_id,
        symbol=pred.symbol,
        trade_date=pred.trade_date,
        decision_cutoff=pred.decision_cutoff,
        evaluated_as_of="2026-05-28",
        status=status,
        realized_returns=returns,
        ladder_touches={},
    )


def _calibrated_horizon_returns(positive: bool) -> dict[str, float]:
    sign = 0.01 if positive else -0.01
    return {"1D": sign, "3D": sign, "5D": sign}


def _build_paired(n: int, *, positive_ratio: float, predicted_prob: float):
    """Construct n prediction/outcome pairs with given win ratio + predicted prob."""
    preds: list[PredictionRecord] = []
    outs: list[OutcomeRecord] = []
    n_positive = int(round(n * positive_ratio))
    for i in range(n):
        pred = _prediction(f"S{i:04d}.T", buy=predicted_prob, snapshot_seed=f"s{i}")
        preds.append(pred)
        outs.append(_outcome_for(pred, returns=_calibrated_horizon_returns(i < n_positive)))
    return preds, outs


def test_below_min_samples_returns_insufficient_calibration():
    preds, outs = _build_paired(n=10, positive_ratio=0.6, predicted_prob=0.7)
    report = build_calibration_report(
        predictions=preds, outcomes=outs, source="opportunity", min_samples=100,
    )
    assert report.status == "insufficient_calibration"
    assert report.sample_count == 10
    assert report.brier_score is None
    assert report.log_loss is None
    assert report.bins == ()


def test_above_min_samples_returns_calibrated_with_metrics():
    preds, outs = _build_paired(n=120, positive_ratio=0.65, predicted_prob=0.7)
    report = build_calibration_report(
        predictions=preds, outcomes=outs, source="opportunity", min_samples=100,
    )
    assert report.status == "calibrated"
    assert report.sample_count == 120
    assert report.brier_score is not None
    assert report.log_loss is not None
    assert len(report.bins) == 10


def test_brier_matches_known_answer_for_uniform_prediction():
    # Predict 0.5 for every record, 65 of 100 win → brier = 0.25 for each
    preds, outs = _build_paired(n=100, positive_ratio=0.65, predicted_prob=0.5)
    report = build_calibration_report(
        predictions=preds, outcomes=outs, source="opportunity", min_samples=100,
    )
    assert report.brier_score == pytest.approx(0.25)


def test_predictions_without_outcomes_are_skipped():
    preds, outs = _build_paired(n=150, positive_ratio=0.5, predicted_prob=0.5)
    # Truncate outcomes: only first 80 have matches
    truncated = outs[:80]
    report = build_calibration_report(
        predictions=preds, outcomes=truncated, source="opportunity", min_samples=100,
    )
    assert report.sample_count == 80
    assert report.status == "insufficient_calibration"


def test_incomplete_outcomes_are_skipped():
    """`status='insufficient_data'` outcomes don't contribute to calibration."""
    preds = [_prediction(f"S{i:04d}.T", buy=0.5, snapshot_seed=f"s{i}") for i in range(120)]
    # First 30 are 'complete', the rest are 'insufficient_data'
    outs = [_outcome_for(p, returns=_calibrated_horizon_returns(i % 2 == 0))
            for i, p in enumerate(preds[:30])]
    outs += [_outcome_for(p, returns={"1D": 0.0}, status="insufficient_data")
             for p in preds[30:]]
    report = build_calibration_report(
        predictions=preds, outcomes=outs, source="opportunity", min_samples=100,
    )
    assert report.sample_count == 30
    assert report.status == "insufficient_calibration"


def test_horizon_parameter_affects_ground_truth():
    """A prediction wins on 1D but loses on 3D; horizon param controls outcome."""
    preds = [_prediction(f"S{i:04d}.T", buy=0.5, snapshot_seed=f"s{i}") for i in range(120)]
    outs = [
        _outcome_for(p, returns={"1D": 0.02, "3D": -0.01, "5D": 0.01})
        for p in preds
    ]
    r_1d = build_calibration_report(
        predictions=preds, outcomes=outs, source="opportunity",
        horizon_days=1, min_samples=100,
    )
    r_3d = build_calibration_report(
        predictions=preds, outcomes=outs, source="opportunity",
        horizon_days=3, min_samples=100,
    )
    # 1D: all win → mean_actual in last bin ~ 1; 3D: all lose → ~ 0
    populated_1d = [b for b in r_1d.bins if b.sample_count > 0]
    populated_3d = [b for b in r_3d.bins if b.sample_count > 0]
    assert all(b.mean_actual == 1.0 for b in populated_1d)
    assert all(b.mean_actual == 0.0 for b in populated_3d)


def test_rejects_invalid_source():
    with pytest.raises(ValueError, match="source"):
        build_calibration_report(
            predictions=[], outcomes=[], source="invented",
        )


def test_default_min_samples_is_documented_constant():
    assert DEFAULT_MIN_SAMPLES == 100
