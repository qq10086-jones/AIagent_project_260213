"""Tests for hot_theme_rotator.decision_log.schema."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.decision_log.schema import (  # noqa: E402
    PredictionRecord,
    PredictionRecordValidationError,
    compute_prediction_id,
)


def _build_kwargs(**overrides):
    base = dict(
        symbol="1306.T",
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        input_snapshot_id="pit-1306.T-2026-05-23-abc123",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=0.5,
        sell=0.2,
        hold=0.3,
        extra={"opportunity_score": 64.2},
    )
    base.update(overrides)
    return base


def test_compute_prediction_id_is_deterministic_and_stable():
    pid1 = compute_prediction_id(
        input_snapshot_id="snap-1",
        model_version="v0",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        symbol="1306.T",
    )
    pid2 = compute_prediction_id(
        input_snapshot_id="snap-1",
        model_version="v0",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        symbol="1306.T",
    )
    assert pid1 == pid2
    assert pid1.startswith("pred-")
    assert len(pid1) == len("pred-") + 16


def test_compute_prediction_id_differs_per_symbol_and_model():
    base = dict(
        input_snapshot_id="snap-1",
        model_version="v0",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        symbol="1306.T",
    )
    other_symbol = {**base, "symbol": "7203.T"}
    other_model = {**base, "model_version": "v1"}
    assert compute_prediction_id(**base) != compute_prediction_id(**other_symbol)
    assert compute_prediction_id(**base) != compute_prediction_id(**other_model)


def test_build_produces_valid_record_with_auto_prediction_id():
    record = PredictionRecord.build(**_build_kwargs())
    assert record.prediction_id == compute_prediction_id(
        input_snapshot_id=record.input_snapshot_id,
        model_version=record.model_version,
        decision_cutoff=record.decision_cutoff,
        symbol=record.symbol,
    )
    assert record.extra == {"opportunity_score": 64.2}


def test_record_roundtrips_through_dict():
    record = PredictionRecord.build(**_build_kwargs())
    restored = PredictionRecord.from_dict(record.to_dict())
    assert restored == record


def test_from_dict_rejects_missing_required_field():
    record = PredictionRecord.build(**_build_kwargs())
    payload = record.to_dict()
    del payload["model_version"]
    with pytest.raises(PredictionRecordValidationError, match="model_version"):
        PredictionRecord.from_dict(payload)


def test_from_dict_rejects_tampered_prediction_id():
    record = PredictionRecord.build(**_build_kwargs())
    payload = record.to_dict()
    payload["prediction_id"] = "pred-deadbeefdeadbeef"
    with pytest.raises(PredictionRecordValidationError, match="prediction_id does not match"):
        PredictionRecord.from_dict(payload)


def test_build_rejects_invalid_score_status():
    with pytest.raises(PredictionRecordValidationError, match="score_status"):
        PredictionRecord.build(**_build_kwargs(score_status="made_up_status"))


def test_build_rejects_out_of_range_buy_sell_hold():
    with pytest.raises(PredictionRecordValidationError, match="buy"):
        PredictionRecord.build(**_build_kwargs(buy=1.5))
    with pytest.raises(PredictionRecordValidationError, match="hold"):
        PredictionRecord.build(**_build_kwargs(hold=-0.1))


def test_build_rejects_non_positive_horizon():
    with pytest.raises(PredictionRecordValidationError, match="horizon_days"):
        PredictionRecord.build(**_build_kwargs(horizon_days=0))


def test_build_rejects_invalid_trade_date_and_cutoff():
    with pytest.raises(PredictionRecordValidationError, match="trade_date"):
        PredictionRecord.build(**_build_kwargs(trade_date="not-a-date"))
    with pytest.raises(PredictionRecordValidationError, match="decision_cutoff"):
        PredictionRecord.build(**_build_kwargs(decision_cutoff="not-a-timestamp"))


def test_build_rejects_empty_required_text():
    with pytest.raises(PredictionRecordValidationError, match="symbol"):
        PredictionRecord.build(**_build_kwargs(symbol="  "))
    with pytest.raises(PredictionRecordValidationError, match="input_snapshot_id"):
        PredictionRecord.build(**_build_kwargs(input_snapshot_id=""))
