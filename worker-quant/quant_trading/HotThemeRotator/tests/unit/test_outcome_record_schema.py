"""Tests for OutcomeRecord (§10 gate 4 / consumed by P9-03 calibration)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.decision_log.schema import (  # noqa: E402
    OutcomeRecord,
    OutcomeRecordValidationError,
    compute_outcome_id,
)


def _build_kwargs(**overrides):
    base = dict(
        prediction_id="pred-abcd1234ef567890",
        symbol="1306.T",
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        evaluated_as_of="2026-05-28",
        status="complete",
        realized_returns={"1D": 0.012, "3D": -0.005, "5D": 0.034},
        ladder_touches={
            "balanced_entry": {"touched": True, "touched_at": "2026-05-23T09:30:00+09:00"},
            "first_exit": {"touched": False, "touched_at": None},
            "stop_price": {"touched": False, "touched_at": None},
        },
    )
    base.update(overrides)
    return base


def test_compute_outcome_id_is_deterministic_per_prediction_and_date():
    a = compute_outcome_id(prediction_id="pred-x", evaluated_as_of="2026-05-28")
    b = compute_outcome_id(prediction_id="pred-x", evaluated_as_of="2026-05-28")
    c = compute_outcome_id(prediction_id="pred-x", evaluated_as_of="2026-05-29")
    assert a == b
    assert a != c
    assert a.startswith("out-")
    assert len(a) == len("out-") + 16


def test_build_produces_valid_outcome_with_auto_outcome_id():
    record = OutcomeRecord.build(**_build_kwargs())
    assert record.outcome_id == compute_outcome_id(
        prediction_id=record.prediction_id,
        evaluated_as_of=record.evaluated_as_of,
    )
    assert record.status == "complete"
    assert record.realized_returns["1D"] == 0.012
    assert record.ladder_touches["balanced_entry"]["touched"] is True


def test_outcome_roundtrips_through_dict():
    record = OutcomeRecord.build(**_build_kwargs())
    restored = OutcomeRecord.from_dict(record.to_dict())
    assert restored == record


def test_build_rejects_invalid_status():
    with pytest.raises(OutcomeRecordValidationError, match="status"):
        OutcomeRecord.build(**_build_kwargs(status="made_up_status"))


def test_build_accepts_documented_status_values():
    for status in ("complete", "insufficient_data", "symbol_not_found", "future_cutoff"):
        record = OutcomeRecord.build(**_build_kwargs(status=status))
        assert record.status == status


def test_build_rejects_invalid_dates_and_cutoff():
    with pytest.raises(OutcomeRecordValidationError, match="trade_date"):
        OutcomeRecord.build(**_build_kwargs(trade_date="not-a-date"))
    with pytest.raises(OutcomeRecordValidationError, match="evaluated_as_of"):
        OutcomeRecord.build(**_build_kwargs(evaluated_as_of="not-a-date"))
    with pytest.raises(OutcomeRecordValidationError, match="decision_cutoff"):
        OutcomeRecord.build(**_build_kwargs(decision_cutoff="not-iso"))


def test_build_rejects_empty_prediction_id_or_symbol():
    with pytest.raises(OutcomeRecordValidationError, match="prediction_id"):
        OutcomeRecord.build(**_build_kwargs(prediction_id=""))
    with pytest.raises(OutcomeRecordValidationError, match="symbol"):
        OutcomeRecord.build(**_build_kwargs(symbol=" "))


def test_build_rejects_non_numeric_realized_returns():
    with pytest.raises(OutcomeRecordValidationError, match="realized_returns"):
        OutcomeRecord.build(**_build_kwargs(realized_returns={"1D": "not a number"}))


def test_build_rejects_ladder_touches_missing_touched_key():
    with pytest.raises(OutcomeRecordValidationError, match="touched"):
        OutcomeRecord.build(
            **_build_kwargs(
                ladder_touches={"stop_price": {"touched_at": "2026-05-23T10:00:00+09:00"}}
            )
        )


def test_from_dict_rejects_missing_required_field():
    record = OutcomeRecord.build(**_build_kwargs())
    payload = record.to_dict()
    del payload["evaluated_as_of"]
    with pytest.raises(OutcomeRecordValidationError, match="evaluated_as_of"):
        OutcomeRecord.from_dict(payload)


def test_from_dict_rejects_tampered_outcome_id():
    record = OutcomeRecord.build(**_build_kwargs())
    payload = record.to_dict()
    payload["outcome_id"] = "out-deadbeefdeadbeef"
    with pytest.raises(OutcomeRecordValidationError, match="outcome_id does not match"):
        OutcomeRecord.from_dict(payload)


def test_build_rejects_complete_without_all_standard_horizons():
    """F5 — `status='complete'` must include 1D/3D/5D realized returns."""
    with pytest.raises(OutcomeRecordValidationError, match="missing"):
        OutcomeRecord.build(
            **_build_kwargs(status="complete", realized_returns={"1D": 0.01})
        )
    with pytest.raises(OutcomeRecordValidationError, match="missing"):
        OutcomeRecord.build(
            **_build_kwargs(
                status="complete",
                realized_returns={"1D": 0.01, "3D": 0.02},  # missing 5D
            )
        )


def test_build_accepts_insufficient_with_partial_returns():
    """Partial horizons are allowed for `insufficient_data` status (F5 corollary)."""
    record = OutcomeRecord.build(
        **_build_kwargs(
            status="insufficient_data",
            realized_returns={"1D": 0.01},  # 3D / 5D not yet evaluable
        )
    )
    assert record.status == "insufficient_data"
    assert record.realized_returns == {"1D": 0.01}


def test_build_accepts_malformed_data_status():
    """F2/F3/F4 use the new `malformed_data` status."""
    record = OutcomeRecord.build(
        **_build_kwargs(
            status="malformed_data",
            realized_returns={},
            ladder_touches={},
            failure_reason="reference_price missing",
        )
    )
    assert record.status == "malformed_data"
