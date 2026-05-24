"""Tests for outcome JSONL writer (mirror of prediction writer for §10 gate 4)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    DecisionLogStorageError,
    append_outcome,
    append_outcomes,
    outcomes_path,
    read_outcomes,
)
from hot_theme_rotator.decision_log.schema import OutcomeRecord  # noqa: E402


def _outcome(**overrides):
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
        },
    )
    base.update(overrides)
    return OutcomeRecord.build(**base)


def test_outcomes_path_resolves_under_reports_outcomes(tmp_path):
    path = outcomes_path(trade_date="2026-05-23", base_dir=tmp_path)
    assert path == tmp_path / "reports" / "outcomes" / "2026-05-23.jsonl"


def test_predictions_and_outcomes_paths_are_separate(tmp_path):
    from hot_theme_rotator.decision_log.jsonl_writer import predictions_path

    pred_p = predictions_path(trade_date="2026-05-23", base_dir=tmp_path)
    out_p = outcomes_path(trade_date="2026-05-23", base_dir=tmp_path)
    assert pred_p != out_p
    assert pred_p.parent.name == "predictions"
    assert out_p.parent.name == "outcomes"


def test_append_then_read_roundtrip_single_outcome(tmp_path):
    record = _outcome()
    target = append_outcome(record, base_dir=tmp_path)
    assert target.exists()
    restored = read_outcomes(trade_date=record.trade_date, base_dir=tmp_path)
    assert restored == (record,)


def test_append_multiple_outcomes_preserves_order(tmp_path):
    a = _outcome(prediction_id="pred-aaaaaaaaaaaaaaaa")
    b = _outcome(prediction_id="pred-bbbbbbbbbbbbbbbb")
    c = _outcome(prediction_id="pred-cccccccccccccccc")
    append_outcomes([a, b, c], base_dir=tmp_path)
    restored = read_outcomes(trade_date="2026-05-23", base_dir=tmp_path)
    assert tuple(r.prediction_id for r in restored) == (
        "pred-aaaaaaaaaaaaaaaa",
        "pred-bbbbbbbbbbbbbbbb",
        "pred-cccccccccccccccc",
    )


def test_read_missing_file_returns_empty_tuple(tmp_path):
    assert read_outcomes(trade_date="2026-05-23", base_dir=tmp_path) == ()


def test_append_rejects_duplicate_outcome_id(tmp_path):
    record = _outcome()
    append_outcome(record, base_dir=tmp_path)
    with pytest.raises(DecisionLogStorageError, match="already present"):
        append_outcome(record, base_dir=tmp_path)


def test_append_allows_reevaluation_with_later_evaluated_as_of(tmp_path):
    """Same prediction re-evaluated on a later date produces a NEW outcome_id."""
    early = _outcome(evaluated_as_of="2026-05-24", status="insufficient_data")
    late = _outcome(evaluated_as_of="2026-05-28", status="complete")
    append_outcome(early, base_dir=tmp_path)
    append_outcome(late, base_dir=tmp_path)
    restored = read_outcomes(trade_date="2026-05-23", base_dir=tmp_path)
    assert len(restored) == 2
    assert {r.status for r in restored} == {"insufficient_data", "complete"}


def test_append_requires_outcome_record_instance(tmp_path):
    with pytest.raises(DecisionLogStorageError, match="OutcomeRecord"):
        append_outcome({"not": "an outcome"}, base_dir=tmp_path)  # type: ignore[arg-type]


def test_read_fails_closed_on_corrupted_jsonl(tmp_path):
    record = _outcome()
    target = append_outcome(record, base_dir=tmp_path)
    with target.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
    with pytest.raises(DecisionLogStorageError, match="not valid JSON"):
        read_outcomes(trade_date=record.trade_date, base_dir=tmp_path)


def test_outcomes_path_rejects_non_iso_trade_date(tmp_path):
    """F6 — path-traversal-style or formatted strings must not resolve."""
    for bad in ("../etc/passwd", "2026/05/23", "today", "2026-13-99"):
        with pytest.raises(DecisionLogStorageError, match="trade_date"):
            outcomes_path(trade_date=bad, base_dir=tmp_path)


def test_predictions_path_rejects_non_iso_trade_date(tmp_path):
    from hot_theme_rotator.decision_log.jsonl_writer import predictions_path
    for bad in ("../etc/passwd", "2026/05/23", "today", "2026-13-99"):
        with pytest.raises(DecisionLogStorageError, match="trade_date"):
            predictions_path(trade_date=bad, base_dir=tmp_path)
