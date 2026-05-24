"""Tests for hot_theme_rotator.decision_log.jsonl_writer."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    DecisionLogStorageError,
    append_prediction,
    append_predictions,
    predictions_path,
    read_predictions,
)
from hot_theme_rotator.decision_log.schema import PredictionRecord  # noqa: E402


def _record(**overrides):
    base = dict(
        symbol="1306.T",
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        input_snapshot_id="snap-1306.T-2026-05-23",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=0.5,
        sell=0.2,
        hold=0.3,
        extra={"opportunity_score": 64.2},
    )
    base.update(overrides)
    return PredictionRecord.build(**base)


def test_predictions_path_resolves_under_reports_predictions(tmp_path):
    path = predictions_path(trade_date="2026-05-23", base_dir=tmp_path)
    assert path == tmp_path / "reports" / "predictions" / "2026-05-23.jsonl"


def test_append_then_read_roundtrip_single_record(tmp_path):
    record = _record()
    written_path = append_prediction(record, base_dir=tmp_path)
    assert written_path.exists()
    restored = read_predictions(trade_date=record.trade_date, base_dir=tmp_path)
    assert restored == (record,)


def test_append_multiple_records_preserves_order(tmp_path):
    r1 = _record(symbol="1306.T")
    r2 = _record(symbol="7203.T")
    r3 = _record(symbol="8035.T")
    append_predictions([r1, r2, r3], base_dir=tmp_path)
    restored = read_predictions(trade_date="2026-05-23", base_dir=tmp_path)
    assert tuple(r.symbol for r in restored) == ("1306.T", "7203.T", "8035.T")


def test_read_missing_file_returns_empty_tuple(tmp_path):
    assert read_predictions(trade_date="2026-05-23", base_dir=tmp_path) == ()


def test_append_rejects_duplicate_prediction_id(tmp_path):
    record = _record()
    append_prediction(record, base_dir=tmp_path)
    with pytest.raises(DecisionLogStorageError, match="already present"):
        append_prediction(record, base_dir=tmp_path)


def test_records_for_different_trade_dates_go_to_different_files(tmp_path):
    r1 = _record(
        trade_date="2026-05-22",
        decision_cutoff="2026-05-22T06:00:00+09:00",
        input_snapshot_id="snap-1306.T-2026-05-22",
    )
    r2 = _record(
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        input_snapshot_id="snap-1306.T-2026-05-23",
    )
    append_predictions([r1, r2], base_dir=tmp_path)
    assert read_predictions(trade_date="2026-05-22", base_dir=tmp_path) == (r1,)
    assert read_predictions(trade_date="2026-05-23", base_dir=tmp_path) == (r2,)


def test_append_requires_prediction_record_instance(tmp_path):
    with pytest.raises(DecisionLogStorageError, match="PredictionRecord"):
        append_prediction({"not": "a record"}, base_dir=tmp_path)  # type: ignore[arg-type]


def test_read_fails_closed_on_corrupted_jsonl(tmp_path):
    record = _record()
    path = append_prediction(record, base_dir=tmp_path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{this is not valid json\n")
    with pytest.raises(DecisionLogStorageError, match="not valid JSON"):
        read_predictions(trade_date=record.trade_date, base_dir=tmp_path)
