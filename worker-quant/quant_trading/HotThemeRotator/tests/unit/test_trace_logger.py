"""P11-01 Decision Trace Logger tests (ADR-0007 Layer 1)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.reflection import (  # noqa: E402
    ModuleStep,
    ReflectionTraceError,
    TraceRecord,
    append_trace,
    compute_trace_id,
    read_traces,
    traces_path,
)


def _step(module="opportunity_scanner", branch="passed"):
    return ModuleStep(
        module=module,
        input_summary={"universe_size": 951, "freshness": "ok"},
        output_summary={"ranked": 12, "top_symbol": "1306.T"},
        branch_decision=branch,
    )


def _build_trace(*, action="BUY", reason="aggressive entry tier touched",
                 symbol="1306.T", snapshot_id="abc123def4567890",
                 prediction_id="pred-0123456789abcdef",
                 created_ts="2026-05-26T10:00:00+09:00",
                 trade_date="2026-05-26", chain=None):
    if chain is None:
        chain = (_step(), _step(module="risk_governor", branch="within_limits"))
    tid = compute_trace_id(
        snapshot_id=snapshot_id, prediction_id=prediction_id,
        symbol=symbol, created_ts=created_ts, final_action=action,
    )
    return TraceRecord(
        trace_id=tid, snapshot_id=snapshot_id, prediction_id=prediction_id,
        trade_date=trade_date, created_ts=created_ts, symbol=symbol,
        module_chain=chain, final_action=action, final_reason=reason,
    )


# ─── compute_trace_id ──────────────────────────────────────────────────────


def test_compute_trace_id_is_deterministic():
    a = compute_trace_id(
        snapshot_id="abc", prediction_id="p1", symbol="1306.T",
        created_ts="2026-05-26T10:00:00+09:00", final_action="BUY",
    )
    b = compute_trace_id(
        snapshot_id="abc", prediction_id="p1", symbol="1306.T",
        created_ts="2026-05-26T10:00:00+09:00", final_action="BUY",
    )
    assert a == b
    assert len(a) == 16


def test_compute_trace_id_changes_with_final_action():
    base = compute_trace_id(
        snapshot_id="abc", prediction_id="p1", symbol="1306.T",
        created_ts="2026-05-26T10:00:00+09:00", final_action="BUY",
    )
    other = compute_trace_id(
        snapshot_id="abc", prediction_id="p1", symbol="1306.T",
        created_ts="2026-05-26T10:00:00+09:00", final_action="NO_TRADE",
    )
    assert base != other


# ─── ModuleStep ────────────────────────────────────────────────────────────


def test_module_step_accepts_valid_input():
    s = _step()
    assert s.module == "opportunity_scanner"
    assert s.branch_decision == "passed"


def test_module_step_rejects_empty_module():
    with pytest.raises(ReflectionTraceError, match="module"):
        ModuleStep(module="", input_summary={}, output_summary={}, branch_decision="x")


def test_module_step_rejects_empty_branch_decision():
    with pytest.raises(ReflectionTraceError, match="branch_decision"):
        ModuleStep(module="m", input_summary={}, output_summary={}, branch_decision="")


def test_module_step_rejects_non_mapping_summary():
    with pytest.raises(ReflectionTraceError, match="input_summary"):
        ModuleStep(module="m", input_summary="not a dict",
                   output_summary={}, branch_decision="x")


def test_module_step_roundtrips():
    s = _step()
    restored = ModuleStep.from_dict(s.to_dict())
    assert restored == s


# ─── TraceRecord ───────────────────────────────────────────────────────────


def test_trace_record_accepts_valid_input():
    t = _build_trace()
    assert t.final_action == "BUY"
    assert len(t.module_chain) == 2
    assert t.module_chain[0].module == "opportunity_scanner"


def test_trace_record_rejects_empty_snapshot_id():
    with pytest.raises(ReflectionTraceError, match="snapshot_id"):
        _build_trace(snapshot_id="")


def test_trace_record_allows_empty_prediction_id():
    """NO_TRADE branches legitimately log before any prediction is emitted."""
    t = _build_trace(action="NO_TRADE", prediction_id="")
    assert t.prediction_id == ""


def test_trace_record_rejects_empty_module_chain():
    with pytest.raises(ReflectionTraceError, match="module_chain"):
        _build_trace(chain=())


def test_trace_record_rejects_non_module_step_in_chain():
    with pytest.raises(ReflectionTraceError, match="ModuleStep"):
        _build_trace(chain=("not a step",))


def test_trace_record_rejects_non_iso_trade_date():
    with pytest.raises(ReflectionTraceError, match="trade_date"):
        _build_trace(trade_date="2026/05/26")


def test_trace_record_rejects_naive_created_ts():
    with pytest.raises(ReflectionTraceError, match="timezone"):
        _build_trace(created_ts="2026-05-26T10:00:00")


def test_trace_record_id_mismatch_rejected():
    chain = (_step(),)
    with pytest.raises(ReflectionTraceError, match="trace_id"):
        TraceRecord(
            trace_id="0000000000000000",
            snapshot_id="abc", prediction_id="p1",
            trade_date="2026-05-26", created_ts="2026-05-26T10:00:00+09:00",
            symbol="1306.T", module_chain=chain,
            final_action="BUY", final_reason="r",
        )


def test_trace_record_to_dict_roundtrips():
    t = _build_trace()
    restored = TraceRecord.from_dict(t.to_dict())
    assert restored == t


# ─── writer / reader ───────────────────────────────────────────────────────


def test_append_trace_writes_jsonl(tmp_path):
    t = _build_trace()
    path = append_trace(t, base_dir=tmp_path)
    assert path.exists()
    assert path.name == "2026-05-26.jsonl"
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_append_trace_rejects_duplicate(tmp_path):
    t = _build_trace()
    append_trace(t, base_dir=tmp_path)
    with pytest.raises(ReflectionTraceError, match="duplicate"):
        append_trace(t, base_dir=tmp_path)


def test_append_trace_rejects_non_trace_record(tmp_path):
    with pytest.raises(ReflectionTraceError, match="TraceRecord"):
        append_trace({"foo": "bar"}, base_dir=tmp_path)


def test_read_traces_empty_when_missing(tmp_path):
    assert read_traces("2026-05-26", base_dir=tmp_path) == ()


def test_read_traces_round_trips(tmp_path):
    t1 = _build_trace(action="BUY")
    t2 = _build_trace(action="SELL")
    append_trace(t1, base_dir=tmp_path)
    append_trace(t2, base_dir=tmp_path)
    rows = read_traces("2026-05-26", base_dir=tmp_path)
    assert len(rows) == 2
    assert rows[0] == t1
    assert rows[1] == t2


def test_read_traces_rejects_malformed_line(tmp_path):
    t = _build_trace()
    append_trace(t, base_dir=tmp_path)
    path = traces_path("2026-05-26", base_dir=tmp_path)
    with path.open("a", encoding="utf-8") as h:
        h.write("not json\n")
    with pytest.raises(ReflectionTraceError, match="malformed"):
        read_traces("2026-05-26", base_dir=tmp_path)


def test_read_traces_skips_blank_lines(tmp_path):
    t = _build_trace()
    append_trace(t, base_dir=tmp_path)
    path = traces_path("2026-05-26", base_dir=tmp_path)
    existing = path.read_text(encoding="utf-8")
    path.write_text(existing + "\n\n   \n", encoding="utf-8")
    rows = read_traces("2026-05-26", base_dir=tmp_path)
    assert len(rows) == 1


def test_no_update_or_delete_api():
    """Rule 14.1-style discipline: no mutation API."""
    import hot_theme_rotator.reflection.trace_logger as tl
    forbidden = {"update_trace", "delete_trace", "overwrite_trace"}
    public = {n for n in dir(tl) if not n.startswith("_")}
    assert forbidden.isdisjoint(public)
