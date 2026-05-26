"""P10-21 Cycle 2 — Portfolio journal_writer (Rule 14.1 append-only) tests.

Single-writer assumption: this slice does not implement file locking. Per
Rule 14.1 the journal is append-only — UPDATE / DELETE / random-access mutation
APIs are NOT provided, so accidental clobbering is impossible from this module.
Concurrent writers from multiple processes are out of scope and would require a
later cycle's lock layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.portfolio.schema import (  # noqa: E402
    CashEvent,
    FillEntry,
    derive_cash_event_id,
    derive_fill_entry_id,
)
from hot_theme_rotator.portfolio.journal_writer import (  # noqa: E402
    PortfolioJournalError,
    append_cash_event,
    append_fill,
    journal_path,
    read_journal,
)


def _build_fill(*, ts="2026-05-25T14:00:00+09:00", qty=400, price=417.6, side="SELL",
                source="manual", note="user reported broker trade"):
    eid = derive_fill_entry_id(
        ts=ts, symbol="1306.T", side=side, qty=qty, price=price, source=source, note=note
    )
    return FillEntry(
        entry_id=eid, ts=ts, symbol="1306.T", side=side, qty=qty, price=price,
        source=source, fee=0.0, note=note,
    )


def _build_cash(*, ts="2026-06-15T09:00:00+09:00", amount=1500.0, reason="dividend",
                source="manual", note="1306.T June distribution", symbol="1306.T"):
    eid = derive_cash_event_id(
        ts=ts, reason=reason, amount=amount, source=source, note=note, symbol=symbol
    )
    return CashEvent(
        entry_id=eid, ts=ts, amount=amount, reason=reason, source=source, note=note, symbol=symbol
    )


def test_journal_path_uses_iso_trade_date(tmp_path):
    p = journal_path("2026-06-08", base_dir=tmp_path)
    assert p == tmp_path / "reports" / "portfolio" / "journal" / "2026-06-08.jsonl"


def test_journal_path_rejects_non_iso_trade_date(tmp_path):
    with pytest.raises(PortfolioJournalError, match="trade_date"):
        journal_path("2026/06/08", base_dir=tmp_path)
    with pytest.raises(PortfolioJournalError, match="trade_date"):
        journal_path("../escape", base_dir=tmp_path)


def test_append_fill_writes_jsonl_line_and_creates_parent_dir(tmp_path):
    fill = _build_fill()
    written = append_fill(fill, base_dir=tmp_path)
    assert written.exists()
    assert written.parent.is_dir()
    content = written.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1


def test_append_fill_partitions_by_jst_date(tmp_path):
    # UTC midnight on 2026-05-26 = JST 09:00 on 2026-05-26 → partition under 2026-05-26
    fill = _build_fill(ts="2026-05-26T00:00:00+00:00")
    written = append_fill(fill, base_dir=tmp_path)
    assert written.name == "2026-05-26.jsonl"

    # UTC 22:00 on 2026-05-25 = JST 07:00 on 2026-05-26 → also partition under 2026-05-26
    fill2 = _build_fill(ts="2026-05-25T22:00:00+00:00")
    written2 = append_fill(fill2, base_dir=tmp_path)
    assert written2.name == "2026-05-26.jsonl"


def test_append_fill_rejects_duplicate_entry_id(tmp_path):
    fill = _build_fill()
    append_fill(fill, base_dir=tmp_path)
    with pytest.raises(PortfolioJournalError, match="duplicate"):
        append_fill(fill, base_dir=tmp_path)


def test_append_cash_event_writes_jsonl_line(tmp_path):
    cash = _build_cash()
    written = append_cash_event(cash, base_dir=tmp_path)
    assert written.exists()
    assert written.name == "2026-06-15.jsonl"


def test_append_cash_event_rejects_duplicate(tmp_path):
    cash = _build_cash()
    append_cash_event(cash, base_dir=tmp_path)
    with pytest.raises(PortfolioJournalError, match="duplicate"):
        append_cash_event(cash, base_dir=tmp_path)


def test_read_journal_returns_empty_tuple_when_file_missing(tmp_path):
    assert read_journal("2026-06-08", base_dir=tmp_path) == ()


def test_read_journal_round_trips_fill_entry(tmp_path):
    fill = _build_fill()
    append_fill(fill, base_dir=tmp_path)
    rows = read_journal("2026-05-25", base_dir=tmp_path)
    assert len(rows) == 1
    assert isinstance(rows[0], FillEntry)
    assert rows[0] == fill


def test_read_journal_round_trips_cash_event(tmp_path):
    cash = _build_cash()
    append_cash_event(cash, base_dir=tmp_path)
    rows = read_journal("2026-06-15", base_dir=tmp_path)
    assert len(rows) == 1
    assert isinstance(rows[0], CashEvent)
    assert rows[0] == cash


def test_read_journal_handles_mixed_types_in_same_file(tmp_path):
    # Pin all events to same JST trade date
    fill = _build_fill(ts="2026-06-15T09:30:00+09:00")
    cash = _build_cash(ts="2026-06-15T10:00:00+09:00", reason="dividend")
    fill2 = _build_fill(ts="2026-06-15T14:00:00+09:00", side="BUY", qty=100, price=420.0,
                       note="rebalance test")

    append_fill(fill, base_dir=tmp_path)
    append_cash_event(cash, base_dir=tmp_path)
    append_fill(fill2, base_dir=tmp_path)

    rows = read_journal("2026-06-15", base_dir=tmp_path)
    assert len(rows) == 3
    # Append-only ordering preserved
    assert isinstance(rows[0], FillEntry) and rows[0].note == "user reported broker trade"
    assert isinstance(rows[1], CashEvent) and rows[1].reason == "dividend"
    assert isinstance(rows[2], FillEntry) and rows[2].note == "rebalance test"


def test_read_journal_rejects_malformed_line(tmp_path):
    fill = _build_fill()
    append_fill(fill, base_dir=tmp_path)
    path = journal_path("2026-05-25", base_dir=tmp_path)
    with path.open("a", encoding="utf-8") as h:
        h.write("not a json line\n")
    with pytest.raises(PortfolioJournalError, match="malformed"):
        read_journal("2026-05-25", base_dir=tmp_path)


def test_read_journal_rejects_unknown_entry_type(tmp_path):
    path = journal_path("2026-05-25", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"_type": "magic", "foo": "bar"}\n', encoding="utf-8")
    with pytest.raises(PortfolioJournalError, match="entry _type"):
        read_journal("2026-05-25", base_dir=tmp_path)


def test_read_journal_skips_blank_lines(tmp_path):
    fill = _build_fill()
    append_fill(fill, base_dir=tmp_path)
    path = journal_path("2026-05-25", base_dir=tmp_path)
    existing = path.read_text(encoding="utf-8")
    path.write_text(existing + "\n\n   \n", encoding="utf-8")
    rows = read_journal("2026-05-25", base_dir=tmp_path)
    assert len(rows) == 1


def test_journal_writer_provides_no_update_or_delete_api():
    # Rule 14.1: no UPDATE / DELETE / overwrite path may exist.
    import hot_theme_rotator.portfolio.journal_writer as jw
    forbidden = {"update_entry", "delete_entry", "overwrite_journal", "clear_journal"}
    exported = set(getattr(jw, "__all__", []))
    public_attrs = {n for n in dir(jw) if not n.startswith("_")}
    assert forbidden.isdisjoint(exported)
    assert forbidden.isdisjoint(public_attrs)
