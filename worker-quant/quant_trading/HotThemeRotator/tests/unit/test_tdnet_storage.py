"""Tests for TDnet JSONL storage layer (P10-14 Cycle 1)."""
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.tdnet_schema import (  # noqa: E402
    TdnetDisclosure,
    compute_disclosure_id,
)
from hot_theme_rotator.data.external.tdnet_storage import (  # noqa: E402
    TdnetStorageError,
    _trade_date_from_published_ts,
    append_disclosure,
    append_disclosures,
    disclosures_path,
    read_disclosures,
)


def _make_disclosure(
    ticker="6779.T",
    published_ts="2026-05-25T08:30:00",
    title="業績予想の修正に関するお知らせ",
    collected_ts="2026-05-25T08:35:00",
    category="earnings",
    url="https://example.com/disclosure.pdf",
    summary=None,
    company_name=None,
):
    return TdnetDisclosure(
        disclosure_id=compute_disclosure_id(ticker, published_ts, title),
        ticker=ticker,
        published_ts=published_ts,
        title=title,
        collected_ts=collected_ts,
        category=category,
        url=url,
        summary=summary,
        company_name=company_name,
    )


def test_disclosures_path_routes_to_trade_date_file(tmp_path):
    target = disclosures_path(trade_date="2026-05-25", base_dir=tmp_path)
    assert target == tmp_path / "reports" / "tdnet" / "2026-05-25.jsonl"


def test_disclosures_path_rejects_non_iso_trade_date(tmp_path):
    with pytest.raises(TdnetStorageError):
        disclosures_path(trade_date="2026/05/25", base_dir=tmp_path)


def test_disclosures_path_rejects_path_traversal(tmp_path):
    with pytest.raises(TdnetStorageError):
        disclosures_path(trade_date="../../etc/passwd", base_dir=tmp_path)


def test_disclosures_path_rejects_empty_trade_date(tmp_path):
    with pytest.raises(TdnetStorageError):
        disclosures_path(trade_date="", base_dir=tmp_path)


def test_trade_date_helper_extracts_date_from_naive_iso():
    assert _trade_date_from_published_ts("2026-05-25T08:30:00") == "2026-05-25"


def test_trade_date_helper_extracts_date_from_tz_aware_iso():
    assert _trade_date_from_published_ts("2026-05-25T23:59:59+09:00") == "2026-05-25"


def test_trade_date_helper_rejects_non_iso_ts():
    with pytest.raises(TdnetStorageError):
        _trade_date_from_published_ts("not iso")


def test_append_disclosure_creates_file_and_directory(tmp_path):
    d = _make_disclosure()
    path = append_disclosure(d, base_dir=tmp_path)
    assert path.exists()
    assert path.parent.name == "tdnet"
    assert path.name == "2026-05-25.jsonl"


def test_append_disclosure_writes_one_jsonl_line(tmp_path):
    d = _make_disclosure()
    path = append_disclosure(d, base_dir=tmp_path)
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ticker"] == "6779.T"
    assert payload["disclosure_id"] == d.disclosure_id


def test_append_disclosure_rejects_non_TdnetDisclosure_input(tmp_path):
    with pytest.raises(TdnetStorageError):
        append_disclosure({"foo": "bar"}, base_dir=tmp_path)


def test_append_disclosure_rejects_duplicate_disclosure_id(tmp_path):
    d = _make_disclosure()
    append_disclosure(d, base_dir=tmp_path)
    with pytest.raises(TdnetStorageError):
        append_disclosure(d, base_dir=tmp_path)


def test_append_multiple_different_disclosures(tmp_path):
    d1 = _make_disclosure(title="title 1")
    d2 = _make_disclosure(title="title 2")
    append_disclosures([d1, d2], base_dir=tmp_path)
    records = read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)
    assert len(records) == 2


def test_append_disclosures_halts_on_first_duplicate(tmp_path):
    d1 = _make_disclosure(title="title 1")
    d2 = _make_disclosure(title="title 2")
    append_disclosure(d1, base_dir=tmp_path)
    # batch [d2, d1] — d2 writes, then d1 duplicates and raises
    with pytest.raises(TdnetStorageError):
        append_disclosures([d2, d1], base_dir=tmp_path)
    records = read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)
    # d1 from first call + d2 from batch = 2 records
    assert len(records) == 2


def test_read_disclosures_returns_empty_when_file_missing(tmp_path):
    records = read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)
    assert records == ()


def test_read_disclosures_round_trip(tmp_path):
    original = _make_disclosure(summary="summary", company_name="日本電波工業")
    append_disclosure(original, base_dir=tmp_path)
    records = read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)
    assert records == (original,)


def test_read_disclosures_skips_empty_lines(tmp_path):
    d = _make_disclosure()
    append_disclosure(d, base_dir=tmp_path)
    path = disclosures_path(trade_date="2026-05-25", base_dir=tmp_path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n\n")
    records = read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)
    assert len(records) == 1


def test_read_disclosures_fails_closed_on_malformed_jsonl(tmp_path):
    path = disclosures_path(trade_date="2026-05-25", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(TdnetStorageError):
        read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)


def test_read_disclosures_fails_closed_on_schema_violation(tmp_path):
    path = disclosures_path(trade_date="2026-05-25", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    invalid = {
        "disclosure_id": "deadbeefdeadbeef",
        "ticker": "6779.T",
        "published_ts": "2026-05-25T08:30:00",
        "collected_ts": "2026-05-25T08:35:00",
        "title": "title",
        "category": "earnings",
        "url": "https://example.com/x.pdf",
    }
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(TdnetStorageError):
        read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)


def test_append_routes_disclosures_by_published_ts_date(tmp_path):
    d_25 = _make_disclosure(published_ts="2026-05-25T08:30:00", title="title 25")
    d_26 = _make_disclosure(published_ts="2026-05-26T08:30:00", title="title 26")
    append_disclosure(d_25, base_dir=tmp_path)
    append_disclosure(d_26, base_dir=tmp_path)

    file_25 = disclosures_path(trade_date="2026-05-25", base_dir=tmp_path)
    file_26 = disclosures_path(trade_date="2026-05-26", base_dir=tmp_path)
    assert file_25.exists()
    assert file_26.exists()
    assert len(read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)) == 1
    assert len(read_disclosures(trade_date="2026-05-26", base_dir=tmp_path)) == 1


def test_append_disclosure_with_timezone_in_published_ts(tmp_path):
    d = _make_disclosure(published_ts="2026-05-25T08:30:00+09:00")
    path = append_disclosure(d, base_dir=tmp_path)
    assert path.name == "2026-05-25.jsonl"
