"""Tests for tools/poll_tdnet_rss.py CLI (P10-14 Cycle 2)."""
import io
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "tools"
SRC_ROOT = PROJECT_ROOT / "src"
for _candidate in (TOOLS_ROOT, SRC_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import poll_tdnet_rss  # noqa: E402

from hot_theme_rotator.data.external.tdnet_rss_adapter import (  # noqa: E402
    TdnetFetchError,
)
from hot_theme_rotator.data.external.tdnet_schema import (  # noqa: E402
    TdnetDisclosure,
    compute_disclosure_id,
)
from hot_theme_rotator.data.external.tdnet_storage import (  # noqa: E402
    read_disclosures,
)


def _make_disclosure(
    ticker="6779.T", title="業績予想の修正に関するお知らせ", published_ts="2026-05-25T08:30:00"
):
    return TdnetDisclosure(
        disclosure_id=compute_disclosure_id(ticker, published_ts, title),
        ticker=ticker,
        published_ts=published_ts,
        collected_ts="2026-05-25T08:35:00",
        title=title,
        category="earnings",
        url="https://example.com/x.pdf",
    )


class _StubAdapter:
    def __init__(self, responses=None, error_for=None):
        self.responses = responses or {}
        self.error_for = error_for or set()
        self.calls: list[tuple[str, int]] = []

    def fetch_list_for_date(self, trade_date, limit=100):
        self.calls.append((trade_date, limit))
        if trade_date in self.error_for:
            raise TdnetFetchError(f"stub error for {trade_date}")
        return tuple(self.responses.get(trade_date, ()))


def test_iso_date_range_single_day():
    result = list(poll_tdnet_rss.iso_date_range("2026-05-25", "2026-05-25"))
    assert result == ["2026-05-25"]


def test_iso_date_range_multi_day():
    result = list(poll_tdnet_rss.iso_date_range("2026-05-23", "2026-05-25"))
    assert result == ["2026-05-23", "2026-05-24", "2026-05-25"]


def test_iso_date_range_rejects_reversed():
    with pytest.raises(ValueError):
        list(poll_tdnet_rss.iso_date_range("2026-05-25", "2026-05-23"))


def test_resolve_dates_uses_date_flag():
    args = poll_tdnet_rss.parse_args(["--date", "2026-05-25"])
    assert poll_tdnet_rss.resolve_dates(args) == ["2026-05-25"]


def test_resolve_dates_uses_date_range_flag():
    args = poll_tdnet_rss.parse_args(["--date-range", "2026-05-23", "2026-05-25"])
    assert poll_tdnet_rss.resolve_dates(args) == [
        "2026-05-23",
        "2026-05-24",
        "2026-05-25",
    ]


def test_poll_writes_records_via_storage(tmp_path):
    d = _make_disclosure()
    adapter = _StubAdapter(responses={"2026-05-25": [d]})
    out = io.StringIO()

    total = poll_tdnet_rss.poll(
        ["2026-05-25"], adapter=adapter, base_dir=tmp_path, limit=100, out_stream=out
    )
    assert total == 1

    records = read_disclosures(trade_date="2026-05-25", base_dir=tmp_path)
    assert len(records) == 1
    assert records[0].ticker == "6779.T"


def test_poll_handles_empty_response(tmp_path):
    adapter = _StubAdapter(responses={"2026-05-25": []})
    out = io.StringIO()
    total = poll_tdnet_rss.poll(
        ["2026-05-25"], adapter=adapter, base_dir=tmp_path, limit=100, out_stream=out
    )
    assert total == 0
    assert "no disclosures" in out.getvalue()


def test_poll_continues_after_fetch_failure_for_one_date(tmp_path):
    d = _make_disclosure(published_ts="2026-05-26T08:30:00", title="title 26")
    adapter = _StubAdapter(
        responses={"2026-05-26": [d]},
        error_for={"2026-05-25"},
    )
    out = io.StringIO()
    total = poll_tdnet_rss.poll(
        ["2026-05-25", "2026-05-26"],
        adapter=adapter,
        base_dir=tmp_path,
        limit=100,
        out_stream=out,
    )
    assert total == 1
    assert "FETCH FAIL" in out.getvalue()
    assert "wrote 1 disclosures" in out.getvalue()


def test_poll_passes_limit_to_adapter(tmp_path):
    adapter = _StubAdapter(responses={"2026-05-25": []})
    poll_tdnet_rss.poll(
        ["2026-05-25"],
        adapter=adapter,
        base_dir=tmp_path,
        limit=42,
        out_stream=io.StringIO(),
    )
    assert adapter.calls == [("2026-05-25", 42)]
