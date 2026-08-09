"""P36-02 tests — 決算短信 classification and after-hours event dating."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.earnings_events import (  # noqa: E402
    EarningsEventError,
    classify_tanshin,
    event_date_for,
    extract_earnings_events,
    parse_earnings_event,
    summarize_events,
)

DAYS = ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]


def _disc(title, ts="2026-05-12T15:00:00+09:00", ticker="7203.T"):
    return {"ticker": ticker, "published_ts": ts, "title": title,
            "disclosure_id": "d" * 32}


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("2026年３月期決算短信〔日本基準〕(連結)", "annual"),
    ("令和６年５月期決算短信〔日本基準〕(連結)", "annual"),
    ("2026年12月期 第１四半期決算短信〔日本基準〕(非連結)", "quarterly"),
    ("2026年３月期中間決算短信〔日本基準〕", "quarterly"),
    ("【再訂正】過年度の決算短信等の訂正に関するお知らせ", "correction"),
    ("（訂正）2026年３月期決算短信の一部訂正について", "correction"),
    ("2025年１月期決算短信の開示が期末後50日を超えたことに関するお知らせ", "notice_about_tanshin"),
])
def test_classification(title, expected):
    assert classify_tanshin(title) == expected


def test_non_tanshin_returns_none():
    assert classify_tanshin("自己株式の取得に係る事項の決定に関するお知らせ") is None
    assert classify_tanshin("") is None
    assert classify_tanshin(None) is None


def test_correction_of_a_quarterly_is_a_correction_not_a_quarterly():
    """Order matters: counting it as quarterly would double-count the original."""
    assert classify_tanshin("（訂正）第２四半期決算短信の一部訂正") == "correction"


# --- after-hours event dating (the core PIT rule) ---------------------------

def test_before_close_is_same_day():
    assert event_date_for("2026-05-12T14:00:00+09:00", DAYS) == ("2026-05-12", False)


def test_after_close_moves_to_next_trading_day():
    assert event_date_for("2026-05-12T16:00:00+09:00", DAYS) == ("2026-05-13", True)


def test_exactly_at_close_is_after_close():
    assert event_date_for("2026-05-12T15:30:00+09:00", DAYS) == ("2026-05-13", True)


def test_weekend_publication_lands_on_the_next_trading_day():
    days = ["2026-05-11", "2026-05-15", "2026-05-18"]   # gap = weekend/holiday
    assert event_date_for("2026-05-15T17:00:00+09:00", days) == ("2026-05-18", True)


def test_event_beyond_the_calendar_is_undatable_not_approximated():
    assert event_date_for("2026-12-31T10:00:00+09:00", DAYS) == (None, False)


def test_malformed_timestamp_raises():
    with pytest.raises(EarningsEventError, match="ISO 8601"):
        event_date_for("sometime tuesday", DAYS)


# --- parsing ----------------------------------------------------------------

def test_primary_annual_event():
    ev = parse_earnings_event(_disc("2026年３月期決算短信〔日本基準〕(連結)"), DAYS)
    assert ev.is_primary and ev.subtype == "annual"
    assert ev.event_date == "2026-05-12" and ev.after_close is False


def test_after_close_annual_is_dated_forward():
    ev = parse_earnings_event(
        _disc("2026年３月期決算短信〔日本基準〕(連結)",
              ts="2026-05-12T16:30:00+09:00"), DAYS)
    assert ev.after_close is True and ev.event_date == "2026-05-13"


def test_quarterly_and_correction_are_not_primary():
    q = parse_earnings_event(_disc("2026年12月期 第１四半期決算短信"), DAYS)
    c = parse_earnings_event(_disc("（訂正）決算短信の一部訂正"), DAYS)
    assert q.is_primary is False and c.is_primary is False


def test_non_tanshin_parses_to_none():
    assert parse_earnings_event(_disc("自己株式の取得に係る事項の決定"), DAYS) is None


def test_missing_provenance_raises():
    bad = _disc("2026年３月期決算短信")
    del bad["disclosure_id"]
    with pytest.raises(EarningsEventError, match="provenance"):
        parse_earnings_event(bad, DAYS)


def test_undatable_event_is_dropped_not_raised():
    ev = parse_earnings_event(
        _disc("2026年３月期決算短信", ts="2027-01-01T10:00:00+09:00"), DAYS)
    assert ev is None


# --- extraction + summary ---------------------------------------------------

def test_extract_counts_and_summary():
    discs = [
        _disc("2026年３月期決算短信〔日本基準〕(連結)", ticker="1111.T"),
        _disc("2026年３月期決算短信〔日本基準〕(連結)", ticker="2222.T",
              ts="2026-05-12T16:00:00+09:00"),
        _disc("2026年12月期 第１四半期決算短信", ticker="3333.T"),
        _disc("（訂正）決算短信の訂正", ticker="4444.T"),
        _disc("自己株式の取得に係る事項の決定", ticker="5555.T"),
    ]
    events, skipped = extract_earnings_events(discs, DAYS)
    assert skipped["not_tanshin"] == 1
    s = summarize_events(events)
    assert s["primary_annual"] == 2
    assert s["primary_symbols"] == 2
    assert s["primary_after_close"] == 1
    assert s["primary_after_close_fraction"] == pytest.approx(0.5)
    assert s["by_subtype"]["quarterly"] == 1
    assert s["by_subtype"]["correction"] == 1


def test_empty_input_summary_is_safe():
    s = summarize_events([])
    assert s["primary_annual"] == 0 and s["primary_after_close_fraction"] is None
