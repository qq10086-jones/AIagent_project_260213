"""Tests for P10-17 watchlist event detector."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.health import PriceSourceHealth  # noqa: E402
from hot_theme_rotator.watchlist_intelligence.event_detector import (  # noqa: E402
    detect_watchlist_events,
)


def test_detector_creates_quote_unavailable_event_for_missing_health():
    events = detect_watchlist_events(
        watchlist=("6779.T",),
        trade_date="2026-05-26",
        created_ts="2026-05-26T09:00:00+09:00",
        price_health_rows=(),
        tdnet_disclosures=(),
    )

    assert len(events) == 1
    assert events[0].symbol == "6779.T"
    assert events[0].event_type == "quote_unavailable"
    assert events[0].push_allowed is False


def test_detector_creates_quote_uncertain_event_from_health_row():
    events = detect_watchlist_events(
        watchlist=("6779.T",),
        trade_date="2026-05-26",
        created_ts="2026-05-26T09:00:00+09:00",
        price_health_rows=(
            PriceSourceHealth(
                source="yahoo_japan",
                symbol="6779.T",
                ok=True,
                checked_ts="2026-05-26T09:00:00+09:00",
                price=3015.0,
                data_ts="2026-05-26T09:00:00+09:00",
                wall_ts="2026-05-26T09:00:00+09:00",
                price_uncertain=True,
                fail_reason="consensus unavailable",
            ),
        ),
        tdnet_disclosures=(),
    )

    assert len(events) == 1
    assert events[0].event_type == "quote_uncertain"
    assert events[0].severity == "warning"
    assert events[0].source == "price_health"


def test_detector_creates_tdnet_disclosure_event_for_watch_symbol():
    events = detect_watchlist_events(
        watchlist=("6779.T", "1306.T"),
        trade_date="2026-05-26",
        created_ts="2026-05-26T09:00:00+09:00",
        price_health_rows=(),
        tdnet_disclosures=(
            {
                "ticker": "6779.T",
                "title": "業績予想の修正",
                "published_ts": "2026-05-26T08:30:00+09:00",
            },
        ),
    )

    tdnet_events = [event for event in events if event.event_type == "tdnet_disclosure"]
    assert len(tdnet_events) == 1
    assert tdnet_events[0].symbol == "6779.T"
    assert tdnet_events[0].study_only is True


def test_detector_ignores_disclosures_outside_watchlist():
    events = detect_watchlist_events(
        watchlist=("1306.T",),
        trade_date="2026-05-26",
        created_ts="2026-05-26T09:00:00+09:00",
        price_health_rows=(),
        tdnet_disclosures=(
            {
                "ticker": "6779.T",
                "title": "業績予想の修正",
                "published_ts": "2026-05-26T08:30:00+09:00",
            },
        ),
    )

    assert {event.event_type for event in events} == {"quote_unavailable"}
