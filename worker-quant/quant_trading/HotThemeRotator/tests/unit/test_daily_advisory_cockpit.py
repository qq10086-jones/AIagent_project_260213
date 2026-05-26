"""Tests for P10-20 Daily Advisory Cockpit payload contracts."""
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.health import PriceSourceHealth  # noqa: E402
from hot_theme_rotator.reporting.daily_advisory_cockpit import (  # noqa: E402
    build_daily_advisory_cockpit,
)


def test_cockpit_is_pull_only_research_payload_with_no_notifications():
    payload = build_daily_advisory_cockpit(
        trade_date="2026-05-26",
        watchlist=("6779.T",),
        price_health_rows=(),
        tdnet_disclosures=(),
    )

    assert payload["activationStage"] == "stage_0_pull_only"
    assert payload["researchOnly"] is True
    assert payload["notificationsInvoked"] is False
    assert payload["execution"]["broker"] is False
    assert payload["execution"]["orders"] is False


def test_cockpit_surfaces_quote_freshness_and_uncertainty():
    payload = build_daily_advisory_cockpit(
        trade_date="2026-05-26",
        watchlist=("6779.T",),
        price_health_rows=(
            PriceSourceHealth(
                source="yahoo_japan",
                symbol="6779.T",
                ok=True,
                checked_ts="2026-05-26T09:00:00+09:00",
                price=3015.0,
                data_ts="2026-05-26T09:00:00+09:00",
                wall_ts="2026-05-26T09:00:00+09:00",
                data_ts_inferred=True,
                price_uncertain=True,
                fail_reason="consensus unavailable",
            ),
        ),
        tdnet_disclosures=(),
    )

    quote = payload["watchlist"][0]["quotes"][0]
    assert quote["source"] == "yahoo_japan"
    assert quote["price"] == 3015.0
    assert quote["dataTsInferred"] is True
    assert quote["freshnessStatus"] == "timestamp_inferred"
    assert quote["priceUncertain"] is True
    assert quote["failReason"] == "consensus unavailable"


def test_cockpit_marks_missing_watchlist_quote_unavailable():
    payload = build_daily_advisory_cockpit(
        trade_date="2026-05-26",
        watchlist=("1306.T",),
        price_health_rows=(),
        tdnet_disclosures=(),
    )

    row = payload["watchlist"][0]
    assert row["symbol"] == "1306.T"
    assert row["quoteStatus"] == "unavailable"
    assert "quote_unavailable" in row["dataGaps"]


def test_cockpit_counts_tdnet_disclosures_per_watch_symbol():
    payload = build_daily_advisory_cockpit(
        trade_date="2026-05-26",
        watchlist=("6779.T", "1306.T"),
        price_health_rows=(),
        tdnet_disclosures=(
            {"ticker": "6779.T", "title": "業績予想の修正", "published_ts": "2026-05-26T08:30:00+09:00"},
            {"ticker": "6779.T", "title": "配当", "published_ts": "2026-05-26T09:10:00+09:00"},
        ),
    )

    rows = {row["symbol"]: row for row in payload["watchlist"]}
    assert rows["6779.T"]["tdnetCount"] == 2
    assert rows["1306.T"]["tdnetCount"] == 0
    assert payload["summary"]["tdnetDisclosures"] == 2


def test_cockpit_rejects_non_iso_trade_date():
    with pytest.raises(ValueError, match="trade_date must be ISO"):
        build_daily_advisory_cockpit(
            trade_date="2026/05/26",
            watchlist=("6779.T",),
            price_health_rows=(),
            tdnet_disclosures=(),
        )


def test_cockpit_does_not_emit_calibrated_win_rate_language():
    payload = build_daily_advisory_cockpit(
        trade_date="2026-05-26",
        watchlist=("6779.T",),
        price_health_rows=(),
        tdnet_disclosures=(),
    )

    rendered = json.dumps(payload, ensure_ascii=False).lower()
    assert "win rate" not in rendered
    assert "probability" not in rendered
    assert payload["calibration"]["status"] == "insufficient_calibration"
