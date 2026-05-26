"""Tests for Yahoo Japan scraper (P10-19 Cycle 1)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.yahoo_japan_scraper import (  # noqa: E402
    YAHOO_JP_BASE_URL,
    YahooJapanParseError,
    parse_yahoo_japan_html,
    yahoo_japan_url,
)


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "realtime_price" / "yahoo_japan_sample.html"


def test_yahoo_japan_url_construction():
    assert yahoo_japan_url("6779.T") == f"{YAHOO_JP_BASE_URL}/6779.T"


def test_yahoo_japan_url_rejects_bad_symbol():
    with pytest.raises(YahooJapanParseError):
        yahoo_japan_url("6779")


def test_parse_yahoo_japan_html_from_fixture():
    html = FIXTURE.read_text(encoding="utf-8")
    quote = parse_yahoo_japan_html(
        html, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.symbol == "6779.T"
    assert quote.price == 3015.0
    assert quote.source == "yahoo_japan"


def test_parse_yahoo_japan_marks_data_ts_inferred():
    """Per Codex review 2026-05-25: Yahoo Japan does not expose source ts in HTML.
    Parser must flag inferred=True so Rule 12.2 stale-check applies."""
    html = FIXTURE.read_text(encoding="utf-8")
    quote = parse_yahoo_japan_html(
        html, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.data_ts_inferred is True
    assert quote.data_ts == quote.wall_ts


def test_parse_yahoo_japan_html_rejects_empty():
    with pytest.raises(YahooJapanParseError):
        parse_yahoo_japan_html("", symbol="6779.T")


def test_parse_yahoo_japan_html_rejects_no_price_element():
    html = "<html><body><div>no price here</div></body></html>"
    with pytest.raises(YahooJapanParseError):
        parse_yahoo_japan_html(html, symbol="6779.T")


def test_parse_yahoo_japan_html_handles_comma_thousands():
    html = """
    <html><body>
    <span class="StyledNumber__value">12,345</span>
    </body></html>
    """
    quote = parse_yahoo_japan_html(
        html, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.price == 12345.0
