"""Tests for Kabutan scraper (P10-19 Cycle 1)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.kabutan_scraper import (  # noqa: E402
    KABUTAN_BASE_URL,
    KabutanParseError,
    kabutan_url,
    parse_kabutan_html,
)


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "realtime_price" / "kabutan_sample.html"


def test_kabutan_url_construction():
    assert kabutan_url("6779.T") == f"{KABUTAN_BASE_URL}?code=6779"


def test_kabutan_url_rejects_bad_symbol():
    with pytest.raises(KabutanParseError):
        kabutan_url("6779")


def test_parse_kabutan_html_from_fixture():
    html = FIXTURE.read_text(encoding="utf-8")
    quote = parse_kabutan_html(
        html, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.symbol == "6779.T"
    assert quote.price == 3015.0
    assert quote.source == "kabutan"


def test_parse_kabutan_marks_data_ts_inferred():
    """Per Codex review 2026-05-25: Kabutan does not expose source ts. inferred=True."""
    html = FIXTURE.read_text(encoding="utf-8")
    quote = parse_kabutan_html(
        html, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.data_ts_inferred is True
    assert quote.data_ts == quote.wall_ts


def test_parse_kabutan_html_rejects_empty():
    with pytest.raises(KabutanParseError):
        parse_kabutan_html("", symbol="6779.T")


def test_parse_kabutan_html_rejects_no_price():
    html = "<html><body><div>no price element</div></body></html>"
    with pytest.raises(KabutanParseError):
        parse_kabutan_html(html, symbol="6779.T")


def test_parse_kabutan_html_handles_comma_thousands():
    html = """
    <html><body>
    <span class="kabuka">12,345 円</span>
    </body></html>
    """
    quote = parse_kabutan_html(
        html, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.price == 12345.0
