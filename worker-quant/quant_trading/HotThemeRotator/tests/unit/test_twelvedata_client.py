"""Tests for TwelveData client (P10-19 Cycle 1)."""
import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.twelvedata_client import (  # noqa: E402
    TWELVEDATA_API_KEY_ENV,
    TWELVEDATA_BASE_URL,
    TwelveDataError,
    get_api_key_from_env,
    parse_twelvedata_response,
    twelvedata_url,
)


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "realtime_price" / "twelvedata_sample.json"


def test_twelvedata_url_construction():
    url = twelvedata_url("6779.T", api_key="testkey123")
    assert TWELVEDATA_BASE_URL in url
    assert "symbol=6779.T" in url
    assert "apikey=testkey123" in url


def test_twelvedata_url_rejects_empty_key():
    with pytest.raises(TwelveDataError):
        twelvedata_url("6779.T", api_key="")


def test_get_api_key_from_env_returns_value(monkeypatch):
    monkeypatch.setenv(TWELVEDATA_API_KEY_ENV, "secret123")
    assert get_api_key_from_env() == "secret123"


def test_get_api_key_from_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv(TWELVEDATA_API_KEY_ENV, raising=False)
    with pytest.raises(TwelveDataError):
        get_api_key_from_env()


def test_parse_twelvedata_response_from_fixture():
    payload = FIXTURE.read_text(encoding="utf-8")
    quote = parse_twelvedata_response(
        payload, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.price == 3015.0
    assert quote.source == "twelvedata"


def test_parse_twelvedata_marks_inferred_when_no_datetime_field():
    """The /price endpoint returns only {"price": "..."}, no datetime.
    Per Codex review 2026-05-25: must mark data_ts_inferred=True."""
    payload = '{"price": "3015.00"}'
    quote = parse_twelvedata_response(
        payload, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.data_ts_inferred is True


def test_parse_twelvedata_uses_source_ts_when_datetime_field_present():
    """If response includes a datetime field (e.g., /quote endpoint),
    use it as data_ts and mark inferred=False."""
    payload = '{"price": "3015.00", "datetime": "2026-05-25 08:30:00"}'
    quote = parse_twelvedata_response(
        payload, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.data_ts_inferred is False
    assert "2026-05-25" in quote.data_ts


def test_parse_twelvedata_falls_back_to_inferred_on_bad_datetime():
    """If datetime field exists but is unparseable, fall back to wall_ts+inferred."""
    payload = '{"price": "3015.00", "datetime": "garbage"}'
    quote = parse_twelvedata_response(
        payload, symbol="6779.T", wall_ts="2026-05-25T08:35:00+09:00"
    )
    assert quote.data_ts_inferred is True


def test_parse_twelvedata_response_rejects_non_json():
    with pytest.raises(TwelveDataError):
        parse_twelvedata_response("not json", symbol="6779.T")


def test_parse_twelvedata_response_rejects_non_dict():
    with pytest.raises(TwelveDataError):
        parse_twelvedata_response("[1,2,3]", symbol="6779.T")


def test_parse_twelvedata_response_rejects_missing_price_field():
    with pytest.raises(TwelveDataError):
        parse_twelvedata_response(
            '{"status": "error", "message": "Invalid symbol"}', symbol="6779.T"
        )


def test_parse_twelvedata_response_rejects_non_numeric_price():
    with pytest.raises(TwelveDataError):
        parse_twelvedata_response('{"price": "abc"}', symbol="6779.T")
