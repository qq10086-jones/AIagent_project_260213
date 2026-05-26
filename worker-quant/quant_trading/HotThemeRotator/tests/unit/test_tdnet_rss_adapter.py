"""Tests for YanoshinTdnetAdapter (P10-14 Cycle 2).

All tests use injected fake http_get + sleep + monotonic to avoid any real
network call or wall-clock delay.
"""
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.tdnet_rss_adapter import (  # noqa: E402
    DEFAULT_YANOSHIN_BASE_URL,
    HttpResponse,
    TdnetFetchError,
    YanoshinTdnetAdapter,
)


_VALID_PAYLOAD = json.dumps(
    {
        "items": [
            {
                "company_code": "67790",
                "pubdate": "2026-05-25T08:30:00+09:00",
                "company_name": "日本電波工業",
                "title": "業績予想の修正に関するお知らせ",
                "url": "https://example.com/x.pdf",
            }
        ]
    }
)


def _ok_responder(payload=_VALID_PAYLOAD):
    captured: dict = {}

    def _http_get(url, *, headers, timeout):
        captured["url"] = url
        captured["headers"] = dict(headers)
        captured["timeout"] = timeout
        return HttpResponse(status_code=200, text=payload, headers={})

    return _http_get, captured


def _no_sleep(_seconds):
    pass


def _zero_monotonic():
    return 0.0


def test_fetch_calls_http_get_with_yanoshin_url():
    http_get, captured = _ok_responder()
    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    adapter.fetch_list_for_date("2026-05-25")
    assert DEFAULT_YANOSHIN_BASE_URL in captured["url"]
    assert "20260525" in captured["url"]
    assert "limit=100" in captured["url"]


def test_fetch_passes_user_agent_and_accept_headers():
    http_get, captured = _ok_responder()
    adapter = YanoshinTdnetAdapter(
        http_get=http_get,
        sleep=_no_sleep,
        monotonic=_zero_monotonic,
        user_agent="custom-agent/2.0",
    )
    adapter.fetch_list_for_date("2026-05-25")
    assert captured["headers"]["User-Agent"] == "custom-agent/2.0"
    assert captured["headers"]["Accept"] == "application/json"


def test_fetch_parses_yanoshin_payload_into_disclosures():
    http_get, _ = _ok_responder()
    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    records = adapter.fetch_list_for_date("2026-05-25")
    assert len(records) == 1
    assert records[0].ticker == "6779.T"
    assert records[0].category == "earnings"


def test_fetch_returns_empty_tuple_for_empty_items():
    payload = json.dumps({"items": []})
    http_get, _ = _ok_responder(payload)
    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    assert adapter.fetch_list_for_date("2026-05-25") == ()


def test_fetch_rejects_non_iso_trade_date():
    http_get, _ = _ok_responder()
    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026/05/25")


def test_fetch_rejects_out_of_range_limit():
    http_get, _ = _ok_responder()
    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25", limit=0)
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25", limit=1001)


def test_fetch_fails_on_non_json_response():
    def http_get(url, *, headers, timeout):
        return HttpResponse(status_code=200, text="not json {", headers={})

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25")


def test_fetch_propagates_parser_error_as_fetch_error():
    """If payload lacks `items` key, parser raises TdnetParseError → wrapped."""
    payload = json.dumps({"data": []})

    def http_get(url, *, headers, timeout):
        return HttpResponse(status_code=200, text=payload, headers={})

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25")


def test_fetch_retries_on_429_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    def http_get(url, *, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return HttpResponse(
                status_code=429,
                text="rate limited",
                headers={"Retry-After": "1"},
            )
        return HttpResponse(status_code=200, text=_VALID_PAYLOAD, headers={})

    def sleep(seconds):
        sleeps.append(seconds)

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=sleep, monotonic=_zero_monotonic, max_retries=3
    )
    records = adapter.fetch_list_for_date("2026-05-25")
    assert len(records) == 1
    assert calls["n"] == 3
    assert sleeps[-2:] == [1.0, 1.0]


def test_fetch_retries_on_503_with_backoff():
    calls = {"n": 0}
    sleeps: list[float] = []

    def http_get(url, *, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            return HttpResponse(
                status_code=503, text="unavailable", headers={}
            )
        return HttpResponse(status_code=200, text=_VALID_PAYLOAD, headers={})

    def sleep(seconds):
        sleeps.append(seconds)

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=sleep, monotonic=_zero_monotonic, max_retries=3
    )
    adapter.fetch_list_for_date("2026-05-25")
    assert calls["n"] == 3
    assert sleeps[-2] == 1.0
    assert sleeps[-1] == 2.0


def test_fetch_gives_up_after_max_retries_on_503():
    def http_get(url, *, headers, timeout):
        return HttpResponse(status_code=503, text="unavailable", headers={})

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic, max_retries=2
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25")


def test_fetch_fails_immediately_on_non_retryable_status():
    def http_get(url, *, headers, timeout):
        return HttpResponse(status_code=404, text="not found", headers={})

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25")


def test_fetch_retries_on_network_exception():
    calls = {"n": 0}

    def http_get(url, *, headers, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("DNS failed")
        return HttpResponse(status_code=200, text=_VALID_PAYLOAD, headers={})

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic, max_retries=3
    )
    adapter.fetch_list_for_date("2026-05-25")
    assert calls["n"] == 3


def test_fetch_wraps_network_exception_as_fetch_error_after_max_retries():
    def http_get(url, *, headers, timeout):
        raise ConnectionError("DNS broken")

    adapter = YanoshinTdnetAdapter(
        http_get=http_get, sleep=_no_sleep, monotonic=_zero_monotonic, max_retries=1
    )
    with pytest.raises(TdnetFetchError):
        adapter.fetch_list_for_date("2026-05-25")


def test_rate_limit_sleeps_between_consecutive_requests():
    http_get, _ = _ok_responder()
    sleeps: list[float] = []
    current = {"t": 0.0}

    def sleep(seconds):
        sleeps.append(seconds)
        current["t"] += seconds

    def monotonic():
        return current["t"]

    adapter = YanoshinTdnetAdapter(
        http_get=http_get,
        sleep=sleep,
        monotonic=monotonic,
        rate_limit_seconds=5.0,
    )
    adapter.fetch_list_for_date("2026-05-25")
    current["t"] += 1.0  # simulate 1 second elapsed between calls
    adapter.fetch_list_for_date("2026-05-26")
    assert sleeps[0] == 4.0


def test_rate_limit_does_not_sleep_when_enough_time_elapsed():
    http_get, _ = _ok_responder()
    sleeps: list[float] = []
    current = {"t": 0.0}

    def sleep(seconds):
        sleeps.append(seconds)
        current["t"] += seconds

    def monotonic():
        return current["t"]

    adapter = YanoshinTdnetAdapter(
        http_get=http_get,
        sleep=sleep,
        monotonic=monotonic,
        rate_limit_seconds=5.0,
    )
    adapter.fetch_list_for_date("2026-05-25")
    current["t"] += 10.0  # plenty of time passed
    adapter.fetch_list_for_date("2026-05-26")
    assert sleeps == []
