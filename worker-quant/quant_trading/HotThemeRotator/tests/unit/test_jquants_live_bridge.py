"""Tests for JquantsLiveBridge (P10-16)."""
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.jquants_live_bridge import (  # noqa: E402
    JquantsAuthError,
    JquantsCredentials,
    JquantsFetchError,
    JquantsLiveBridge,
    _normalize_jquants_code,
)
from hot_theme_rotator.data.external.tdnet_rss_adapter import HttpResponse  # noqa: E402


# ---------- ticker normalization ----------


def test_normalize_jquants_code_from_4digit():
    assert _normalize_jquants_code("6779.T") == "67790"


def test_normalize_jquants_code_from_5digit():
    assert _normalize_jquants_code("67790.T") == "67790"


def test_normalize_jquants_code_rejects_missing_suffix():
    with pytest.raises(JquantsFetchError):
        _normalize_jquants_code("6779")


def test_normalize_jquants_code_rejects_non_digits():
    with pytest.raises(JquantsFetchError):
        _normalize_jquants_code("AAPL.T")


# ---------- credentials ----------


def test_credentials_from_env_with_email_password(monkeypatch):
    monkeypatch.setenv("JQUANTS_EMAIL", "test@example.com")
    monkeypatch.setenv("JQUANTS_PASSWORD", "secret")
    monkeypatch.delenv("JQUANTS_REFRESH_TOKEN", raising=False)
    creds = JquantsCredentials.from_env()
    assert creds.email == "test@example.com"
    assert creds.password == "secret"
    assert creds.refresh_token is None


def test_credentials_from_env_with_refresh_token(monkeypatch):
    monkeypatch.setenv("JQUANTS_REFRESH_TOKEN", "tok_abc")
    creds = JquantsCredentials.from_env()
    assert creds.refresh_token == "tok_abc"


# ---------- auth flow ----------


def test_id_token_uses_cached_refresh_token():
    creds = JquantsCredentials(refresh_token="cached_ref")
    calls = []

    def http_post(url, *, json_payload, headers, timeout):
        calls.append(url)
        assert "auth_refresh" in url, (
            "should not call auth_user when refresh_token cached"
        )
        return HttpResponse(200, '{"idToken": "id_xyz"}', {})

    bridge = JquantsLiveBridge(credentials=creds, http_post=http_post)
    assert bridge._get_id_token() == "id_xyz"
    assert len(calls) == 1


def test_id_token_uses_email_password_when_no_refresh_token():
    creds = JquantsCredentials(email="u@x.com", password="pw")
    seen = []

    def http_post(url, *, json_payload, headers, timeout):
        seen.append(url)
        if "auth_user" in url:
            return HttpResponse(200, '{"refreshToken": "ref_abc"}', {})
        if "auth_refresh" in url:
            assert "ref_abc" in url
            return HttpResponse(200, '{"idToken": "id_xyz"}', {})
        raise AssertionError(f"unexpected url {url}")

    bridge = JquantsLiveBridge(credentials=creds, http_post=http_post)
    assert bridge._get_id_token() == "id_xyz"
    assert any("auth_user" in u for u in seen)
    assert any("auth_refresh" in u for u in seen)


def test_id_token_fails_when_no_credentials():
    bridge = JquantsLiveBridge(credentials=JquantsCredentials())
    with pytest.raises(JquantsAuthError):
        bridge._get_id_token()


def test_id_token_fails_on_auth_user_http_error():
    creds = JquantsCredentials(email="u@x.com", password="pw")

    def http_post(url, **kw):
        return HttpResponse(401, "unauthorized", {})

    bridge = JquantsLiveBridge(credentials=creds, http_post=http_post)
    with pytest.raises(JquantsAuthError):
        bridge._get_id_token()


def test_id_token_fails_on_missing_refresh_token_field():
    creds = JquantsCredentials(email="u@x.com", password="pw")

    def http_post(url, **kw):
        if "auth_user" in url:
            return HttpResponse(200, '{"foo": "bar"}', {})
        return HttpResponse(200, '{"idToken": "x"}', {})

    bridge = JquantsLiveBridge(credentials=creds, http_post=http_post)
    with pytest.raises(JquantsAuthError):
        bridge._get_id_token()


def test_id_token_cached_across_fetches():
    creds = JquantsCredentials(refresh_token="ref")
    post_count = {"n": 0}

    def http_post(url, **kw):
        post_count["n"] += 1
        return HttpResponse(200, '{"idToken": "id_xyz"}', {})

    def http_get(url, **kw):
        return HttpResponse(200, json.dumps({"daily_quotes": []}), {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    bridge.fetch_daily_quotes("6779.T", date_from="2026-05-23")
    bridge.fetch_daily_quotes("1306.T", date_from="2026-05-23")
    assert post_count["n"] == 1


# ---------- daily_quotes ----------


_QUOTES_PAYLOAD = {
    "daily_quotes": [
        {
            "Date": "2026-05-23",
            "Code": "67790",
            "Open": 2945.0,
            "High": 2990.0,
            "Low": 2920.0,
            "Close": 2980.0,
            "Volume": 1100000,
            "TurnoverValue": 3267000000,
        },
        {
            "Date": "2026-05-25",
            "Code": "67790",
            "Open": 3010.0,
            "High": 3050.0,
            "Low": 2990.0,
            "Close": 3015.0,
            "Volume": 1300000,
            "TurnoverValue": 3919500000,
        },
        {
            "Date": "2026-05-24",
            "Code": "67790",
            "Open": 2985.0,
            "High": 3025.0,
            "Low": 2960.0,
            "Close": 3000.0,
            "Volume": 1250000,
            "TurnoverValue": 3750000000,
        },
    ]
}


def test_fetch_daily_quotes_returns_sorted_price_bars():
    creds = JquantsCredentials(refresh_token="ref")
    captured = {}

    def http_post(url, **kw):
        return HttpResponse(200, '{"idToken": "id_xyz"}', {})

    def http_get(url, *, headers, timeout):
        captured["url"] = url
        captured["auth"] = headers.get("Authorization")
        return HttpResponse(200, json.dumps(_QUOTES_PAYLOAD), {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    bars = bridge.fetch_daily_quotes(
        "6779.T", date_from="2026-05-23", date_to="2026-05-25"
    )
    assert len(bars) == 3
    assert bars[0].asof == "2026-05-23"
    assert bars[1].asof == "2026-05-24"
    assert bars[2].asof == "2026-05-25"
    assert bars[-1].close == 3015.0
    assert "code=67790" in captured["url"]
    assert captured["auth"] == "Bearer id_xyz"


def test_fetch_daily_quotes_rejects_non_iso_date():
    bridge = JquantsLiveBridge(credentials=JquantsCredentials(refresh_token="ref"))
    with pytest.raises(JquantsFetchError):
        bridge.fetch_daily_quotes("6779.T", date_from="2026/05/23")


def test_fetch_daily_quotes_fails_on_500():
    creds = JquantsCredentials(refresh_token="ref")

    def http_post(url, **kw):
        return HttpResponse(200, '{"idToken": "id"}', {})

    def http_get(url, **kw):
        return HttpResponse(500, "internal error", {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    with pytest.raises(JquantsFetchError):
        bridge.fetch_daily_quotes("6779.T", date_from="2026-05-23")


def test_fetch_daily_quotes_fails_on_missing_key():
    creds = JquantsCredentials(refresh_token="ref")

    def http_post(url, **kw):
        return HttpResponse(200, '{"idToken": "id"}', {})

    def http_get(url, **kw):
        return HttpResponse(200, '{"unexpected": []}', {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    with pytest.raises(JquantsFetchError):
        bridge.fetch_daily_quotes("6779.T", date_from="2026-05-23")


def test_fetch_daily_quotes_skips_malformed_items():
    creds = JquantsCredentials(refresh_token="ref")
    payload = {
        "daily_quotes": [
            {
                "Date": "2026-05-23",
                "Code": "67790",
                "Open": 2945,
                "High": 2990,
                "Low": 2920,
                "Close": 2980,
                "Volume": 1100000,
                "TurnoverValue": 0,
            },
            {"Date": "", "Code": "67790"},  # bad date
            {"Code": "67790"},  # missing Date
            {"Date": "2026-05-25"},  # missing OHLC
        ]
    }

    def http_post(url, **kw):
        return HttpResponse(200, '{"idToken": "id"}', {})

    def http_get(url, **kw):
        return HttpResponse(200, json.dumps(payload), {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    bars = bridge.fetch_daily_quotes("6779.T", date_from="2026-05-23")
    assert len(bars) == 1
    assert bars[0].asof == "2026-05-23"


def test_fetch_daily_quotes_uses_adjustment_fields_when_present():
    creds = JquantsCredentials(refresh_token="ref")
    payload = {
        "daily_quotes": [
            {
                "Date": "2026-05-25",
                "Code": "67790",
                "Open": 3010,
                "High": 3050,
                "Low": 2990,
                "Close": 3015,
                "Volume": 1300000,
                "TurnoverValue": 3919500000,
                "AdjustmentOpen": 1505,
                "AdjustmentHigh": 1525,
                "AdjustmentLow": 1495,
                "AdjustmentClose": 1507.5,
                "AdjustmentVolume": 2600000,
            }
        ]
    }

    def http_post(url, **kw):
        return HttpResponse(200, '{"idToken": "id"}', {})

    def http_get(url, **kw):
        return HttpResponse(200, json.dumps(payload), {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    bars = bridge.fetch_daily_quotes("6779.T", date_from="2026-05-25")
    assert bars[0].close == 1507.5
    assert bars[0].volume == 2600000


def test_fetch_daily_quotes_with_only_date_from():
    """date_to is optional — J-Quants returns single date or open range."""
    creds = JquantsCredentials(refresh_token="ref")
    captured_url = {}

    def http_post(url, **kw):
        return HttpResponse(200, '{"idToken": "id"}', {})

    def http_get(url, **kw):
        captured_url["url"] = url
        return HttpResponse(200, json.dumps({"daily_quotes": []}), {})

    bridge = JquantsLiveBridge(
        credentials=creds, http_post=http_post, http_get=http_get
    )
    bridge.fetch_daily_quotes("6779.T", date_from="2026-05-25")
    assert "from=2026-05-25" in captured_url["url"]
    assert "to=" not in captured_url["url"]
