"""Tests for the external ADR watch schema (P20-02, Rule 11.15).

Pure, deterministic, dependency-free. SKHY/000660.KS/MU/NVDA/SOX/USDJPY are
EXTERNAL catalyst inputs — never JP candidates, never calibrated edge. The schema
must carry NO probability/win-rate/expected-return/edge fields.
"""
import json

import pytest

from hot_theme_rotator.data.external.adr_watch import (
    ALLOWED_ADR_STATUSES,
    AdrInstrumentSnapshot,
    is_stale,
    overnight_return,
)


def _snap(**kw):
    base = dict(
        symbol="SKHY", role="adr", asof="2026-06-25",
        data_ts="2026-06-25T20:00:00+09:00", status="active",
        last_price=100.0, prev_close=98.0, overnight_return=None,
        volume=1_000.0, volume_z=0.5, currency="USD", source="yfinance",
        stale=False, reasons=(),
    )
    base.update(kw)
    return AdrInstrumentSnapshot(**base)


def test_allowed_statuses_exact():
    assert ALLOWED_ADR_STATUSES == {"pending_listing", "active", "stale", "unavailable"}


def test_valid_external_symbols_accepted():
    for s in ["SKHY", "000660.KS", "MU", "NVDA", "SOXX", "^SOX", "USDJPY=X"]:
        assert _snap(symbol=s).symbol == s


def test_rejects_unknown_status():
    with pytest.raises(ValueError):
        _snap(status="bullish")


def test_rejects_empty_symbol():
    with pytest.raises(ValueError):
        _snap(symbol="")


# ── stale detection from data_ts vs asof ─────────────────────────────────────


def test_is_stale_when_data_ts_missing():
    assert is_stale("2026-06-25", None) is True


def test_is_stale_when_data_ts_old():
    assert is_stale("2026-06-25", "2026-06-10T00:00:00+09:00") is True


def test_not_stale_when_fresh():
    assert is_stale("2026-06-25", "2026-06-25T20:00:00+09:00") is False


def test_not_stale_within_window():
    # 2 days old, default window tolerates it (weekends/holidays)
    assert is_stale("2026-06-25", "2026-06-23T15:00:00+09:00") is False


# ── overnight return helper ──────────────────────────────────────────────────


def test_overnight_return_value():
    assert overnight_return(100.0, 98.0) == pytest.approx(100.0 / 98.0 - 1.0)


def test_overnight_return_none_when_missing_or_zero():
    assert overnight_return(100.0, None) is None
    assert overnight_return(None, 98.0) is None
    assert overnight_return(100.0, 0.0) is None


# ── no forbidden fields + JSON round trip ────────────────────────────────────


def test_no_forbidden_edge_fields():
    d = _snap().to_dict()
    for bad in ("probability", "win_rate", "expected_return", "edge", "winRate", "expectedReturn"):
        assert bad not in d


def test_json_round_trip_preserves_core_fields():
    s = _snap(status="active", currency="USD", source="yfinance",
              reasons=("fresh_quote", "sox_confirmation"))
    d = s.to_dict()
    s2 = AdrInstrumentSnapshot.from_dict(json.loads(json.dumps(d)))
    assert s2.status == "active"
    assert s2.currency == "USD"
    assert s2.source == "yfinance"
    assert s2.reasons == ("fresh_quote", "sox_confirmation")
    assert s2 == s


def test_is_stale_when_data_ts_is_future():
    # P20 fix#3: future-dated data (clock skew / bad feed) is stale, never fresh
    assert is_stale("2026-06-25", "2026-06-27T00:00:00+09:00") is True
    assert is_stale("2026-06-25", "2026-12-31") is True
