"""Tests for the SKHY ADR watch refresh tool (P20-03 / Rule 11.15).

Injected fetcher only — NO live network. Proves: missing SKHY → pending_listing
(not a crash); fresh SKHY → active; old ts → stale; missing peer → unavailable;
output path reports/adr/adr_watch_{asof}.json; no probability/edge fields;
fetcher exceptions are fail-soft.
"""
import json

import pytest

import tools.refresh_skhy_adr_watch as r

ASOF = "2026-06-25"


def _fetcher(quotes):
    return lambda syms: {s: quotes[s] for s in syms if s in quotes}


def test_skhy_pending_when_no_quote_peers_unavailable():
    p = r.build_adr_watch_payload(ASOF, _fetcher({}))
    assert p["instruments"]["SKHY"]["status"] == "pending_listing"
    assert p["instruments"]["MU"]["status"] == "unavailable"
    assert p["instruments"]["SKHY"]["stale"] is True


def test_skhy_active_when_fresh():
    q = {"SKHY": {"last_price": 100.0, "prev_close": 95.0, "volume": 1e6,
                  "data_ts": "2026-06-25", "source": "yfinance"}}
    p = r.build_adr_watch_payload(ASOF, _fetcher(q))
    sk = p["instruments"]["SKHY"]
    assert sk["status"] == "active"
    assert sk["stale"] is False
    assert sk["overnight_return"] == pytest.approx(100.0 / 95.0 - 1.0)


def test_stale_when_old_timestamp():
    q = {"SKHY": {"last_price": 100.0, "prev_close": 95.0, "data_ts": "2026-06-10", "source": "yf"}}
    p = r.build_adr_watch_payload(ASOF, _fetcher(q))
    assert p["instruments"]["SKHY"]["status"] == "stale"


def test_output_path_and_write(tmp_path):
    p = r.build_adr_watch_payload(ASOF, _fetcher({}))
    path = r.write_adr_watch(p, tmp_path / "adr")
    assert path.name == "adr_watch_2026-06-25.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["asof"] == ASOF


def test_no_forbidden_fields_in_payload():
    q = {"SKHY": {"last_price": 100.0, "prev_close": 95.0, "data_ts": "2026-06-25", "source": "x"}}
    blob = json.dumps(r.build_adr_watch_payload(ASOF, _fetcher(q))).lower()
    for bad in ('"probability"', '"win_rate"', '"winrate"', '"expected_return"',
                '"expectedreturn"', '"edge"'):
        assert bad not in blob


def test_fetcher_exception_is_fail_soft():
    def boom(_syms):
        raise RuntimeError("network down")
    p = r.build_adr_watch_payload(ASOF, boom)
    assert p["instruments"]["SKHY"]["status"] == "pending_listing"
    assert p["instruments"]["NVDA"]["status"] == "unavailable"


def test_does_not_substitute_another_symbol_for_skhy():
    # 000660.KS active, SKHY absent → SKHY stays pending_listing, NOT filled from KR line
    q = {"000660.KS": {"last_price": 200000.0, "prev_close": 195000.0,
                       "data_ts": "2026-06-25", "source": "yf", "currency": "KRW"}}
    p = r.build_adr_watch_payload(ASOF, _fetcher(q))
    assert p["instruments"]["SKHY"]["status"] == "pending_listing"
    assert p["instruments"]["SKHY"]["last_price"] is None
    assert p["instruments"]["000660.KS"]["status"] == "active"
