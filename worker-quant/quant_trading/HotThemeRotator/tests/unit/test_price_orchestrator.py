"""Tests for PriceOrchestrator fallback chain + consensus (P10-19 Cycle 1)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.orchestrator import (  # noqa: E402
    PriceOrchestrator,
    PriceOrchestratorError,
)
from hot_theme_rotator.data.external.realtime_price.schema import (  # noqa: E402
    PriceQuote,
)


def _make_quote(symbol="6779.T", price=3015.0, source="yahoo_japan"):
    return PriceQuote(
        symbol=symbol,
        price=price,
        source=source,
        data_ts="2026-05-25T08:30:00+09:00",
        wall_ts="2026-05-25T08:35:00+09:00",
    )


def _fixed_clock(t=0.0):
    state = {"t": t}

    def monotonic():
        return state["t"]

    return monotonic, state


def test_orchestrator_rejects_empty_chain():
    with pytest.raises(ValueError):
        PriceOrchestrator(source_chain=[])


def test_get_quote_returns_first_source_when_it_succeeds():
    fetcher_calls = {"yahoo": 0, "kabutan": 0}

    def yahoo(symbol):
        fetcher_calls["yahoo"] += 1
        return _make_quote(source="yahoo_japan")

    def kabutan(symbol):
        fetcher_calls["kabutan"] += 1
        return _make_quote(source="kabutan", price=3000.0)

    orch = PriceOrchestrator(source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)])
    q = orch.get_quote("6779.T")
    assert q.source == "yahoo_japan"
    assert fetcher_calls["kabutan"] == 0


def test_get_quote_falls_back_when_first_source_raises():
    def yahoo(symbol):
        raise RuntimeError("scrape failed")

    def kabutan(symbol):
        return _make_quote(source="kabutan", price=3000.0)

    orch = PriceOrchestrator(source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)])
    q = orch.get_quote("6779.T")
    assert q.source == "kabutan"


def test_get_quote_raises_when_all_sources_fail():
    def yahoo(symbol):
        raise RuntimeError("yahoo fail")

    def kabutan(symbol):
        raise RuntimeError("kabutan fail")

    orch = PriceOrchestrator(source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)])
    with pytest.raises(PriceOrchestratorError):
        orch.get_quote("6779.T")


def test_cache_serves_within_ttl():
    calls = {"n": 0}

    def yahoo(symbol):
        calls["n"] += 1
        return _make_quote(source="yahoo_japan")

    monotonic, clock = _fixed_clock()
    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo)],
        cache_ttl_seconds=60.0,
        monotonic=monotonic,
    )
    orch.get_quote("6779.T")
    clock["t"] = 30.0  # within TTL
    orch.get_quote("6779.T")
    assert calls["n"] == 1


def test_cache_expires_after_ttl():
    calls = {"n": 0}

    def yahoo(symbol):
        calls["n"] += 1
        return _make_quote(source="yahoo_japan")

    monotonic, clock = _fixed_clock()
    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo)],
        cache_ttl_seconds=60.0,
        monotonic=monotonic,
    )
    orch.get_quote("6779.T")
    clock["t"] = 120.0  # beyond TTL
    orch.get_quote("6779.T")
    assert calls["n"] == 2


def test_cache_keyed_by_symbol():
    calls = {"n": 0}

    def yahoo(symbol):
        calls["n"] += 1
        return _make_quote(symbol=symbol, source="yahoo_japan")

    monotonic, _ = _fixed_clock()
    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo)],
        monotonic=monotonic,
    )
    orch.get_quote("6779.T")
    orch.get_quote("1306.T")
    assert calls["n"] == 2


def test_high_salience_triggers_consensus_check():
    def yahoo(symbol):
        return _make_quote(price=3015.0, source="yahoo_japan")

    def kabutan(symbol):
        return _make_quote(price=3020.0, source="kabutan")  # 0.17% delta, OK

    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)],
        consensus_threshold_pct=1.0,
    )
    q = orch.get_quote("6779.T", high_salience=True)
    assert q.price_uncertain is False
    assert q.source == "yahoo_japan"


def test_high_salience_flags_price_uncertain_on_mismatch():
    def yahoo(symbol):
        return _make_quote(price=3015.0, source="yahoo_japan")

    def kabutan(symbol):
        return _make_quote(price=3100.0, source="kabutan")  # 2.82% delta

    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)],
        consensus_threshold_pct=1.0,
    )
    q = orch.get_quote("6779.T", high_salience=True)
    assert q.price_uncertain is True
    assert "consensus mismatch" in (q.fail_reason or "")
    assert "kabutan" in (q.fail_reason or "")


def test_high_salience_marks_uncertain_when_no_consensus_source_works():
    """Per Codex review 2026-05-25: consensus unavailable for high-salience
    lookup MUST mark price_uncertain=True. Returning primary unflagged would
    be overconfident when the caller explicitly asked for a second opinion.
    """
    def yahoo(symbol):
        return _make_quote(price=3015.0, source="yahoo_japan")

    def kabutan(symbol):
        raise RuntimeError("kabutan fail")

    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)],
        consensus_threshold_pct=1.0,
    )
    q = orch.get_quote("6779.T", high_salience=True)
    assert q.price_uncertain is True
    assert "consensus unavailable" in (q.fail_reason or "")
    assert q.source == "yahoo_japan"


def test_high_salience_marks_uncertain_when_only_primary_source_in_chain():
    """Single-source chain → no possible secondary → consensus unavailable."""
    def yahoo(symbol):
        return _make_quote(price=3015.0, source="yahoo_japan")

    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo)],
        consensus_threshold_pct=1.0,
    )
    q = orch.get_quote("6779.T", high_salience=True)
    assert q.price_uncertain is True
    assert "consensus unavailable" in (q.fail_reason or "")


def test_explicit_consensus_pair_chain_overrides_default():
    def yahoo(symbol):
        return _make_quote(price=3015.0, source="yahoo_japan")

    def kabutan(symbol):
        return _make_quote(price=3100.0, source="kabutan")

    def twelvedata(symbol):
        return _make_quote(price=3014.0, source="twelvedata")

    orch = PriceOrchestrator(
        source_chain=[("yahoo_japan", yahoo), ("kabutan", kabutan)],
        consensus_pair_chain=[("twelvedata", twelvedata)],
        consensus_threshold_pct=1.0,
    )
    q = orch.get_quote("6779.T", high_salience=True)
    # twelvedata agrees with yahoo, so no uncertain flag
    assert q.price_uncertain is False
