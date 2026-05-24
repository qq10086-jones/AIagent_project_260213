"""Tests for outcome_join.compute_outcome (§10 gate 4)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.common.schema import PriceBar  # noqa: E402
from hot_theme_rotator.decision_log.outcome_join import (  # noqa: E402
    DEFAULT_HORIZONS_DAYS,
    compute_outcome,
    compute_outcomes,
)
from hot_theme_rotator.decision_log.schema import PredictionRecord  # noqa: E402


def _bar(symbol: str, asof: str, *, close: float, high: float | None = None,
         low: float | None = None) -> PriceBar:
    open_p = close * 0.995
    h = high if high is not None else close * 1.01
    l = low if low is not None else close * 0.99
    return PriceBar.from_dict(
        {
            "symbol": symbol,
            "asof": asof,
            "open": open_p,
            "high": h,
            "low": l,
            "close": close,
            "volume": 1_000_000,
            "turnover_jpy": close * 1_000_000,
        }
    )


def _prediction(**overrides) -> PredictionRecord:
    base = dict(
        symbol="1306.T",
        trade_date="2026-05-23",
        decision_cutoff="2026-05-23T06:00:00+09:00",
        input_snapshot_id="snap-1306.T-2026-05-23",
        model_version="opportunity-v0",
        score_status="uncalibrated_research_score",
        horizon_days=3,
        buy=0.5,
        sell=0.0,
        hold=0.5,
        extra={
            "reference_price": 100.0,
            "ladder": {
                "aggressive_entry": 99.0,
                "balanced_entry": 98.0,
                "conservative_entry": 96.0,
                "stop_price": 94.0,
                "first_exit": 102.0,
                "second_exit": 104.0,
                "stretch_exit": 108.0,
            },
        },
    )
    base.update(overrides)
    return PredictionRecord.build(**base)


class _StubFetcher:
    """Predetermined bars per symbol; raises if symbol is in `error_symbols`.

    Intentionally does NOT pre-filter by date window — real fetchers may
    return slightly stale or malformed bars, and we want `compute_outcome`'s
    own validator to be tested against that, not the test fixture's filter.
    """

    def __init__(self, bars_by_symbol=None, error_symbols=None):
        self._bars = dict(bars_by_symbol or {})
        self._errors = set(error_symbols or ())

    def fetch(self, *, symbol, start_date, end_date):
        if symbol in self._errors:
            raise RuntimeError(f"simulated fetcher failure for {symbol}")
        return list(self._bars.get(symbol, []))


# ----------------------------------------------------------------------------
# Status branches
# ----------------------------------------------------------------------------


def test_compute_outcome_marks_future_cutoff_when_eval_date_before_cutoff():
    fetcher = _StubFetcher()
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-20",  # before trade_date
    )
    assert outcome.status == "future_cutoff"
    assert outcome.realized_returns == {}
    assert outcome.ladder_touches == {}
    assert "before decision cutoff" in outcome.failure_reason


def test_compute_outcome_marks_symbol_not_found_when_fetcher_returns_empty():
    fetcher = _StubFetcher(bars_by_symbol={})
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.status == "symbol_not_found"
    assert "no bars" in outcome.failure_reason


def test_compute_outcome_marks_symbol_not_found_when_fetcher_raises():
    fetcher = _StubFetcher(error_symbols={"1306.T"})
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.status == "symbol_not_found"
    assert "RuntimeError" in outcome.failure_reason


def test_compute_outcome_partial_horizons_when_only_two_bars_available():
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "2026-05-24", close=101.0),
                _bar("1306.T", "2026-05-25", close=102.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-26",
    )
    assert outcome.status == "insufficient_data"
    assert outcome.realized_returns["1D"] == pytest.approx(0.01)
    assert "3D" not in outcome.realized_returns
    assert "5D" not in outcome.realized_returns


def test_compute_outcome_complete_when_all_horizons_evaluable():
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "2026-05-24", close=101.0),
                _bar("1306.T", "2026-05-25", close=102.0),
                _bar("1306.T", "2026-05-26", close=103.0),
                _bar("1306.T", "2026-05-27", close=104.0),
                _bar("1306.T", "2026-05-28", close=105.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.status == "complete"
    assert outcome.realized_returns["1D"] == pytest.approx(0.01)
    assert outcome.realized_returns["3D"] == pytest.approx(0.03)
    assert outcome.realized_returns["5D"] == pytest.approx(0.05)


# ----------------------------------------------------------------------------
# Ladder touch detection
# ----------------------------------------------------------------------------


def test_compute_outcome_marks_entry_tier_touched_when_low_dips_to_level():
    # Dip to 97.5 → balanced_entry (98) touched but conservative (96) is not
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "2026-05-24", close=100.5, high=101.0, low=97.5),
                _bar("1306.T", "2026-05-25", close=101.0),
                _bar("1306.T", "2026-05-26", close=102.0),
                _bar("1306.T", "2026-05-27", close=101.0),
                _bar("1306.T", "2026-05-28", close=100.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.ladder_touches["aggressive_entry"]["touched"] is True
    assert outcome.ladder_touches["balanced_entry"]["touched"] is True
    assert outcome.ladder_touches["conservative_entry"]["touched"] is False
    assert outcome.ladder_touches["stop_price"]["touched"] is False
    assert outcome.ladder_touches["balanced_entry"]["touched_at"] == "2026-05-24"


def test_compute_outcome_marks_exit_tier_touched_when_high_rallies_to_level():
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "2026-05-24", close=102.0, high=103.0, low=99.5),
                _bar("1306.T", "2026-05-25", close=103.5, high=105.0, low=102.0),
                _bar("1306.T", "2026-05-26", close=104.0, high=104.5, low=103.0),
                _bar("1306.T", "2026-05-27", close=104.0),
                _bar("1306.T", "2026-05-28", close=103.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.ladder_touches["first_exit"]["touched"] is True   # 102 hit day 1
    assert outcome.ladder_touches["second_exit"]["touched"] is True  # 104 hit day 2
    assert outcome.ladder_touches["stretch_exit"]["touched"] is False  # 108 never hit


def test_compute_outcome_no_ladder_in_extra_returns_empty_touches():
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [_bar("1306.T", "2026-05-24", close=101.0)]
        }
    )
    pred = _prediction(extra={"reference_price": 100.0})  # no ladder key
    outcome = compute_outcome(pred, fetcher=fetcher, evaluated_as_of="2026-05-30")
    assert outcome.ladder_touches == {}


def test_compute_outcome_fails_closed_when_ladder_is_partial():
    """F4 — opportunity record with a partial ladder is unsafe to evaluate."""
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [_bar("1306.T", f"2026-05-{24+i:02d}", close=100.0+i) for i in range(5)]
        }
    )
    # ladder is present but missing three tiers — silently skipping them used
    # to produce a "complete" outcome with hidden gaps.
    pred = _prediction(
        extra={
            "reference_price": 100.0,
            "ladder": {
                "aggressive_entry": 99.0,
                "balanced_entry": 98.0,
                # missing: conservative_entry, stop_price, first/second/stretch
            },
        }
    )
    outcome = compute_outcome(pred, fetcher=fetcher, evaluated_as_of="2026-05-30")
    assert outcome.status == "malformed_data"
    assert "missing required tiers" in outcome.failure_reason


def test_compute_outcome_fails_closed_when_ladder_tier_is_non_numeric():
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [_bar("1306.T", f"2026-05-{24+i:02d}", close=100.0+i) for i in range(5)]
        }
    )
    pred = _prediction(
        extra={
            "reference_price": 100.0,
            "ladder": {
                "aggressive_entry": "not a number",
                "balanced_entry": 98.0,
                "conservative_entry": 96.0,
                "stop_price": 94.0,
                "first_exit": 102.0,
                "second_exit": 104.0,
                "stretch_exit": 108.0,
            },
        }
    )
    outcome = compute_outcome(pred, fetcher=fetcher, evaluated_as_of="2026-05-30")
    assert outcome.status == "malformed_data"
    assert "not numeric" in outcome.failure_reason


def test_compute_outcome_fails_closed_when_reference_price_missing():
    """F2 — silent fallback to first_bar.open is removed; now fail closed."""
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [_bar("1306.T", "2026-05-24", close=110.0)]
        }
    )
    pred = _prediction(extra={})  # no reference_price, no ladder
    outcome = compute_outcome(pred, fetcher=fetcher, evaluated_as_of="2026-05-30")
    assert outcome.status == "malformed_data"
    assert outcome.realized_returns == {}
    assert outcome.ladder_touches == {}
    assert "reference_price" in outcome.failure_reason


def test_compute_outcome_fails_closed_when_reference_price_non_positive():
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [_bar("1306.T", "2026-05-24", close=110.0)]
        }
    )
    pred = _prediction(extra={"reference_price": 0.0})
    outcome = compute_outcome(pred, fetcher=fetcher, evaluated_as_of="2026-05-30")
    assert outcome.status == "malformed_data"
    assert "reference_price" in outcome.failure_reason


def test_compute_outcome_fails_closed_on_duplicate_bar_asof():
    """F3 — same `asof` appearing twice corrupts horizon indexing."""
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "2026-05-24", close=101.0),
                _bar("1306.T", "2026-05-24", close=102.0),  # duplicate date
                _bar("1306.T", "2026-05-25", close=103.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.status == "malformed_data"
    assert "duplicate" in outcome.failure_reason


def test_compute_outcome_fails_closed_on_non_iso_bar_asof():
    """F3 — non-ISO `asof` (e.g., locale string) silently broke sort."""
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "May 24 2026", close=101.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.status == "malformed_data"
    assert "ISO" in outcome.failure_reason


def test_compute_outcome_fails_closed_on_bar_at_or_before_cutoff():
    """F3 — a bar dated on or before the cutoff would be lookback contamination."""
    fetcher = _StubFetcher(
        bars_by_symbol={
            "1306.T": [
                _bar("1306.T", "2026-05-23", close=100.5),  # same day as cutoff
                _bar("1306.T", "2026-05-24", close=101.0),
            ]
        }
    )
    outcome = compute_outcome(
        _prediction(),
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert outcome.status == "malformed_data"
    assert "strictly after" in outcome.failure_reason


# ----------------------------------------------------------------------------
# Batch + summary
# ----------------------------------------------------------------------------


def test_compute_outcomes_aggregates_status_counts_across_batch():
    bars_okay = [_bar("1306.T", f"2026-05-{24 + i:02d}", close=100.0 + i)
                 for i in range(5)]
    fetcher = _StubFetcher(
        bars_by_symbol={"1306.T": bars_okay},
        error_symbols={"BAD.T"},
    )
    predictions = [
        _prediction(symbol="1306.T"),
        _prediction(
            symbol="BAD.T",
            input_snapshot_id="snap-bad",  # change snapshot so prediction_id differs
        ),
    ]
    summary = compute_outcomes(
        predictions,
        fetcher=fetcher,
        evaluated_as_of="2026-05-30",
    )
    assert summary.evaluated_as_of == "2026-05-30"
    assert summary.horizons_days == DEFAULT_HORIZONS_DAYS
    assert len(summary.outcomes) == 2
    assert summary.status_counts.get("complete", 0) == 1
    assert summary.status_counts.get("symbol_not_found", 0) == 1


def test_compute_outcome_rejects_empty_horizons_tuple():
    fetcher = _StubFetcher()
    with pytest.raises(ValueError, match="horizons_days"):
        compute_outcome(
            _prediction(),
            fetcher=fetcher,
            evaluated_as_of="2026-05-30",
            horizons_days=(),
        )
