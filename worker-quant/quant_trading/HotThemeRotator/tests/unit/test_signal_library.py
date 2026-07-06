"""Tests for the signal library + gate runner (ADR-0011 / §16, P19-02)."""
import pytest

from hot_theme_rotator.backtesting.signal_library import (
    NameDayRecord,
    cross_signal_rank_corr,
    group_scored_daily,
    make_price_reversal_signal,
    reversal_of_score,
)


def _rec(pid, buy, date="2026-06-01", symbol="1000.T", price=1000.0):
    return NameDayRecord(pid, symbol, date, buy, price)


def test_reversal_of_score_negates_buy():
    recs = [_rec("a", 0.8), _rec("b", 0.3)]
    out = reversal_of_score(recs)
    assert out == {"a": -0.8, "b": -0.3}


def test_group_scored_daily_pairs_scores_with_returns():
    recs = [_rec(f"p{i}", buy=i / 10, date="2026-06-01") for i in range(5)]
    scores = {f"p{i}": i / 10 for i in range(5)}
    rets = {f"p{i}": i / 100 for i in range(5)}
    daily, dret = group_scored_daily(recs, scores, rets, min_names=5)
    assert len(daily) == 1
    s, r = daily[0]
    assert s == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert r == [0.0, 0.01, 0.02, 0.03, 0.04]
    assert dret == [[0.0, 0.01, 0.02, 0.03, 0.04]]


def test_group_scored_daily_drops_names_missing_score_or_return():
    recs = [_rec(f"p{i}", 0.5, date="2026-06-02") for i in range(5)]
    recs.append(_rec("noscore", 0.5, date="2026-06-02"))
    recs.append(_rec("noret", 0.5, date="2026-06-02"))
    scores = {f"p{i}": 0.5 for i in range(5)}
    scores["noret"] = 0.5  # has score but no return
    rets = {f"p{i}": 0.01 for i in range(5)}
    rets["noscore"] = 0.01  # has return but no score
    daily, _ = group_scored_daily(recs, scores, rets, min_names=5)
    assert len(daily) == 1
    assert len(daily[0][0]) == 5  # only the 5 complete names


def test_group_scored_daily_skips_small_days():
    recs = [_rec(f"q{i}", 0.5, date="2026-06-03") for i in range(3)]
    scores = {f"q{i}": 0.5 for i in range(3)}
    rets = {f"q{i}": 0.01 for i in range(3)}
    daily, dret = group_scored_daily(recs, scores, rets, min_names=5)
    assert daily == []
    assert dret == []


# ── independent price-reversal signal (P19-02-02) ────────────────────────────


def test_price_reversal_negates_prior_return_and_skips_missing():
    recs = [_rec("a", 0.5, symbol="1.T"), _rec("b", 0.5, symbol="2.T"), _rec("c", 0.5, symbol="3.T")]
    returns = {"1.T": 0.10, "2.T": -0.05, "3.T": None}
    sig = make_price_reversal_signal(lookback=5, return_lookup=lambda s, d: returns[s])
    out = sig(recs)
    assert out == {"a": -0.10, "b": 0.05}  # up 10% → -0.10 (reversal); 3.T skipped (None)
    assert sig.__name__ == "price_reversal_5d"


def test_cross_signal_rank_corr_identical_is_one():
    recs = [_rec(f"p{i}", 0.5, symbol=f"{i}.T") for i in range(5)]
    a = {f"p{i}": float(i) for i in range(5)}
    assert cross_signal_rank_corr(recs, a, a) == pytest.approx(1.0)


def test_cross_signal_rank_corr_reversed_is_minus_one():
    recs = [_rec(f"p{i}", 0.5, symbol=f"{i}.T") for i in range(5)]
    a = {f"p{i}": float(i) for i in range(5)}
    b = {f"p{i}": float(-i) for i in range(5)}
    assert cross_signal_rank_corr(recs, a, b) == pytest.approx(-1.0)


def test_cross_signal_rank_corr_none_when_too_few():
    recs = [_rec(f"p{i}", 0.5, symbol=f"{i}.T") for i in range(3)]
    a = {f"p{i}": float(i) for i in range(3)}
    assert cross_signal_rank_corr(recs, a, a, min_names=5) is None
