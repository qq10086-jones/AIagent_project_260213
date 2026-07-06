"""Unit tests for the forward-test harness core (ADR-0011 / §16, P19-01)."""
import math

import pytest

from hot_theme_rotator.backtesting.forward_signal_eval import (
    ForwardEvalError,
    clears_hurdle,
    cost_hurdle,
    cross_sectional_dispersion,
    ic_decay,
    net_ic_after_cost,
    rank_ic,
    spearman,
    top_minus_mean_spread,
)


# ── spearman ─────────────────────────────────────────────────────────────────


def test_spearman_perfect_monotonic_is_one():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_spearman_perfect_reverse_is_minus_one():
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_with_ties_matches_hand_calc():
    # ranks(xs)=[1,2.5,2.5,4], ranks(ys)=[1,2,3,4] → r=0.94868...
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.94868, abs=1e-4)


def test_spearman_no_dispersion_returns_none():
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_spearman_length_mismatch_raises():
    with pytest.raises(ForwardEvalError):
        spearman([1, 2, 3], [1, 2])


# ── rank_ic aggregation ──────────────────────────────────────────────────────


def test_rank_ic_detects_positive_signal_with_high_t():
    # 6 days, all strongly positive but not identical → sd>0, large t
    days = [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),       # IC 1.0
        ([1, 2, 3, 4, 5], [1, 2, 3, 5, 4]),       # one swap, <1
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [2, 1, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 3, 2, 4, 5]),
    ]
    r = rank_ic(days)
    assert r.n_days == 6
    assert r.mean_ic > 0.7
    assert r.t_stat > 3.0


def test_rank_ic_noise_averages_near_zero():
    # alternating +1 / -1 ICs → mean ~0, |t| small
    days = [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),       # +1
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),       # -1
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),       # +1
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),       # -1
    ]
    r = rank_ic(days)
    assert r.n_days == 4
    assert abs(r.mean_ic) < 1e-9
    assert abs(r.t_stat) < 2.0


def test_rank_ic_skips_days_below_min_names():
    days = [
        ([1, 2], [1, 2]),                          # too few → skipped
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ]
    r = rank_ic(days, min_names=5)
    assert r.n_days == 2


def test_rank_ic_empty_is_zero():
    r = rank_ic([])
    assert r == (0.0, 0.0, 0, ())


# ── dispersion + cost hurdle ─────────────────────────────────────────────────


def test_cross_sectional_dispersion_known():
    # stdev([1,2,3,4,5], ddof=1) = 1.5811...
    assert cross_sectional_dispersion(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    ) == pytest.approx(1.58113, abs=1e-4)


def test_cost_hurdle_known_value():
    # 0.7 * 0.006 / 0.04 = 0.105
    assert cost_hurdle(0.04, 0.7, 0.006) == pytest.approx(0.105)


def test_cost_hurdle_falls_with_larger_dispersion():
    assert cost_hurdle(0.10, 0.7, 0.006) < cost_hurdle(0.04, 0.7, 0.006)


def test_cost_hurdle_rejects_nonpositive_sigma():
    with pytest.raises(ForwardEvalError):
        cost_hurdle(0.0, 0.7, 0.006)


def test_net_ic_after_cost_value():
    # 0.05*0.10 - 0.7*0.006 = 0.005 - 0.0042 = 0.0008
    assert net_ic_after_cost(0.05, 0.10, 0.7, 0.006) == pytest.approx(0.0008)


def test_clears_hurdle_positive_case():
    assert clears_hurdle(0.05, 0.10, 0.7, 0.006) is True


def test_clears_hurdle_fails_when_net_negative():
    # tiny IC at short horizon (small sigma) cannot clear
    assert clears_hurdle(0.03, 0.025, 0.7, 0.006) is False


def test_negative_ic_never_clears_even_if_large_magnitude():
    # reversal signal: |net| big but ic<0 → must NOT clear as-is (§16)
    assert clears_hurdle(-0.085, 0.06, 0.7, 0.006) is False


# ── top-k spread + ic decay ──────────────────────────────────────────────────


def test_top_minus_mean_spread_known():
    # top2 by score = returns 10,8 (mean 9); universe mean 6 → spread 3
    spread = top_minus_mean_spread([5, 4, 3, 2, 1], [10, 8, 6, 4, 2], k=2)
    assert spread == pytest.approx(3.0)


def test_top_minus_mean_spread_insufficient_returns_none():
    assert top_minus_mean_spread([5, 4], [1, 2], k=5) is None


def test_ic_decay_flags_1d_only_microstructure():
    d = ic_decay({1: 0.05, 3: -0.01, 5: -0.02})
    assert d["only_1d"] is True
    assert d["best_horizon"] == 1


def test_ic_decay_normal_picks_best_horizon():
    d = ic_decay({1: 0.01, 3: 0.03, 5: 0.05})
    assert d["only_1d"] is False
    assert d["best_horizon"] == 5
    assert d["horizons"] == (1, 3, 5)
