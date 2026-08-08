"""P34-08 tests — six-arm trend overlay, validated on constructed price paths."""
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.research.trend_overlay import (  # noqa: E402
    ARM_NAMES,
    TrendOverlayError,
    compare_arms,
    detect_price_jumps,
    longest_clean_segment,
    run_arm,
    simulate_leveraged,
    sma_signal,
    trend_signal,
    vol_target_weights,
)


def _rising(n=300, step=0.003):
    p, out = 100.0, []
    for _ in range(n):
        out.append(p)
        p *= (1 + step)
    return out


def _sawtooth(n=300, amp=0.05, period=10):
    """Whipsaw path: repeated up/down swings with no net drift."""
    return [100.0 * (1 + amp * math.sin(2 * math.pi * i / period)) for i in range(n)]


# --- leverage simulation ----------------------------------------------------

def test_leveraged_returns_apply_factor_daily_with_fee():
    lev = simulate_leveraged([0.01, 0.01], factor=2.0, annual_fee=0.0)
    assert lev == pytest.approx([0.02, 0.02])


def test_leverage_drag_makes_compound_less_than_factor_times_index():
    raw = [0.10, -0.10, 0.10, -0.10]
    lev = simulate_leveraged(raw, factor=2.0, annual_fee=0.0)
    idx = math.prod(1 + r for r in raw)
    lv = math.prod(1 + r for r in lev)
    assert lv < idx, "daily-rebalanced 2x must lose to the index on a chop path"


def test_fee_drag_reduces_returns():
    a = simulate_leveraged([0.0] * 245, factor=2.0, annual_fee=0.0)
    b = simulate_leveraged([0.0] * 245, factor=2.0, annual_fee=0.01)
    assert sum(b) < sum(a)


def test_negative_factor_refused():
    with pytest.raises(TrendOverlayError):
        simulate_leveraged([0.01], factor=-1.0)


# --- signals are causal -----------------------------------------------------

def test_trend_signal_is_false_before_lookback_is_available():
    sig = trend_signal(_rising(50), lookback=20)
    assert sig[:20] == [False] * 20
    assert all(sig[20:])


def test_trend_signal_detects_downtrend():
    prices = list(reversed(_rising(50)))
    assert not any(trend_signal(prices, lookback=10)[10:])


def test_sma_signal_warmup_is_false():
    sig = sma_signal(_rising(50), window=20)
    assert sig[:19] == [False] * 19
    assert sig[25] is True


def test_signal_lengths_match_input():
    assert len(trend_signal(_rising(40), 10)) == 40
    assert len(sma_signal(_rising(40), 10)) == 40


def test_zero_lookback_refused():
    with pytest.raises(TrendOverlayError):
        trend_signal([1.0, 2.0], lookback=0)


# --- vol targeting ----------------------------------------------------------

def test_vol_target_scales_down_in_high_vol():
    calm = [0.002 * (1 if i % 2 else -1) for i in range(100)]
    wild = [0.05 * (1 if i % 2 else -1) for i in range(100)]
    w_calm = vol_target_weights(calm, window=30, target_vol=0.2)[-1]
    w_wild = vol_target_weights(wild, window=30, target_vol=0.2)[-1]
    assert w_wild < w_calm


def test_zero_realized_vol_fails_closed_to_no_position():
    """Exactly-zero variance means stale data, not a free lunch."""
    w = vol_target_weights([0.001] * 100, window=30, target_vol=0.2)
    assert w[-1] == 0.0


def test_vol_target_is_capped():
    w = vol_target_weights([0.0001] * 100, window=30, target_vol=0.2, max_weight=1.0)
    assert max(w) <= 1.0


def test_vol_target_warmup_is_zero():
    w = vol_target_weights([0.01] * 100, window=30, target_vol=0.2)
    assert w[:30] == [0.0] * 30


# --- arm mechanics ----------------------------------------------------------

def test_buy_and_hold_matches_compounded_returns():
    rets = [0.01, 0.02, -0.01]
    res = run_arm("bh", rets, [1.0] * 3)
    assert res.total_return == pytest.approx(math.prod(1 + r for r in rets) - 1)
    assert res.time_in_market == pytest.approx(1.0)


def test_cash_arm_earns_nothing_and_is_out_of_market():
    res = run_arm("cash", [0.05] * 10, [0.0] * 10)
    assert res.total_return == pytest.approx(0.0)
    assert res.time_in_market == pytest.approx(0.0)


def test_switch_cost_is_charged_on_weight_changes_only():
    steady = run_arm("steady", [0.0] * 10, [1.0] * 10, switch_cost_bp=100.0)
    flippy = run_arm("flippy", [0.0] * 10, [1.0, 0.0] * 5, switch_cost_bp=100.0)
    assert flippy.total_return < steady.total_return
    assert flippy.n_switches > steady.n_switches


def test_misaligned_weights_refused():
    with pytest.raises(TrendOverlayError, match="align"):
        run_arm("x", [0.01, 0.02], [1.0])


def test_max_drawdown_is_negative_on_a_loss_path():
    res = run_arm("dd", [0.1, -0.5, 0.1], [1.0] * 3)
    assert res.max_drawdown < 0


# --- whipsaw: the mechanism risk arm 6 exists to measure --------------------

def test_reentry_delay_reduces_switching_on_a_whipsaw_path():
    out = compare_arms(_sawtooth(400), periods_per_year=245, trend_lookback=20,
                       sma_window=20, vol_window=30, reentry_delay=5,
                       switch_cost_bp=20.0)
    plain = out["arms"]["trend_12m_long_cash"]["n_switches"]
    delayed = out["arms"]["trend_with_reentry_delay"]["n_switches"]
    assert delayed < plain


# --- full comparison --------------------------------------------------------

def test_compare_arms_reports_all_six():
    out = compare_arms(_rising(400), periods_per_year=245, trend_lookback=60,
                       sma_window=50, vol_window=30)
    assert set(out["arms"]) == set(ARM_NAMES)
    assert len(ARM_NAMES) == 6


def test_every_arm_is_flagged_as_simulated_leverage():
    out = compare_arms(_rising(400), trend_lookback=60, sma_window=50, vol_window=30)
    assert all(a["leverage_is_simulated"] for a in out["arms"].values())


def test_thin_sample_is_labelled_inadequate():
    out = compare_arms(_rising(300), periods_per_year=245, trend_lookback=245,
                       sma_window=200, vol_window=30)
    assert out["independent_lookback_windows"] < 5
    assert out["sample_adequacy"] == "INADEQUATE"


def test_caveats_disclaim_the_mop_attribution():
    out = compare_arms(_rising(400), trend_lookback=60, sma_window=50, vol_window=30)
    joined = " ".join(out["caveats"])
    assert "MOP" in joined and "NOT" in joined
    assert "SHADOW" in joined


def test_short_series_refused():
    with pytest.raises(TrendOverlayError, match=">= 30"):
        compare_arms([100.0] * 10)


# --- unadjusted corporate actions (a real defect in the price store) --------

def _with_split(n=400, at=200, ratio=10.0):
    p = _rising(n)
    return p[:at] + [x / ratio for x in p[at:]]


def test_split_artifact_is_detected():
    jumps = detect_price_jumps(_with_split())
    assert len(jumps) == 1
    idx, move = jumps[0]
    assert idx == 200 and move < -0.85


def test_clean_series_has_no_jumps():
    assert detect_price_jumps(_rising(400)) == []


def test_compare_arms_refuses_a_series_with_an_unadjusted_split():
    with pytest.raises(TrendOverlayError, match="UNADJUSTED CORPORATE ACTIONS"):
        compare_arms(_with_split(), trend_lookback=60, sma_window=50, vol_window=30)


def test_allow_jumps_is_an_explicit_opt_in():
    out = compare_arms(_with_split(), trend_lookback=60, sma_window=50,
                       vol_window=30, allow_jumps=True)
    assert set(out["arms"]) == set(ARM_NAMES)


def test_longest_clean_segment_skips_the_split():
    prices = _with_split(n=400, at=100)
    a, b = longest_clean_segment(prices)
    # the post-split tail (300) is longer than the pre-split head (100)
    assert (a, b) == (100, 400)
    assert detect_price_jumps(prices[a:b]) == []


def test_clean_segment_of_a_clean_series_is_the_whole_series():
    assert longest_clean_segment(_rising(200)) == (0, 200)


def test_leveraged_overlay_through_a_split_produces_impossible_equity():
    """Why the guard exists: the arithmetic goes past -100% without it."""
    prices = _with_split(n=400, at=200)
    out = compare_arms(prices, trend_lookback=60, sma_window=50, vol_window=30,
                       allow_jumps=True)
    assert out["arms"]["buy_and_hold"]["max_drawdown"] < -1.0


def test_trend_arm_avoids_the_worst_drawdown_on_a_crash_path():
    prices = _rising(200) + [_rising(200)[-1] * (0.99 ** i) for i in range(1, 150)]
    out = compare_arms(prices, periods_per_year=245, trend_lookback=40,
                       sma_window=40, vol_window=30, switch_cost_bp=0.0)
    bh = out["arms"]["buy_and_hold"]["max_drawdown"]
    tr = out["arms"]["trend_12m_long_cash"]["max_drawdown"]
    assert tr > bh, "a trend exit should cut the drawdown on a sustained decline"
