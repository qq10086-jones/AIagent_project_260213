"""P35-01 tests — hardened adjusted-return contract."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.adjusted_prices import (  # noqa: E402
    CorporateActionError,
    PriceBar,
    adjust_prices,
    adjusted_returns,
    ambiguous_indices,
    detect_corporate_actions,
    validate_bars,
)


def _bars(closes, volumes=None, start_day=1):
    volumes = volumes or [None] * len(closes)
    return [PriceBar(date=f"2026-03-{start_day + i:02d}", close=c, volume=v)
            for i, (c, v) in enumerate(zip(closes, volumes))]


# --- the real 1306.T case ---------------------------------------------------

def test_1306t_ten_for_one_split_with_volume_is_adjusted():
    bars = _bars([3800.0, 3817.0, 376.4, 380.1],
                 volumes=[1_000_000, 1_100_000, 9_800_000, 9_500_000], start_day=27)
    actions = detect_corporate_actions(bars)
    assert actions[0].kind == "split" and actions[0].factor == 10.0
    assert actions[0].evidence == "volume_agrees"
    rets, _ = adjusted_returns(bars)
    assert all(abs(r) < 0.05 for r in rets)


def test_adjusted_latest_price_is_the_raw_latest_price():
    bars = _bars([3817.0, 376.4, 380.1], volumes=[1e6, 1e7, 9e6])
    adj, _ = adjust_prices(bars)
    assert adj[-1] == 380.1
    assert adj[0] == pytest.approx(381.7)


def test_clean_series_is_untouched():
    adj, actions = adjust_prices(_bars([100.0, 101.0, 99.5, 102.0]))
    assert actions == [] and adj == [100.0, 101.0, 99.5, 102.0]


def test_reverse_split_with_volume_is_adjusted():
    bars = _bars([50.0, 250.0, 252.0], volumes=[5e6, 1e6, 1.1e6])
    actions = detect_corporate_actions(bars)
    assert actions[0].kind == "reverse_split"
    rets, _ = adjusted_returns(bars)
    assert all(abs(r) < 0.05 for r in rets)


# --- fail-closed policy (hardened) ------------------------------------------

def test_integer_ratio_WITHOUT_volume_is_ambiguous():
    """Silence is not corroboration: a clean 2:1 ratio with no volume refuses."""
    bars = _bars([100.0, 50.0, 49.0])
    actions = detect_corporate_actions(bars)
    assert actions[0].kind == "ambiguous"
    assert actions[0].evidence == "volume_unavailable"
    with pytest.raises(CorporateActionError, match="Refusing to adjust"):
        adjust_prices(bars)


def test_no_volume_50pct_crash_is_refused_not_adjusted():
    """A -50% crash with no volume must never be erased as a '2:1 split'."""
    with pytest.raises(CorporateActionError):
        adjusted_returns(_bars([200.0, 100.0, 98.0]))


def test_volume_contradiction_is_ambiguous():
    bars = _bars([100.0, 50.0, 49.0], volumes=[1_000_000, 100_000, 90_000])
    actions = detect_corporate_actions(bars)
    assert actions[0].kind == "ambiguous"
    assert actions[0].evidence == "volume_contradicts"


def test_non_integer_jump_is_ambiguous():
    bars = _bars([100.0, 45.0, 44.0], volumes=[1e6, 2e6, 2e6])
    assert detect_corporate_actions(bars)[0].evidence == "ratio_not_integer"


def test_verified_override_is_explicit_and_auditable():
    """No volume, but the split is externally confirmed via verified_actions."""
    bars = _bars([100.0, 50.0, 49.0])
    adj, actions = adjust_prices(bars, verified_actions={"2026-03-02": 2.0})
    assert actions[0].evidence == "verified_override"
    assert adj[0] == pytest.approx(50.0)
    rets, _ = adjusted_returns(bars, verified_actions={"2026-03-02": 2.0})
    assert all(abs(r) < 0.05 for r in rets)


def test_override_only_applies_to_its_date():
    bars = _bars([100.0, 50.0, 49.0])
    with pytest.raises(CorporateActionError):
        adjust_prices(bars, verified_actions={"2026-03-03": 2.0})


def test_non_strict_reports_ambiguous_and_window_scoping():
    """Contamination is per-window: only windows containing the jump index."""
    bars = _bars([100.0, 101.0, 50.0, 51.0, 52.0])   # jump at index 2, no volume
    adj, actions = adjust_prices(bars, strict=False)
    assert adj == [100.0, 101.0, 50.0, 51.0, 52.0]   # never guesses
    bad = ambiguous_indices(actions)
    assert bad == [2]
    # window [0,2) precedes the jump -> clean; window [1,4) intersects -> contaminated
    assert not any(0 <= i < 2 for i in bad)
    assert any(1 <= i < 4 for i in bad)


# --- input validation --------------------------------------------------------

def test_zero_price_is_refused_not_skipped():
    with pytest.raises(CorporateActionError, match="finite and > 0"):
        validate_bars(_bars([0.0, 100.0]))


def test_nan_and_inf_close_refused():
    with pytest.raises(CorporateActionError):
        validate_bars(_bars([float("nan"), 100.0]))
    with pytest.raises(CorporateActionError):
        validate_bars(_bars([float("inf"), 100.0]))


def test_negative_and_nan_volume_refused():
    with pytest.raises(CorporateActionError, match="volume"):
        validate_bars(_bars([100.0, 101.0], volumes=[-5.0, 1e6]))
    with pytest.raises(CorporateActionError, match="volume"):
        validate_bars(_bars([100.0, 101.0], volumes=[float("nan"), 1e6]))


def test_duplicate_dates_refused():
    bars = [PriceBar("2026-03-01", 100.0), PriceBar("2026-03-01", 101.0)]
    with pytest.raises(CorporateActionError, match="duplicate"):
        validate_bars(bars)


def test_out_of_order_dates_refused():
    bars = [PriceBar("2026-03-02", 100.0), PriceBar("2026-03-01", 101.0)]
    with pytest.raises(CorporateActionError, match="out-of-order"):
        validate_bars(bars)


def test_returns_never_fabricate_zero_from_bad_denominator():
    """The old code silently emitted 0.0 for a <=0 denominator; now it refuses."""
    with pytest.raises(CorporateActionError):
        adjusted_returns(_bars([100.0, 0.0, 101.0]))


# --- multiple actions --------------------------------------------------------

def test_two_sequential_splits_both_adjust():
    bars = _bars([1000.0, 500.0, 505.0, 101.0, 102.0],
                 volumes=[1e6, 2.1e6, 2e6, 9.8e6, 1e7])
    adj, actions = adjust_prices(bars)
    assert [a.factor for a in actions] == [2.0, 5.0]
    rets, _ = adjusted_returns(bars)
    assert all(abs(r) < 0.05 for r in rets)


def test_out_of_window_anomaly_does_not_kill_the_symbol():
    """An early ambiguous jump must not discard windows that never touch it."""
    bars = _bars([100.0, 44.0, 45.0, 46.0, 47.0, 46.5])  # ambiguous at index 1
    _, actions = adjust_prices(bars, strict=False)
    bad = ambiguous_indices(actions)
    late_window = range(3, 6)
    assert not any(i in late_window for i in bad), \
        "windows after the anomaly are computable"
