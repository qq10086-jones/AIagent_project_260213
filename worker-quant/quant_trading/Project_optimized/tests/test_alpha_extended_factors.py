"""v3 A-1 — tests for Alpha-extended factor library.

Each factor is tested on synthetic data where the expected answer is
known analytically (constant-trend series → constant ROC; spike series
→ MAX picks spike; etc.). Property-based invariants where tractable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors import FACTOR_REGISTRY, compute, factor_names


def _synthetic_prices(n: int = 250, start: float = 100.0, trend: float = 0.001,
                       noise: float = 0.0) -> pd.DataFrame:
    """Generate synthetic OHLC+volume. ``noise=0`` gives perfectly geometric
    series (useful for exact-value tests). ``noise>0`` adds per-day lognormal
    noise so skew / std factors have non-degenerate inputs."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    if noise > 0:
        daily = rng.normal(trend, noise, n)
        close = start * np.cumprod(1 + daily)
    else:
        close = start * (1 + trend) ** np.arange(n)
    high = close * (1 + rng.uniform(0.002, 0.01, n))
    low = close * (1 - rng.uniform(0.002, 0.01, n))
    volume = rng.integers(10_000, 1_000_000, n).astype(float)
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=idx)


# ---------- Registry sanity ----------


def test_registry_exposes_ten_factors() -> None:
    assert len(factor_names()) == 10


def test_all_factors_have_citation_and_direction() -> None:
    for n, spec in FACTOR_REGISTRY.items():
        assert spec.citation, f"{n} missing citation"
        assert spec.expected_sign in ("+", "-", "?"), f"{n} bad sign"
        assert spec.description, f"{n} missing description"


def test_unknown_factor_raises() -> None:
    df = _synthetic_prices(50)
    with pytest.raises(KeyError):
        compute("alpha_bogus_factor", df)


# ---------- Momentum ----------


def test_roc_3_on_constant_trend() -> None:
    df = _synthetic_prices(n=50, trend=0.01)
    out = compute("alpha_roc_3", df)
    # Steady 1% trend → 3-day ROC ≈ (1.01)^3 - 1 ≈ 0.0303
    assert out.tail(10).mean() == pytest.approx((1.01) ** 3 - 1, rel=1e-6)


def test_roc_longer_window_bigger_than_short_on_uptrend() -> None:
    df = _synthetic_prices(n=260, trend=0.005)
    last = {n: compute(n, df).iloc[-1] for n in ["alpha_roc_3", "alpha_roc_10", "alpha_roc_120"]}
    assert last["alpha_roc_120"] > last["alpha_roc_10"] > last["alpha_roc_3"]


def test_reversal_1_is_negated_ret1() -> None:
    df = _synthetic_prices(n=30, trend=0.01)
    out = compute("alpha_reversal_1", df)
    expected = -df["close"].pct_change()
    pd.testing.assert_series_equal(out, expected, check_names=False)


# ---------- Lottery ----------


def test_max_ret_20_detects_injected_spike() -> None:
    df = _synthetic_prices(n=60, trend=0.0)
    # Inject a 10% up day at position -5
    close = df["close"].copy()
    close.iloc[-5] = close.iloc[-6] * 1.10
    # Propagate forward
    for i in range(-4, 0):
        close.iloc[i] = close.iloc[i - 1] * 1.001
    df["close"] = close
    out = compute("alpha_max_ret_20", df)
    assert out.iloc[-1] >= 0.08  # captures the ~10% spike


def test_min_ret_20_detects_injected_crash() -> None:
    df = _synthetic_prices(n=60, trend=0.0)
    close = df["close"].copy()
    close.iloc[-5] = close.iloc[-6] * 0.90
    for i in range(-4, 0):
        close.iloc[i] = close.iloc[i - 1] * 0.999
    df["close"] = close
    out = compute("alpha_min_ret_20", df)
    assert out.iloc[-1] <= -0.08


def test_ret_skew_sign_negative_for_crash_series() -> None:
    # Construct returns with one big negative outlier → negative skew
    n = 100
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, n)
    rets[60] = -0.15  # injected crash
    close = 100.0 * np.cumprod(1 + rets)
    df = pd.DataFrame({
        "close": close,
        "high": close * 1.005, "low": close * 0.995,
        "volume": np.full(n, 1e5),
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    out = compute("alpha_ret_skew_60", df)
    assert out.iloc[-1] < -0.5


# ---------- Liquidity ----------


def test_amihud_higher_for_lower_volume() -> None:
    df_hi = _synthetic_prices(n=60)
    df_lo = df_hi.copy()
    df_lo["volume"] = df_hi["volume"] / 100.0  # 100x less liquid
    a_hi = compute("alpha_amihud_20", df_hi).iloc[-1]
    a_lo = compute("alpha_amihud_20", df_lo).iloc[-1]
    assert a_lo > a_hi * 50, "illiquid should be at least 50x higher ILLIQ"


def test_amihud_handles_zero_volume_without_inf() -> None:
    df = _synthetic_prices(n=60)
    df.loc[df.index[-3], "volume"] = 0
    out = compute("alpha_amihud_20", df)
    assert np.isfinite(out.iloc[-1])


# ---------- Range ----------


def test_range_20_positive_for_nonzero_span() -> None:
    df = _synthetic_prices(n=60)
    out = compute("alpha_range_20", df)
    assert (out.dropna() > 0).all()


def test_hl_ratio_20_monotone_with_synthetic_wide_days() -> None:
    df = _synthetic_prices(n=60)
    # Widen last 20 days: high 10% above, low 10% below
    last20 = df.index[-20:]
    df.loc[last20, "high"] = df.loc[last20, "close"] * 1.10
    df.loc[last20, "low"] = df.loc[last20, "close"] * 0.90
    out = compute("alpha_hl_ratio_20", df)
    assert out.iloc[-1] > out.iloc[-21]  # widened period higher than pre


# ---------- Integration: compute all ----------


def test_compute_all_factors_on_same_df_matches_shapes() -> None:
    # Use noisy series so skew/std factors are well-defined
    df = _synthetic_prices(n=260, noise=0.01)
    for name in factor_names():
        s = compute(name, df)
        assert isinstance(s, pd.Series)
        assert s.index.equals(df.index)
        # At least some tail values must be finite
        assert s.tail(20).notna().sum() >= 5, f"{name} mostly NaN at tail"
