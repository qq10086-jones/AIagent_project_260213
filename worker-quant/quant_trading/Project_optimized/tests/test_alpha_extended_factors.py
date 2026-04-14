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


def test_registry_exposes_eleven_factors() -> None:
    assert len(factor_names()) == 11


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
    # alpha_jt_mom_6m_skip1m (window=120, skip=21) is also longer horizon
    short = compute("alpha_roc_3", df).iloc[-1]
    med = compute("alpha_roc_10", df).iloc[-1]
    long = compute("alpha_jt_mom_6m_skip1m", df).iloc[-1]
    assert long > med > short


def test_jt_momentum_ignores_recent_month_crash() -> None:
    """Codex: JT momentum must NOT react to returns inside the skip window.
    Construct a series with clean 6-month uptrend, then a −20% crash in
    the last 21 days. Raw ROC(120) would be polluted; JT(window=120,
    skip=21) should still reflect the pre-crash trend."""
    from factors.momentum import jt_momentum, roc
    n = 260
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100.0 * 1.002 ** np.arange(n), index=idx)
    # Inject −20% crash over the last 21 days
    close.iloc[-21:] *= np.linspace(1.0, 0.80, 21)

    raw = roc(close, 120).iloc[-1]
    jt = jt_momentum(close, window=120, skip=21).iloc[-1]

    # JT should be clean 6-month trend BEFORE the crash window
    assert jt > 0.15, f"JT should preserve pre-crash uptrend, got {jt:.4f}"
    # Raw ROC is dragged down by the crash (definitely smaller than jt)
    assert raw < jt - 0.05, f"raw={raw:.4f}  jt={jt:.4f}"


def test_amihud_accepts_exchange_turnover_override() -> None:
    """Codex #4: registry must forward dollar_volume when df has the column."""
    df = _synthetic_prices(n=60)
    # Baseline Amihud uses close*volume
    baseline = compute("alpha_amihud_20", df).iloc[-1]
    # Provide explicit dollar_volume = 10x close*volume (much more liquid)
    df2 = df.copy()
    df2["dollar_volume"] = df["close"] * df["volume"] * 10
    override = compute("alpha_amihud_20", df2).iloc[-1]
    # ILLIQ ≈ 1/dollar_volume, so 10x liquid → ~1/10x illiquidity
    assert override == pytest.approx(baseline / 10, rel=1e-6)


def test_parkinson_vol_matches_formula_on_constant_hl_ratio() -> None:
    """If H/L is constant r across the window, sigma_hat = |ln r| / sqrt(4 ln 2)."""
    from factors.range_vol import parkinson_volatility
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    low = pd.Series(100.0, index=idx)
    high = pd.Series(101.0, index=idx)  # H/L = 1.01 constant
    out = parkinson_volatility(high, low, window=20)
    import math
    expected = abs(math.log(1.01)) / math.sqrt(4 * math.log(2))
    assert out.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_parkinson_vol_handles_one_sided_limit_days() -> None:
    """Zero-low or equal-high-low (一字板) days should not blow up."""
    from factors.range_vol import parkinson_volatility
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    high = pd.Series(np.linspace(100, 110, n), index=idx)
    low = pd.Series(np.linspace(99, 109, n), index=idx)
    # Inject a limit day (high == low)
    low.iloc[-5] = high.iloc[-5]
    out = parkinson_volatility(high, low, window=20)
    assert np.isfinite(out.iloc[-1])


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
    out = compute("alpha_range_proxy_20", df)
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
