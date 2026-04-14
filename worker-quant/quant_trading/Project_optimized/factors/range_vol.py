"""Range / intraday-range volatility estimators.

These are range *proxies* — descriptive indicators of intraday dispersion.
They are NOT the Parkinson (1980) or Alizadeh-Brandt-Diebold (2002) exact
variance estimators (which require sqrt and 1/(4 ln 2) normalisation).
We keep them as plain range proxies and report the Parkinson estimator
separately below for cases that need the literature-consistent variance.

Citations
---------
- Parkinson (1980). "The Extreme Value Method for Estimating the Variance
  of the Rate of Return." Journal of Business 53.
  Parkinson variance estimator: sigma^2 ~= mean(log(H/L)^2) / (4 ln 2).
- Alizadeh, Brandt, Diebold (2002). "Range-Based Estimation of Stochastic
  Volatility Models." Journal of Finance 57. → log(H/L) distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_LOG2_4 = 4.0 * np.log(2.0)


def intraday_range_proxy(high: pd.Series, low: pd.Series, close: pd.Series,
                          window: int = 20) -> pd.Series:
    """Rolling mean of (high - low) / close — range proxy, NOT Parkinson variance."""
    raw = (high - low) / close.where(close > 0)
    return raw.rolling(window, min_periods=window).mean()


def high_low_ratio(high: pd.Series, low: pd.Series,
                   window: int = 20) -> pd.Series:
    """Rolling mean of (H/L - 1) — scale-free range proxy."""
    raw = high / low.where(low > 0) - 1.0
    return raw.rolling(window, min_periods=window).mean()


def parkinson_volatility(high: pd.Series, low: pd.Series,
                          window: int = 20) -> pd.Series:
    """Parkinson (1980) range-based volatility estimator.

    sigma_hat = sqrt( mean(log(H/L)^2) / (4 ln 2) )

    Returns annualised-scale daily volatility estimate. Handles zero/nan
    highs/lows (one-sided limit days) by masking so they don't blow up.
    """
    valid = (high > 0) & (low > 0) & (high >= low)
    log_hl_sq = (np.log(high.where(valid) / low.where(valid))) ** 2
    mean_sq = log_hl_sq.rolling(window, min_periods=window).mean()
    return np.sqrt(mean_sq / _LOG2_4)
