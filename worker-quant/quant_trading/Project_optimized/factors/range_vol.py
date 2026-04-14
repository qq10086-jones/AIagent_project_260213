"""Range / intraday-range volatility estimators.

Citations
---------
- Parkinson (1980). "The Extreme Value Method for Estimating the Variance
  of the Rate of Return." Journal of Business 53. → high-low range as a
  more efficient volatility proxy than close-to-close.
- Alizadeh, Brandt, Diebold (2002). "Range-Based Estimation of Stochastic
  Volatility Models." Journal of Finance 57. → log(high/low) distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series,
                   window: int = 20) -> pd.Series:
    """Rolling mean of (high - low) / close. Proxy for realized vol."""
    raw = (high - low) / close.where(close > 0)
    return raw.rolling(window, min_periods=max(5, window // 2)).mean()


def high_low_ratio(high: pd.Series, low: pd.Series,
                   window: int = 20) -> pd.Series:
    """Rolling mean of high/low - 1. Scale-free intraday range."""
    raw = high / low.where(low > 0) - 1.0
    return raw.rolling(window, min_periods=max(5, window // 2)).mean()
