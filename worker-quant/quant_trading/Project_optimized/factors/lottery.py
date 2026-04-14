"""Lottery-like and higher-moment return factors.

Citations
---------
- Bali, Cakici, Whitelaw (2011). "Maxing out: Stocks as lotteries and the
  cross-section of expected returns." JFE 99. → MAX / MIN k-day single-day
  return. Stocks with high MAX empirically underperform (investors overpay
  for lottery-like payoffs).
- Harvey & Siddique (2000). "Conditional Skewness in Asset Pricing Tests."
  Journal of Finance 55. → return skewness priced in the cross-section.
"""
from __future__ import annotations

import pandas as pd


def max_single_day_return(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling max of 1-day returns. Positive = recent lottery-like spike."""
    ret1 = close.pct_change()
    return ret1.rolling(window, min_periods=max(5, window // 2)).max()


def min_single_day_return(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling min of 1-day returns (most negative)."""
    ret1 = close.pct_change()
    return ret1.rolling(window, min_periods=max(5, window // 2)).min()


def return_skew(close: pd.Series, window: int = 60) -> pd.Series:
    """Rolling skewness of daily returns.

    Negative skew = fat left tail (bad); positive skew = lottery-like.
    Harvey-Siddique show systematic skewness is priced.
    """
    ret1 = close.pct_change()
    return ret1.rolling(window, min_periods=max(20, window // 2)).skew()
