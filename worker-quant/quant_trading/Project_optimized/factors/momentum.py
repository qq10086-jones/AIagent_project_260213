"""Momentum & reversal factors.

Citations
---------
- Jegadeesh & Titman (1993). "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." Journal of Finance 48(1).
  → ROC_k factors (k-day simple return).
- Jegadeesh (1990). "Evidence of Predictable Behavior of Security Returns."
  Journal of Finance 45(3). → short-term reversal (1-day).
"""
from __future__ import annotations

import pandas as pd


def roc(close: pd.Series, window: int) -> pd.Series:
    """Simple k-day rate of change: close_t / close_{t-k} - 1.

    This is a RAW lookback return — includes the most recent day. For
    medium-term momentum signals the canonical Jegadeesh-Titman formation
    is "past J months SKIP 1 month" to separate momentum from short-term
    reversal / microstructure noise; use ``jt_momentum`` for that.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    return close.pct_change(window)


def jt_momentum(close: pd.Series, window: int = 120, skip: int = 21) -> pd.Series:
    """Jegadeesh-Titman (1993) canonical momentum: past-``window``-day
    return but skipping the most recent ``skip`` days.

    Default 120 / 21 ≈ past 6 months, skip the last month. This isolates
    medium-term momentum from short-term reversal contamination.

        momentum_t = close_{t-skip} / close_{t-skip-window} - 1
    """
    if window <= 0 or skip < 0:
        raise ValueError("window must be > 0, skip >= 0")
    anchor = close.shift(skip)
    return anchor.pct_change(window)


def reversal(close: pd.Series, window: int = 1) -> pd.Series:
    """Short-term reversal = -1 * k-day return.

    Positive signal indicates recent losers (expected rebounders per
    Jegadeesh 1990).
    """
    return -close.pct_change(window)
