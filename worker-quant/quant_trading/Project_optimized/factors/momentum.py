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

    Academic canonical horizons used by this project:
      - 3d, 10d: short-to-medium momentum
      - 120d (approx 6 months): Jegadeesh-Titman classic winner horizon
    """
    if window <= 0:
        raise ValueError("window must be positive")
    return close.pct_change(window)


def reversal(close: pd.Series, window: int = 1) -> pd.Series:
    """Short-term reversal = -1 * k-day return.

    Positive signal indicates recent losers (expected rebounders per
    Jegadeesh 1990).
    """
    return -close.pct_change(window)
