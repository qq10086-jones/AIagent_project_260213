"""Liquidity / illiquidity factors.

Citations
---------
- Amihud (2002). "Illiquidity and stock returns: Cross-section and time-
  series effects." Journal of Financial Markets 5. → ILLIQ = mean(|r_t| /
  dollar_volume_t). Priced premium for illiquid stocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(
    close: pd.Series, volume: pd.Series, window: int = 20,
) -> pd.Series:
    """Amihud ILLIQ = rolling mean of |daily return| / dollar volume.

    dollar_volume ≈ close * volume. Scale: *1e6 so values are readable
    (JPY denomination means raw dollar volumes are huge).
    """
    ret1 = close.pct_change().abs()
    dv = close * volume
    # Avoid divide-by-zero; treat zero-volume days as NaN so they don't dominate
    dv = dv.where(dv > 0)
    ratio = ret1 / dv
    return ratio.rolling(window, min_periods=max(5, window // 2)).mean() * 1e6
