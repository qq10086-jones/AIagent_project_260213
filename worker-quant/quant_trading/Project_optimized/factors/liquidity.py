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
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
    dollar_volume: pd.Series | None = None,
) -> pd.Series:
    """Amihud (2002) ILLIQ = rolling mean of |daily return| / dollar volume.

    WARNING (Codex audit): computing ``close * volume`` as a proxy for
    dollar volume is sensitive to stock splits / corporate actions.
    A split changes raw volume (shares ×2) and close (¥/2) but leaves
    dollar volume unchanged — only if BOTH series are from the same
    adjustment regime. When the caller has true exchange turnover
    (trading-value / 売買代金), pass it via ``dollar_volume``.

    Output scale: ×1e6 so values are readable (JPY denomination means
    raw Amihud is ~1e-11). Document explicitly so downstream z-score /
    rank transforms do not treat the scale as meaningful.
    """
    ret1 = close.pct_change().abs()
    if dollar_volume is not None:
        dv = dollar_volume
    else:
        dv = close * volume
    # Avoid divide-by-zero; treat zero-volume (suspension / halt) days
    # as NaN so they do not dominate the rolling mean.
    dv = dv.where(dv > 0)
    ratio = ret1 / dv
    # Amihud-specific: tolerate isolated suspension days (volume=0 → NaN)
    # by requiring 70% of window to be valid. This preserves distributional
    # stability (Codex audit) while surviving JP stock 取引停止 events.
    min_valid = max(int(window * 0.7), 5)
    return ratio.rolling(window, min_periods=min_valid).mean() * 1e6
