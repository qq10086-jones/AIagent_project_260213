"""P35-03 — one loader for research-grade adjusted series out of a raw price DB.

Every historical backtest used to open ``daily_prices`` itself and compute
returns on raw closes — which is how three separate tools independently
compounded through 80 un-adjusted corporate actions (1306.T's 10:1 among them).
This loader is the single door: it reads raw bars WITH volume, runs the
``adjusted_prices`` contract per symbol, and hands back adjusted closes plus the
indices of any unresolved jumps so the caller can exclude exactly the windows
that touch them (never silently compute through them, never drop clean windows).

Consumers that need RAW closes (ADV, turnover, reference prices, display) must
not use this loader — that separation is the point of the contract.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

from hot_theme_rotator.data.adjusted_prices import (
    CorporateActionError,
    PriceBar,
    adjust_prices,
    ambiguous_indices,
)

__all__ = ["AdjustedSeries", "load_adjusted_series", "window_is_clean"]


class AdjustedSeries(NamedTuple):
    dates: list[str]
    closes: list[float]          # split-adjusted, current share basis
    ambiguous: list[int]         # bar indices of unresolved jumps
    error: str | None            # validation failure — series unusable at all


def load_adjusted_series(
    db_path: Path | str,
    symbols: Iterable[str] | None = None,
) -> dict[str, AdjustedSeries]:
    """Adjusted (dates, closes, ambiguous, error) per symbol.

    A symbol whose raw series fails validation (disorder, non-finite closes)
    comes back with ``error`` set and empty data — visible, not dropped.
    """
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        # No `close>0` filter in SQL — deliberately. Filtering here would
        # silently launder a corrupted row out of existence; the central
        # `validate_bars` must SEE the bad bar and refuse the series, so the
        # corruption is surfaced per symbol instead of vanishing.
        if symbols is not None:
            syms = sorted(set(symbols))
            q = ",".join("?" * len(syms))
            cur = conn.execute(
                f"select symbol,date,close,volume from daily_prices "
                f"where symbol in ({q}) order by symbol,date", syms)
        else:
            cur = conn.execute(
                "select symbol,date,close,volume from daily_prices "
                "order by symbol,date")
        raw: dict[str, list[PriceBar]] = defaultdict(list)
        for s, d, c, v in cur:
            raw[s].append(PriceBar(
                date=d,
                close=float(c) if c is not None else float("nan"),
                volume=float(v) if v else None))
    finally:
        conn.close()

    out: dict[str, AdjustedSeries] = {}
    for sym, bars in raw.items():
        try:
            adjusted, actions = adjust_prices(bars, strict=False)
        except CorporateActionError as exc:
            out[sym] = AdjustedSeries([], [], [], str(exc))
            continue
        out[sym] = AdjustedSeries(
            dates=[b.date for b in bars],
            closes=adjusted,
            ambiguous=ambiguous_indices(actions),
            error=None,
        )
    return out


def window_is_clean(series: AdjustedSeries, start_idx: int, end_idx: int) -> bool:
    """True iff no unresolved jump falls inside [start_idx, end_idx].

    Per-window contamination: a jump outside the window poisons nothing the
    window computes, so symbols are never excluded wholesale.
    """
    if series.error is not None:
        return False
    return not any(start_idx <= k <= end_idx for k in series.ambiguous)
