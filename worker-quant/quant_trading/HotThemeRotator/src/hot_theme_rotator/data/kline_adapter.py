"""Read-only K-line / OHLC adapter for Project_optimized's daily_prices table.

Source of truth (ADR-0005): `Project_optimized/japan_market.db.daily_prices`,
columns: `symbol, date, open, high, low, close, volume`.

Implements two consumers:
1. `fetch_kline(symbol, window)` — dashboard hero K-line (last N sessions).
2. `LegacyDailyPriceFetcher` — satisfies the `PriceFetcher` Protocol used by
   `decision_log.outcome_join.compute_outcome` (P9-02), so real OHLC drives
   the realized-return / ladder-touch computations once predictions land.

Strictly read-only. SQLite is opened with `mode=ro` URI to prevent any
accidental write path.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from hot_theme_rotator.common.schema import PriceBar


logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"symbol", "date", "open", "high", "low", "close", "volume"}

# Clean N:1 split / 1:N reverse-split factors we recognize when auto-detecting a
# split from the price series (no split calendar is available — see _split_adjust).
_SPLIT_FACTORS = (2, 3, 4, 5, 10, 20, 100)


class KlineAdapterError(RuntimeError):
    """Raised when daily_prices cannot be safely read."""


def fetch_kline(
    db_path: str | Path,
    *,
    symbol: str,
    sessions: int = 40,
) -> tuple[PriceBar, ...]:
    """Return the last `sessions` OHLC bars for `symbol`, chronologically ascending."""
    if not str(symbol).strip():
        raise KlineAdapterError("symbol must be non-empty")
    if int(sessions) <= 0:
        raise KlineAdapterError("sessions must be positive")
    src = _validate_db(db_path)
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        _assert_schema(conn)
        rows = conn.execute(
            """
            SELECT symbol, date, open, high, low, close, volume
            FROM daily_prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (symbol, int(sessions)),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return ()
    bars = [_row_to_bar(r) for r in rows]
    bars.reverse()  # chronologically ascending
    return _split_adjust(tuple(bars))


def fetch_latest_close(
    db_path: str | Path,
    *,
    symbol: str,
) -> PriceBar | None:
    """Single most-recent bar for `symbol`. Returns None if no rows."""
    bars = fetch_kline(db_path, symbol=symbol, sessions=1)
    return bars[-1] if bars else None


@dataclass(frozen=True)
class LegacyDailyPriceFetcher:
    """Satisfies `decision_log.outcome_join.PriceFetcher` Protocol (P9-02).

    Reads `daily_prices` for `symbol` in `[start_date, end_date]` (inclusive).
    """

    db_path: str | Path

    def fetch(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Iterable[PriceBar]:
        if not str(symbol).strip():
            raise KlineAdapterError("symbol must be non-empty")
        # ISO date validation prevents path/injection nasties + catches typos.
        for name, value in (("start_date", start_date), ("end_date", end_date)):
            try:
                date.fromisoformat(str(value))
            except ValueError as exc:
                raise KlineAdapterError(
                    f"{name} must be ISO date (YYYY-MM-DD): {value!r}"
                ) from exc
        if str(start_date) > str(end_date):
            raise KlineAdapterError(
                f"start_date {start_date!r} must be <= end_date {end_date!r}"
            )
        src = _validate_db(self.db_path)
        conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            _assert_schema(conn)
            rows = conn.execute(
                """
                SELECT symbol, date, open, high, low, close, volume
                FROM daily_prices
                WHERE symbol = ?
                  AND date >= ?
                  AND date <= ?
                ORDER BY date ASC
                """,
                (symbol, str(start_date), str(end_date)),
            ).fetchall()
        finally:
            conn.close()
        return tuple(_row_to_bar(r) for r in rows)


@dataclass(frozen=True)
class SplitAdjustedDailyPriceFetcher:
    """Like `LegacyDailyPriceFetcher` but returns a SPLIT-ADJUSTED continuous
    series over `[start_date, end_date]` (Rule 11.9.6).

    Reuses the same conservative heuristic back-adjustment as `fetch_kline`
    (`_split_adjust`): the most recent bar is unchanged, historical bars are
    scaled so the series is continuous across clean N:1 splits. Use this wherever
    a fetched window DERIVES its own reference from the same series (e.g. the
    cohort benchmark), so realized returns are not poisoned by a split cliff.

    NOTE: callers that compare a fetched bar against an EXTERNALLY stored raw
    reference price (e.g. `outcome_join.compute_outcome`, whose reference is the
    emit-time raw close) must NOT rely on this fetcher to fix the basis mismatch
    — they fail closed on an in-window split instead (see `compute_outcome`).
    """

    db_path: str | Path

    def fetch(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Iterable[PriceBar]:
        raw = tuple(
            LegacyDailyPriceFetcher(self.db_path).fetch(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
        )
        return _split_adjust(raw)


def default_db_path(project_optimized_root: str | Path | None = None) -> Path:
    """Default price DB for the whole read layer (K-line, candidate refs, market
    temp, positions, outcomes).

    Prefers the HTR-OWNED ``data/raw/htr_market.db`` — a one-time sibling snapshot
    kept fresh with yfinance-appended trading days (P12-03d) — when it exists, so
    every price read sees recent sessions even while the sibling EOD feed is
    frozen. Falls back to the sibling ``Project_optimized/japan_market.db``
    (ADR-0005) on a fresh checkout where the HTR DB has not been built yet. An
    explicit ``project_optimized_root`` always selects the sibling there.
    """
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "japan_market.db"
    here = Path(__file__).resolve()
    htr_db = here.parents[3] / "data" / "raw" / "htr_market.db"
    if htr_db.exists():
        return htr_db
    return here.parents[4] / "Project_optimized" / "japan_market.db"


# ─── split adjustment ───────────────────────────────────────────────────────


def _detect_split_factor(prev_close: float, cur_close: float) -> float | None:
    """Return the multiplicative factor (cur/prev) to apply to PRE-split bars iff
    the day-over-day move is a clean stock split, else None.

    Triggers when the ratio is within 5% of a clean N:1 (price drops to 1/N) or
    1:N reverse (price jumps N×) factor. The 5% band absorbs the ordinary intraday
    move ON the split day (e.g. 7012.T's 5:1 shows 0.195, not exactly 0.200). It
    is still safe against false positives because JP daily price limits make a
    legitimate single-day close-to-close move near a split ratio (e.g. -50% for a
    2:1) effectively impossible.
    """
    if prev_close <= 0 or cur_close <= 0:
        return None
    r = cur_close / prev_close
    for f in _SPLIT_FACTORS:
        for target in (1.0 / f, float(f)):
            if abs(r - target) / target <= 0.05:
                return r
    return None


def detect_split_factor(prev_close: float, cur_close: float) -> float | None:
    """Public wrapper over the split-detection heuristic.

    Returns the multiplicative ratio (cur/prev) iff the day-over-day move is a
    clean N:1 split / 1:N reverse-split, else None. Single source of the split
    heuristic, reused by `outcome_join` to fail closed on an in-window split
    (Rule 11.9.6) without duplicating the factor table / tolerance band.
    """
    return _detect_split_factor(prev_close, cur_close)


def _split_adjust(bars: tuple[PriceBar, ...]) -> tuple[PriceBar, ...]:
    """Back-adjust OHLC for stock splits so the series is continuous (broker-style
    front adjustment): the most recent prices are unchanged, historical prices are
    scaled to match. Fixes the fake ~10× cliff seen for 1306.T (10:1 split on
    2026-03-30) where ¥3817 pre-split and ¥382 post-split are the same value.

    Heuristic auto-detection — no split calendar is available (ADR-0005: HTR is a
    read-only consumer of Project_optimized's raw daily_prices). It is conservative
    (clean N:1 ratios only) and logs every adjustment for auditability. The robust
    long-term fix is an explicit splits table or source-level adjusted prices.
    Turnover (yen) is split-invariant and preserved; volume (shares) is rescaled.
    """
    n = len(bars)
    if n < 2:
        return bars
    factor = 1.0
    adj = [1.0] * n
    for i in range(n - 1, 0, -1):
        sf = _detect_split_factor(bars[i - 1].close, bars[i].close)
        if sf is not None:
            factor *= sf
            logger.info(
                "kline split-adjust: %s split at %s (ratio %.4f) — back-adjusting "
                "%d prior bars by cumulative %.5f",
                bars[i].symbol, bars[i].asof, sf, i, factor,
            )
        adj[i - 1] = factor
    if all(a == 1.0 for a in adj):
        return bars
    out: list[PriceBar] = []
    for j, b in enumerate(bars):
        a = adj[j]
        if a == 1.0:
            out.append(b)
            continue
        vol = b.volume / a if a else b.volume
        out.append(PriceBar.from_dict({
            "symbol": b.symbol,
            "asof": b.asof,
            "open": b.open * a,
            "high": b.high * a,
            "low": b.low * a,
            "close": b.close * a,
            "volume": vol,
            "turnover_jpy": (b.close * a) * vol,  # = close*volume, split-invariant
        }))
    return tuple(out)


# ─── internals ──────────────────────────────────────────────────────────────


def _validate_db(db_path: str | Path) -> Path:
    src = Path(db_path)
    if not src.exists():
        raise KlineAdapterError(f"japan_market.db not found: {src}")
    return src


def _assert_schema(conn: sqlite3.Connection) -> None:
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices'"
    ).fetchone():
        raise KlineAdapterError("missing required table: daily_prices")
    present = {row["name"] for row in conn.execute("PRAGMA table_info(daily_prices)").fetchall()}
    missing = _REQUIRED_COLUMNS - present
    if missing:
        raise KlineAdapterError(
            f"daily_prices missing required columns: {sorted(missing)}"
        )


def _row_to_bar(row: sqlite3.Row) -> PriceBar:
    close = float(row["close"]) if row["close"] is not None else 0.0
    return PriceBar.from_dict({
        "symbol": str(row["symbol"]),
        "asof": str(row["date"]),
        "open": float(row["open"]) if row["open"] is not None else close,
        "high": float(row["high"]) if row["high"] is not None else close,
        "low": float(row["low"]) if row["low"] is not None else close,
        "close": close,
        "volume": float(row["volume"]) if row["volume"] is not None else 0.0,
        "turnover_jpy": float(row["volume"] or 0.0) * close,
    })
