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

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from hot_theme_rotator.common.schema import PriceBar


_REQUIRED_COLUMNS = {"symbol", "date", "open", "high", "low", "close", "volume"}


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
    return tuple(bars)


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


def default_db_path(project_optimized_root: str | Path | None = None) -> Path:
    """Default location of `japan_market.db` (shared with position_adapter)."""
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "japan_market.db"
    here = Path(__file__).resolve()
    return here.parents[4] / "Project_optimized" / "japan_market.db"


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
