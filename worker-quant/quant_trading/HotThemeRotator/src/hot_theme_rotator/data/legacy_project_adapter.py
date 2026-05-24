"""Read-only adapter for the legacy Project_optimized SQLite database."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from hot_theme_rotator.common.schema import NewsItem, PositionSnapshot, PriceBar


class LegacyProjectAdapter:
    """Read normalized records from `Project_optimized/japan_market.db`.

    The adapter never writes to the legacy database. It converts legacy table
    rows into HotThemeRotator schema objects at the boundary.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"legacy database not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_price_bars(
        self,
        asof: str,
        symbols: Sequence[str] | None = None,
    ) -> list[PriceBar]:
        """Return daily price bars for a date, optionally filtered by symbols."""
        query = """
            SELECT symbol, date, open, high, low, close, volume
            FROM daily_prices
            WHERE date = ?
        """
        params: list[object] = [asof]
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        query += " ORDER BY symbol"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        bars: list[PriceBar] = []
        for row in rows:
            close = float(row["close"])
            volume = float(row["volume"])
            bars.append(
                PriceBar.from_dict(
                    {
                        "symbol": row["symbol"],
                        "asof": row["date"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": close,
                        "volume": volume,
                        "turnover_jpy": close * volume,
                    }
                )
            )
        return bars

    def get_news_until(
        self,
        asof_ts: str,
        lookback_days: int = 1,
        symbols: Sequence[str] | None = None,
    ) -> list[NewsItem]:
        """Return news published before `asof_ts` within a lookback window."""
        asof = _parse_datetime(asof_ts)
        start = asof - timedelta(days=int(lookback_days))
        query = """
            SELECT news_id, symbol, published_ts, source, title, content_summary
            FROM news_feed
            WHERE published_ts <= ?
              AND published_ts >= ?
        """
        params: list[object] = [_format_datetime(asof), _format_datetime(start)]
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        query += " ORDER BY published_ts DESC, news_id"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            NewsItem.from_dict(
                {
                    "news_id": row["news_id"],
                    "available_ts": row["published_ts"],
                    "source": row["source"],
                    "headline": row["title"],
                    "body": row["content_summary"] or "",
                    "symbols": [row["symbol"]],
                }
            )
            for row in rows
        ]

    def get_positions(
        self,
        asof: str,
        strategy_id: str | None = None,
    ) -> list[PositionSnapshot]:
        """Return position snapshots for a date and optional strategy id."""
        query = """
            SELECT asof, strategy_id, symbol, qty, avg_cost, market_price,
                   market_value, unrealized_pnl
            FROM positions
            WHERE asof = ?
        """
        params: list[object] = [asof]
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        query += " ORDER BY symbol"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        positions: list[PositionSnapshot] = []
        for row in rows:
            avg_cost = float(row["avg_cost"] or 0.0)
            market_price = float(row["market_price"] or 0.0)
            quantity = float(row["qty"])
            cost_basis = avg_cost * quantity
            unrealized_return = (
                float(row["unrealized_pnl"] or 0.0) / cost_basis
                if cost_basis
                else 0.0
            )
            positions.append(
                PositionSnapshot.from_dict(
                    {
                        "asof": row["asof"],
                        "symbol": row["symbol"],
                        "quantity": quantity,
                        "avg_cost": avg_cost,
                        "market_price": market_price,
                        "market_value": row["market_value"],
                        "unrealized_return": unrealized_return,
                    }
                )
            )
        return positions


def _parse_datetime(value: str) -> datetime:
    normalized = value.strip()
    if len(normalized) == 10:
        normalized = f"{normalized}T23:59:59"
    return datetime.fromisoformat(normalized)


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")

