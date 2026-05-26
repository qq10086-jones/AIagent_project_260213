"""Stooq CSV download (P10-19 Cycle 1).

URL: https://stooq.com/q/d/l/?s={symbol_stooq}&i=d
CSV: Date,Open,High,Low,Close,Volume
Latency: ~15 minutes (most stable but slowest of the free options).
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Optional

from hot_theme_rotator.data.external.realtime_price.schema import (
    PriceQuote,
    PriceQuoteValidationError,
)


STOOQ_BASE_URL = "https://stooq.com/q/d/l/"


class StooqParseError(ValueError):
    """Raised when Stooq CSV cannot be parsed into a valid quote."""


def stooq_url(symbol: str) -> str:
    """Stooq uses lowercase 4-digit + '.jp' suffix for JP tickers."""
    if not symbol.endswith(".T"):
        raise StooqParseError(f"symbol must end with '.T', got {symbol!r}")
    code = symbol[:-2].lower()
    return f"{STOOQ_BASE_URL}?s={code}.jp&i=d"


def parse_stooq_csv(
    csv_text: str,
    *,
    symbol: str,
    wall_ts: Optional[str] = None,
) -> PriceQuote:
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise StooqParseError("csv must be a non-empty string")

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        raise StooqParseError("csv has no data rows")

    last_row = rows[-1]
    try:
        price = float(last_row["Close"])
        data_ts_raw = last_row["Date"]
    except (KeyError, ValueError, TypeError) as exc:
        raise StooqParseError(
            f"could not extract Close/Date from last row: {exc}"
        ) from exc

    # Stooq dates are YYYY-MM-DD; always normalize to full ISO datetime so
    # downstream consumers can compare data_ts uniformly.
    try:
        parsed = datetime.fromisoformat(data_ts_raw)
    except ValueError as exc:
        raise StooqParseError(
            f"could not parse Date {data_ts_raw!r}"
        ) from exc
    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and "T" not in data_ts_raw:
        data_ts = f"{data_ts_raw}T00:00:00"
    else:
        data_ts = parsed.isoformat()

    now_ts = wall_ts or datetime.now(timezone.utc).isoformat()
    try:
        return PriceQuote(
            symbol=symbol,
            price=price,
            source="stooq",
            data_ts=data_ts,
            wall_ts=now_ts,
        )
    except PriceQuoteValidationError as exc:
        raise StooqParseError(
            f"PriceQuote construction failed: {exc}"
        ) from exc
