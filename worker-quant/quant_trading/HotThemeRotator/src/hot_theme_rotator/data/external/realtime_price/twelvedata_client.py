"""TwelveData REST API client (P10-19 Cycle 1).

Free tier: 800 calls/day. Endpoint: https://api.twelvedata.com/price
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from hot_theme_rotator.data.external.realtime_price.schema import (
    PriceQuote,
    PriceQuoteValidationError,
)


TWELVEDATA_BASE_URL = "https://api.twelvedata.com/price"
TWELVEDATA_API_KEY_ENV = "TWELVEDATA_API_KEY"


class TwelveDataError(RuntimeError):
    """Raised when TwelveData request or response cannot be processed."""


def twelvedata_url(symbol: str, *, api_key: str) -> str:
    if not api_key:
        raise TwelveDataError("api_key must be non-empty")
    return f"{TWELVEDATA_BASE_URL}?symbol={symbol}&apikey={api_key}"


def get_api_key_from_env() -> str:
    key = os.environ.get(TWELVEDATA_API_KEY_ENV)
    if not key:
        raise TwelveDataError(
            f"missing environment variable {TWELVEDATA_API_KEY_ENV}"
        )
    return key


def parse_twelvedata_response(
    payload_text: str,
    *,
    symbol: str,
    wall_ts: Optional[str] = None,
) -> PriceQuote:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise TwelveDataError(f"non-JSON response: {exc}") from exc

    if not isinstance(payload, dict):
        raise TwelveDataError(
            f"expected dict, got {type(payload).__name__}"
        )

    if "price" not in payload:
        message = payload.get("message") or payload.get("status") or "no price field"
        raise TwelveDataError(f"TwelveData error: {message}")

    try:
        price = float(payload["price"])
    except (TypeError, ValueError) as exc:
        raise TwelveDataError(
            f"price field not a number: {payload['price']!r}"
        ) from exc

    now_ts = wall_ts or datetime.now(timezone.utc).isoformat()
    # The /price endpoint returns only {"price": "..."}, no source timestamp.
    # If we later switch to /quote endpoint (returns datetime field) we can
    # set data_ts_inferred=False conditionally. For now /price → inferred=True
    # per Codex review 2026-05-25.
    source_ts = payload.get("datetime")
    if source_ts and isinstance(source_ts, str):
        try:
            datetime.fromisoformat(source_ts.replace(" ", "T"))
            data_ts_value = source_ts.replace(" ", "T")
            inferred = False
        except ValueError:
            data_ts_value = now_ts
            inferred = True
    else:
        data_ts_value = now_ts
        inferred = True

    try:
        return PriceQuote(
            symbol=symbol,
            price=price,
            source="twelvedata",
            data_ts=data_ts_value,
            wall_ts=now_ts,
            data_ts_inferred=inferred,
        )
    except PriceQuoteValidationError as exc:
        raise TwelveDataError(
            f"PriceQuote construction failed: {exc}"
        ) from exc
