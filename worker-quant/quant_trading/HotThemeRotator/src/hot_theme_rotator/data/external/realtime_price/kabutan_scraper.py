"""Kabutan scraper for delayed price quotes (P10-19 Cycle 1).

URL pattern: https://kabutan.jp/stock/?code={4-digit-code}
Latency: ~5 minutes.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional

from bs4 import BeautifulSoup

from hot_theme_rotator.data.external.realtime_price.http_policy import HttpFetchPolicy
from hot_theme_rotator.data.external.realtime_price.schema import (
    PriceQuote,
    PriceQuoteValidationError,
)


KABUTAN_BASE_URL = "https://kabutan.jp/stock/"
_PRICE_RE = re.compile(r"([\d,]+(?:\.\d+)?)")


class KabutanParseError(ValueError):
    """Raised when Kabutan HTML cannot be parsed into a valid quote."""


HttpGetText = Callable[[str, Mapping[str, str]], str]


def kabutan_url(symbol: str) -> str:
    if not symbol.endswith(".T"):
        raise KabutanParseError(f"symbol must end with '.T', got {symbol!r}")
    code = symbol[:-2]
    return f"{KABUTAN_BASE_URL}?code={code}"


def default_http_get_text(url: str, headers: Mapping[str, str]) -> str:
    import requests

    response = requests.get(url, headers=dict(headers), timeout=10)
    response.raise_for_status()
    return response.text


def fetch_kabutan_quote(
    symbol: str,
    *,
    http_get_text: HttpGetText = default_http_get_text,
    policy: Optional[HttpFetchPolicy] = None,
    wall_ts: Optional[str] = None,
) -> PriceQuote:
    active_policy = policy or HttpFetchPolicy()
    request = active_policy.prepare_request(kabutan_url(symbol))
    html = http_get_text(request.url, request.headers)
    active_policy.validate_response_text(html)
    return parse_kabutan_html(html, symbol=symbol, wall_ts=wall_ts)


def parse_kabutan_html(
    html: str,
    *,
    symbol: str,
    wall_ts: Optional[str] = None,
) -> PriceQuote:
    if not isinstance(html, str) or not html.strip():
        raise KabutanParseError("html must be a non-empty string")

    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    candidates.extend(soup.find_all("span", attrs={"class": "kabuka"}))
    candidates.extend(
        soup.find_all(
            "div", attrs={"class": re.compile(r"stock_kabuka|stock_kabuka_dt")}
        )
    )

    price = None
    for candidate in candidates:
        text = candidate.get_text(strip=True)
        match = _PRICE_RE.search(text)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            price = value
            break

    if price is None:
        raise KabutanParseError(
            f"could not extract price for {symbol} from Kabutan HTML"
        )

    now_ts = wall_ts or datetime.now(timezone.utc).isoformat()
    # Kabutan does not reliably expose a source-side timestamp.
    # Per Codex review 2026-05-25: flag inferred so Rule 12.2 stale check applies.
    try:
        return PriceQuote(
            symbol=symbol,
            price=price,
            source="kabutan",
            data_ts=now_ts,
            wall_ts=now_ts,
            data_ts_inferred=True,
        )
    except PriceQuoteValidationError as exc:
        raise KabutanParseError(
            f"PriceQuote construction failed: {exc}"
        ) from exc
