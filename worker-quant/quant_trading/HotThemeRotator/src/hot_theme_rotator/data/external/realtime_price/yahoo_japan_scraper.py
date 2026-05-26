"""Yahoo Finance Japan scraper for delayed price quotes (P10-19 Cycle 1).

URL pattern: https://finance.yahoo.co.jp/quote/{ticker}
Latency: ~5 minutes.

Cycle 1: fixture-based parsing only. Cycle 2 will add rate limit + robots.txt
verification + User-Agent rotation + Cloudflare detection.
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


YAHOO_JP_BASE_URL = "https://finance.yahoo.co.jp/quote"
_PRICE_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")


class YahooJapanParseError(ValueError):
    """Raised when Yahoo Japan HTML cannot be parsed into a valid quote."""


HttpGetText = Callable[[str, Mapping[str, str]], str]


def yahoo_japan_url(symbol: str) -> str:
    if not symbol.endswith(".T"):
        raise YahooJapanParseError(f"symbol must end with '.T', got {symbol!r}")
    return f"{YAHOO_JP_BASE_URL}/{symbol}"


def default_http_get_text(url: str, headers: Mapping[str, str]) -> str:
    import requests

    response = requests.get(url, headers=dict(headers), timeout=10)
    response.raise_for_status()
    return response.text


def fetch_yahoo_japan_quote(
    symbol: str,
    *,
    http_get_text: HttpGetText = default_http_get_text,
    policy: Optional[HttpFetchPolicy] = None,
    wall_ts: Optional[str] = None,
) -> PriceQuote:
    active_policy = policy or HttpFetchPolicy()
    request = active_policy.prepare_request(yahoo_japan_url(symbol))
    html = http_get_text(request.url, request.headers)
    active_policy.validate_response_text(html)
    return parse_yahoo_japan_html(html, symbol=symbol, wall_ts=wall_ts)


def parse_yahoo_japan_html(
    html: str,
    *,
    symbol: str,
    wall_ts: Optional[str] = None,
) -> PriceQuote:
    """Extract latest price from Yahoo Japan HTML.

    Tolerates multiple class-name patterns since Yahoo Japan periodically
    redesigns its quote pages.
    """
    if not isinstance(html, str) or not html.strip():
        raise YahooJapanParseError("html must be a non-empty string")

    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    candidates.extend(
        soup.find_all(
            "span", attrs={"class": re.compile(r"StyledNumber__value")}
        )
    )
    candidates.extend(
        soup.find_all("span", attrs={"data-test-id": "stockPrice"})
    )
    candidates.extend(
        soup.find_all("span", attrs={"class": re.compile(r"^_3rXWJKZF")})
    )

    price = None
    for candidate in candidates:
        text = candidate.get_text(strip=True)
        match = _PRICE_NUMBER_RE.search(text)
        if not match:
            continue
        try:
            value = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        if value > 0:
            price = value
            break

    if price is None:
        raise YahooJapanParseError(
            f"could not extract price for {symbol} from Yahoo Japan HTML"
        )

    now_ts = wall_ts or datetime.now(timezone.utc).isoformat()
    # Yahoo Japan does not reliably expose a source-side timestamp in HTML.
    # Per Codex review 2026-05-25: silently using wall_ts as data_ts can turn
    # a stale cached page into a "fresh" quote. Flag explicitly so callers
    # (Rule 12.2 stale fail-closed) treat the timestamp with skepticism.
    try:
        return PriceQuote(
            symbol=symbol,
            price=price,
            source="yahoo_japan",
            data_ts=now_ts,
            wall_ts=now_ts,
            data_ts_inferred=True,
        )
    except PriceQuoteValidationError as exc:
        raise YahooJapanParseError(
            f"PriceQuote construction failed: {exc}"
        ) from exc
