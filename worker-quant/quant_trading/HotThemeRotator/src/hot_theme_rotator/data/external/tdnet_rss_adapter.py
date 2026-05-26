"""TDnet RSS network adapter (P10-14 Cycle 2).

Yanoshin Web API as default upstream source. Rate-limited HTTP client with
retry-after and exponential backoff on 429/503 responses. Parses fetched JSON
through `tdnet_parser.parse_yanoshin_json` to produce `TdnetDisclosure` records.

Network access goes through an injected `http_get` callable for testability:
production default is `requests.get`; tests inject a stub returning
`HttpResponse` dataclass instances. The `sleep` and `monotonic` clocks are
likewise injectable so tests do not block on real time.

Per Rule 9.2 (within-session refresh) the polling cadence at the CLI layer is
15 minutes. The adapter itself enforces a per-request rate limit (default 5s)
to respect Yanoshin's free-tier guidance and our own anti-FOMO Rule 12.2 stale
fail-closed contract.

Research-only. No broker / order / paper-trade path (Rule 3).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping, Optional, Protocol

import requests

from hot_theme_rotator.data.external.tdnet_parser import (
    TdnetParseError,
    parse_yanoshin_json,
)
from hot_theme_rotator.data.external.tdnet_schema import TdnetDisclosure


DEFAULT_YANOSHIN_BASE_URL = "https://webapi.yanoshin.jp/webapi/tdnet/list"
DEFAULT_USER_AGENT = "HotThemeRotator/1.0 (personal research)"


class TdnetFetchError(RuntimeError):
    """Raised when TDnet RSS fetch fails after all retries."""


@dataclass
class HttpResponse:
    """Minimal HTTP response shape for adapter testability."""

    status_code: int
    text: str
    headers: Mapping[str, str]

    def json(self) -> Any:
        return json.loads(self.text)


class HttpGetProtocol(Protocol):
    def __call__(
        self, url: str, *, headers: Mapping[str, str], timeout: float
    ) -> HttpResponse: ...


def default_http_get(
    url: str, *, headers: Mapping[str, str], timeout: float
) -> HttpResponse:
    """Production HTTP GET using `requests.get`."""
    resp = requests.get(url, headers=dict(headers), timeout=timeout)
    return HttpResponse(
        status_code=resp.status_code,
        text=resp.text,
        headers=dict(resp.headers),
    )


@dataclass
class YanoshinTdnetAdapter:
    """Yanoshin Web API client for TDnet 適時開示.

    URL: f"{base_url}/{YYYYMMDD}.json?limit={limit}".
    Returns parsed TdnetDisclosure tuple.
    Fails closed via TdnetFetchError on network, status, JSON, or parse errors.
    """

    base_url: str = DEFAULT_YANOSHIN_BASE_URL
    rate_limit_seconds: float = 5.0
    max_retries: int = 3
    timeout_seconds: float = 30.0
    user_agent: str = DEFAULT_USER_AGENT
    http_get: HttpGetProtocol = field(default=default_http_get)
    sleep: Callable[[float], None] = field(default=time.sleep)
    monotonic: Callable[[], float] = field(default=time.monotonic)

    def __post_init__(self) -> None:
        self._last_request_ts: Optional[float] = None

    def fetch_list_for_date(
        self,
        trade_date: str,
        limit: int = 100,
    ) -> tuple[TdnetDisclosure, ...]:
        """Fetch disclosures for one trade date.

        `trade_date` must be ISO YYYY-MM-DD; URL uses compact YYYYMMDD.
        """
        try:
            date.fromisoformat(trade_date)
        except ValueError as exc:
            raise TdnetFetchError(
                f"trade_date must be ISO date (YYYY-MM-DD): {trade_date!r}"
            ) from exc

        if not (1 <= limit <= 1000):
            raise TdnetFetchError(f"limit must be in [1, 1000], got {limit}")

        compact = trade_date.replace("-", "")
        url = f"{self.base_url}/{compact}.json?limit={limit}"

        self._wait_for_rate_limit()
        collected_ts = datetime.now(timezone.utc).isoformat()

        last_status: Optional[int] = None
        last_text: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.http_get(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "application/json",
                    },
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                if attempt < self.max_retries:
                    self.sleep(self._backoff(attempt))
                    continue
                raise TdnetFetchError(
                    f"GET {url} network failure after {self.max_retries + 1} attempts: {exc}"
                ) from exc

            last_status = resp.status_code
            last_text = resp.text

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise TdnetFetchError(
                        f"GET {url} returned non-JSON body: {exc}"
                    ) from exc
                try:
                    return parse_yanoshin_json(payload, collected_ts=collected_ts)
                except TdnetParseError as exc:
                    raise TdnetFetchError(
                        f"parser rejected payload from {url}: {exc}"
                    ) from exc

            if resp.status_code in (429, 503):
                if attempt < self.max_retries:
                    retry_after = self._get_retry_after(
                        resp.headers, default=self._backoff(attempt)
                    )
                    self.sleep(retry_after)
                    continue
                raise TdnetFetchError(
                    f"GET {url} returned {resp.status_code} after {self.max_retries + 1} attempts"
                )

            raise TdnetFetchError(
                f"GET {url} returned non-retryable status {resp.status_code}: "
                f"{resp.text[:200]}"
            )

        raise TdnetFetchError(
            f"GET {url} exhausted retries (last status {last_status}, text {last_text[:200]})"
        )

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_ts is None:
            self._last_request_ts = self.monotonic()
            return
        elapsed = self.monotonic() - self._last_request_ts
        wait = self.rate_limit_seconds - elapsed
        if wait > 0:
            self.sleep(wait)
        self._last_request_ts = self.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, ..."""
        return float(2 ** attempt)

    @staticmethod
    def _get_retry_after(headers: Mapping[str, str], *, default: float) -> float:
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
