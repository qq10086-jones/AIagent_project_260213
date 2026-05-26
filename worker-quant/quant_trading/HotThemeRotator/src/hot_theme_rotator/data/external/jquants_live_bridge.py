"""J-Quants Live Bridge (P10-16).

Reuses Project_optimized's J-Quants credentials (env-based) per ADR-0005 read-only
consume. Fetches daily_quotes via J-Quants v1 REST API. Returns PriceBar tuple
compatible with `hot_theme_rotator.common.schema.PriceBar` so existing kline
consumers can swap source via `?source=jquants_live`.

J-Quants v1 auth flow:
  1. POST /token/auth_user {mailaddress, password} -> {"refreshToken": "..."}
     (refresh token TTL 7 days)
  2. POST /token/auth_refresh?refreshtoken=... -> {"idToken": "..."}
     (id token TTL 24 hours)
  3. GET /prices/daily_quotes?code=NNNN0&from=YYYY-MM-DD&to=YYYY-MM-DD
     with Authorization: Bearer {idToken}

Cycle 1: http_post + http_get injected for testability; in-memory id_token cache.
Cycle 2 will add disk-backed token cache + integration test with mock HTTP fixtures.

Rule 3 advice-only / Rule 12.2 stale fail-closed; no broker / order path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping, Optional

import requests

from hot_theme_rotator.common.schema import PriceBar, SchemaValidationError
from hot_theme_rotator.data.external.tdnet_rss_adapter import HttpResponse


JQUANTS_BASE_URL = "https://api.jquants.com/v1"
JQUANTS_EMAIL_ENV = "JQUANTS_EMAIL"
JQUANTS_PASSWORD_ENV = "JQUANTS_PASSWORD"
JQUANTS_REFRESH_TOKEN_ENV = "JQUANTS_REFRESH_TOKEN"


class JquantsAuthError(RuntimeError):
    """Raised when authentication fails or credentials are missing."""


class JquantsFetchError(RuntimeError):
    """Raised when daily_quotes fetch fails."""


def default_http_post(
    url: str,
    *,
    json_payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> HttpResponse:
    resp = requests.post(
        url, json=dict(json_payload), headers=dict(headers), timeout=timeout
    )
    return HttpResponse(
        status_code=resp.status_code,
        text=resp.text,
        headers=dict(resp.headers),
    )


def default_http_get(
    url: str,
    *,
    headers: Mapping[str, str],
    timeout: float,
) -> HttpResponse:
    resp = requests.get(url, headers=dict(headers), timeout=timeout)
    return HttpResponse(
        status_code=resp.status_code,
        text=resp.text,
        headers=dict(resp.headers),
    )


def _normalize_jquants_code(symbol: str) -> str:
    """Convert HTR '6779.T' to J-Quants 5-digit code '67790'.

    JP equities use 4-digit ticker; J-Quants codes are 5-digit (4-digit + trailing 0).
    """
    if not isinstance(symbol, str) or not symbol.endswith(".T"):
        raise JquantsFetchError(f"symbol must end with '.T', got {symbol!r}")
    head = symbol[:-2]
    if len(head) == 4 and head.isdigit():
        return f"{head}0"
    if len(head) == 5 and head.isdigit():
        return head
    raise JquantsFetchError(f"symbol head must be 4 or 5 digits, got {symbol!r}")


@dataclass
class JquantsCredentials:
    """J-Quants credentials, typically loaded from env per ADR-0005."""

    email: Optional[str] = None
    password: Optional[str] = None
    refresh_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "JquantsCredentials":
        return cls(
            email=os.environ.get(JQUANTS_EMAIL_ENV),
            password=os.environ.get(JQUANTS_PASSWORD_ENV),
            refresh_token=os.environ.get(JQUANTS_REFRESH_TOKEN_ENV),
        )


@dataclass
class JquantsLiveBridge:
    """J-Quants v1 client returning PriceBar tuples for HTR kline consumers."""

    credentials: JquantsCredentials = field(default_factory=JquantsCredentials.from_env)
    base_url: str = JQUANTS_BASE_URL
    timeout_seconds: float = 30.0
    http_post: Callable[..., HttpResponse] = field(default=default_http_post)
    http_get: Callable[..., HttpResponse] = field(default=default_http_get)

    def __post_init__(self):
        self._cached_id_token: Optional[str] = None
        self._cached_refresh_token: Optional[str] = self.credentials.refresh_token

    def _get_refresh_token(self) -> str:
        if self._cached_refresh_token:
            return self._cached_refresh_token
        if not self.credentials.email or not self.credentials.password:
            raise JquantsAuthError(
                f"missing credentials: provide either {JQUANTS_REFRESH_TOKEN_ENV} "
                f"or both {JQUANTS_EMAIL_ENV} + {JQUANTS_PASSWORD_ENV}"
            )
        url = f"{self.base_url}/token/auth_user"
        resp = self.http_post(
            url,
            json_payload={
                "mailaddress": self.credentials.email,
                "password": self.credentials.password,
            },
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            raise JquantsAuthError(
                f"auth_user failed: {resp.status_code} {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise JquantsAuthError(f"auth_user returned non-JSON: {exc}") from exc
        token = data.get("refreshToken")
        if not token:
            raise JquantsAuthError(
                f"auth_user response missing refreshToken: {data}"
            )
        self._cached_refresh_token = token
        return token

    def _get_id_token(self) -> str:
        if self._cached_id_token:
            return self._cached_id_token
        refresh_token = self._get_refresh_token()
        url = f"{self.base_url}/token/auth_refresh?refreshtoken={refresh_token}"
        resp = self.http_post(
            url,
            json_payload={},
            headers={"Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            raise JquantsAuthError(
                f"auth_refresh failed: {resp.status_code} {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise JquantsAuthError(
                f"auth_refresh returned non-JSON: {exc}"
            ) from exc
        token = data.get("idToken")
        if not token:
            raise JquantsAuthError(
                f"auth_refresh response missing idToken: {data}"
            )
        self._cached_id_token = token
        return token

    def fetch_daily_quotes(
        self,
        symbol: str,
        *,
        date_from: str,
        date_to: Optional[str] = None,
    ) -> tuple[PriceBar, ...]:
        """Fetch daily OHLC bars for one symbol over an ISO date range.

        `date_to` defaults to None (J-Quants returns single day or open range).
        Returns PriceBar tuple sorted by asof ascending.
        Malformed individual quote items are silently skipped (consistent with
        other adapters that prefer partial recovery over total failure).
        """
        try:
            date.fromisoformat(date_from)
            if date_to is not None:
                date.fromisoformat(date_to)
        except ValueError as exc:
            raise JquantsFetchError(
                f"date_from/date_to must be ISO date YYYY-MM-DD: {exc}"
            ) from exc

        code = _normalize_jquants_code(symbol)
        params = f"code={code}&from={date_from}"
        if date_to is not None:
            params += f"&to={date_to}"
        url = f"{self.base_url}/prices/daily_quotes?{params}"

        id_token = self._get_id_token()
        resp = self.http_get(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=self.timeout_seconds,
        )
        if resp.status_code != 200:
            raise JquantsFetchError(
                f"daily_quotes failed: {resp.status_code} {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise JquantsFetchError(
                f"daily_quotes returned non-JSON: {exc}"
            ) from exc

        items = data.get("daily_quotes")
        if items is None:
            raise JquantsFetchError(
                f"daily_quotes response missing 'daily_quotes' key: {data}"
            )

        bars: list[PriceBar] = []
        for item in items:
            bar = self._parse_quote(item, symbol)
            if bar is not None:
                bars.append(bar)
        bars.sort(key=lambda b: b.asof)
        return tuple(bars)

    @staticmethod
    def _parse_quote(
        item: Mapping[str, Any], symbol: str
    ) -> Optional[PriceBar]:
        if not isinstance(item, Mapping):
            return None
        try:
            asof = str(item.get("Date", "")).strip()
            if not asof:
                return None
            date.fromisoformat(asof)
            payload = {
                "symbol": symbol,
                "asof": asof,
                "open": item.get("AdjustmentOpen") or item.get("Open"),
                "high": item.get("AdjustmentHigh") or item.get("High"),
                "low": item.get("AdjustmentLow") or item.get("Low"),
                "close": item.get("AdjustmentClose") or item.get("Close"),
                "volume": item.get("AdjustmentVolume") or item.get("Volume") or 0,
                "turnover_jpy": item.get("TurnoverValue") or 0,
            }
            if any(payload[k] is None for k in ("open", "high", "low", "close")):
                return None
            return PriceBar.from_dict(payload)
        except (KeyError, ValueError, TypeError, SchemaValidationError):
            return None
