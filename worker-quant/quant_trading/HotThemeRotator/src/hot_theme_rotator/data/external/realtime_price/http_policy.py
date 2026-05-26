"""HTTP access policy for best-effort delayed price sources.

This module keeps scraper network behavior explicit and testable: rate limiting,
User-Agent selection, robots checks, and anti-bot page detection happen before
HTML is passed to source parsers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import cycle
from typing import Callable, Mapping, Protocol
from urllib.parse import urlparse


DEFAULT_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)


class HttpPolicyError(RuntimeError):
    """Base class for HTTP policy failures."""


class RobotsBlockedError(HttpPolicyError):
    """Raised when robots policy disallows fetching a URL."""


class CloudflareBlockError(HttpPolicyError):
    """Raised when a response looks like a Cloudflare / anti-bot challenge."""


class RobotsPolicy(Protocol):
    def allowed(self, url: str, user_agent: str) -> bool:
        """Return whether `user_agent` may fetch `url`."""


class FixedRobotsPolicy:
    """Deterministic robots policy for tests and conservative callers."""

    def __init__(self, allowed: bool = True) -> None:
        self._allowed = bool(allowed)

    def allowed(self, url: str, user_agent: str) -> bool:
        return self._allowed


@dataclass(frozen=True)
class PreparedHttpRequest:
    """Policy-approved HTTP request metadata."""

    url: str
    headers: Mapping[str, str]


class HttpFetchPolicy:
    """Apply scraper HTTP policy before callers fetch source HTML."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = 10.0,
        user_agents: tuple[str, ...] = DEFAULT_USER_AGENTS,
        robots_policy: RobotsPolicy | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        if not user_agents:
            raise ValueError("user_agents cannot be empty")
        self.min_interval_seconds = float(min_interval_seconds)
        self.robots_policy = robots_policy or FixedRobotsPolicy(allowed=True)
        self.monotonic = monotonic
        self.sleep = sleep
        self._user_agents = cycle(user_agents)
        self._last_fetch_by_host: dict[str, float] = {}

    def prepare_request(self, url: str) -> PreparedHttpRequest:
        host = _host(url)
        user_agent = next(self._user_agents)
        if not self.robots_policy.allowed(url, user_agent):
            raise RobotsBlockedError(f"robots.txt blocked URL: {url}")

        now = self.monotonic()
        last_fetch = self._last_fetch_by_host.get(host)
        if last_fetch is not None:
            wait_seconds = self.min_interval_seconds - (now - last_fetch)
            if wait_seconds > 0:
                self.sleep(wait_seconds)
                now = self.monotonic()
        self._last_fetch_by_host[host] = now

        return PreparedHttpRequest(
            url=url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

    def validate_response_text(self, text: str) -> None:
        if _looks_like_cloudflare(text):
            raise CloudflareBlockError("Cloudflare or anti-bot challenge detected")


def _host(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"url must be absolute, got {url!r}")
    return parsed.netloc.lower()


def _looks_like_cloudflare(text: str) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text[:5000].lower()
    markers = (
        "window._cf_chl_opt",
        "cf-chl",
        "cf-browser-verification",
        "checking your browser",
        "just a moment...",
        "cloudflare",
    )
    return any(marker in lowered for marker in markers)
