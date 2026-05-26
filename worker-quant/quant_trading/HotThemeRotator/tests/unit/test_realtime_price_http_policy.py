"""Tests for realtime price HTTP access policy (P10-19 Cycle 2)."""
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.external.realtime_price.http_policy import (  # noqa: E402
    CloudflareBlockError,
    FixedRobotsPolicy,
    HttpFetchPolicy,
    RobotsBlockedError,
)


def test_policy_applies_rate_limit_between_same_host_requests():
    sleeps: list[float] = []
    ticks = iter([100.0, 102.0, 102.0])
    policy = HttpFetchPolicy(
        min_interval_seconds=10.0,
        monotonic=lambda: next(ticks),
        sleep=sleeps.append,
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA1",),
    )

    policy.prepare_request("https://finance.yahoo.co.jp/quote/6779.T")
    policy.prepare_request("https://finance.yahoo.co.jp/quote/1306.T")

    assert sleeps == [8.0]


def test_policy_does_not_rate_limit_different_hosts():
    sleeps: list[float] = []
    ticks = iter([100.0, 102.0])
    policy = HttpFetchPolicy(
        min_interval_seconds=10.0,
        monotonic=lambda: next(ticks),
        sleep=sleeps.append,
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA1",),
    )

    policy.prepare_request("https://finance.yahoo.co.jp/quote/6779.T")
    policy.prepare_request("https://kabutan.jp/stock/?code=6779")

    assert sleeps == []


def test_policy_rotates_user_agents():
    policy = HttpFetchPolicy(
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA1", "UA2"),
    )

    first = policy.prepare_request("https://finance.yahoo.co.jp/quote/6779.T")
    second = policy.prepare_request("https://kabutan.jp/stock/?code=6779")
    third = policy.prepare_request("https://example.com/")

    assert first.headers["User-Agent"] == "UA1"
    assert second.headers["User-Agent"] == "UA2"
    assert third.headers["User-Agent"] == "UA1"


def test_policy_blocks_disallowed_robots_url():
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=False))

    with pytest.raises(RobotsBlockedError, match="robots.txt blocked"):
        policy.prepare_request("https://example.com/private")


def test_policy_detects_cloudflare_html():
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=True))

    with pytest.raises(CloudflareBlockError, match="Cloudflare"):
        policy.validate_response_text(
            "<html><title>Just a moment...</title>"
            "<script>window._cf_chl_opt={}</script></html>"
        )


def test_policy_accepts_normal_html():
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=True))

    policy.validate_response_text("<html><body><span>3015</span></body></html>")
