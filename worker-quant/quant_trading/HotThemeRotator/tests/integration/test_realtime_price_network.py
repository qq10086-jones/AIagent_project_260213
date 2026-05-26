"""Mock HTTP integration tests for delayed realtime price sources."""
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
)
from hot_theme_rotator.data.external.realtime_price.kabutan_scraper import (  # noqa: E402
    fetch_kabutan_quote,
)
from hot_theme_rotator.data.external.realtime_price.yahoo_japan_scraper import (  # noqa: E402
    fetch_yahoo_japan_quote,
)


YAHOO_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "realtime_price" / "yahoo_japan_sample.html"
)
KABUTAN_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "realtime_price" / "kabutan_sample.html"
)


class RecordingHttpGet:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> str:
        self.calls.append((url, headers))
        return self.text


def test_fetch_yahoo_quote_uses_policy_and_parses_quote():
    http_get = RecordingHttpGet(YAHOO_FIXTURE.read_text(encoding="utf-8"))
    policy = HttpFetchPolicy(
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA-Y",),
    )

    quote = fetch_yahoo_japan_quote(
        "6779.T",
        http_get_text=http_get,
        policy=policy,
        wall_ts="2026-05-26T09:00:00+09:00",
    )

    assert quote.source == "yahoo_japan"
    assert quote.price == 3015.0
    assert quote.data_ts_inferred is True
    assert http_get.calls[0][1]["User-Agent"] == "UA-Y"


def test_fetch_kabutan_quote_uses_policy_and_parses_quote():
    http_get = RecordingHttpGet(KABUTAN_FIXTURE.read_text(encoding="utf-8"))
    policy = HttpFetchPolicy(
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA-K",),
    )

    quote = fetch_kabutan_quote(
        "6779.T",
        http_get_text=http_get,
        policy=policy,
        wall_ts="2026-05-26T09:00:00+09:00",
    )

    assert quote.source == "kabutan"
    assert quote.price == 3015.0
    assert quote.data_ts_inferred is True
    assert http_get.calls[0][1]["User-Agent"] == "UA-K"


def test_fetch_yahoo_quote_rejects_cloudflare_before_parser():
    http_get = RecordingHttpGet(
        "<html><title>Just a moment...</title>"
        "<script>window._cf_chl_opt={}</script></html>"
    )
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=True))

    with pytest.raises(CloudflareBlockError):
        fetch_yahoo_japan_quote(
            "6779.T",
            http_get_text=http_get,
            policy=policy,
            wall_ts="2026-05-26T09:00:00+09:00",
        )


def test_fetch_kabutan_quote_rejects_cloudflare_before_parser():
    http_get = RecordingHttpGet(
        "<html><title>Just a moment...</title>"
        "<script>window._cf_chl_opt={}</script></html>"
    )
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=True))

    with pytest.raises(CloudflareBlockError):
        fetch_kabutan_quote(
            "6779.T",
            http_get_text=http_get,
            policy=policy,
            wall_ts="2026-05-26T09:00:00+09:00",
        )
