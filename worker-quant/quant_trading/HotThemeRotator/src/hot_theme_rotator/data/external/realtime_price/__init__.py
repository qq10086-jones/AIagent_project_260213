"""Best-Effort Delayed Price Orchestrator (P10-19 Cycle 1, ADR-0007).

Per Codex review: NOT "real-time". Physical floor ≈ 5 minutes (web display
refresh interval for free sources).

ACTIVE sources (post 2026-05-25 live smoke + no-credentials constraint):
- Yahoo Finance Japan: HTML scrape, ~5 min delay (data_ts_inferred=True)
- Kabutan.jp: HTML scrape, ~5 min delay (data_ts_inferred=True)
- yfinance: 15 min delay (existing baseline fallback)

INACTIVE sources (parsers retained for optionality but NOT in default chain):
- TwelveData: REST API; user has no account, deferred indefinitely
- Stooq: CSV download; service policy changed 2026-05-25 to require apikey + captcha
- J-Quants (via jquants_live_bridge): user has no account, deferred indefinitely

Effective default source_chain for callers: `[yahoo_japan, kabutan, yfinance]`.

Conditional consensus: high-salience triggers (chase boundary, stop/exit
thresholds, intraday move >= 5%) ask a second source for confirmation. Codex
2026-05-25 caveat: with only Yahoo JP + Kabutan as scrapers, consensus is
primarily a parser/selector failure detector, NOT independent feed verification
(both likely reflect the same upstream Japan Exchange display surface).

Rule 3 advice-only / Rule 12.2 stale fail-closed both inherit.
"""

from .schema import (
    ALLOWED_PRICE_SOURCES,
    PriceQuote,
    PriceQuoteValidationError,
)
from .health import (
    PriceSourceHealth,
    price_health_report_path,
    read_price_health_report,
    run_price_source_health_checks,
    write_price_health_report,
)
from .http_policy import (
    CloudflareBlockError,
    FixedRobotsPolicy,
    HttpFetchPolicy,
    HttpPolicyError,
    PreparedHttpRequest,
    RobotsBlockedError,
)

__all__ = [
    "ALLOWED_PRICE_SOURCES",
    "CloudflareBlockError",
    "FixedRobotsPolicy",
    "HttpFetchPolicy",
    "HttpPolicyError",
    "PreparedHttpRequest",
    "PriceQuote",
    "PriceSourceHealth",
    "PriceQuoteValidationError",
    "RobotsBlockedError",
    "price_health_report_path",
    "read_price_health_report",
    "run_price_source_health_checks",
    "write_price_health_report",
]
