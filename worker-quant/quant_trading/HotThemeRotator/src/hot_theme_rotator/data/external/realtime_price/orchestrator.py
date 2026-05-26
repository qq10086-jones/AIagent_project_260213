"""Best-Effort Delayed Price Orchestrator (P10-19 Cycle 1).

Per ADR-0007: NOT real-time. Fallback chain across multiple sources with 60s
caching and conditional consensus for high-salience triggers.

Cycle 1: caller injects source fetchers as callables for testability — no
real HTTP. Cycle 2 wires real clients with rate limit + robots.txt + UA
rotation + Cloudflare detection.

Fail-closed per Rule 12.2: when all sources fail, raises PriceOrchestratorError.
Conditional consensus per Codex review: high-salience callers (chase boundary,
stop/exit thresholds, intraday move >= 5%) trigger a second-source lookup and
flag `price_uncertain=True` if delta exceeds threshold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from hot_theme_rotator.data.external.realtime_price.schema import PriceQuote


SourceFetcher = Callable[[str], PriceQuote]


class PriceOrchestratorError(RuntimeError):
    """Raised when no source in the chain can provide a valid quote."""


@dataclass
class PriceOrchestrator:
    """Fallback chain across multiple price sources with optional consensus.

    `source_chain` is an ordered list of `(name, fetcher)` pairs. The
    orchestrator tries each in order until one returns a `PriceQuote` without
    raising; results are cached per `(source_name, symbol)` with TTL.

    `high_salience=True` triggers a second-source lookup against
    `consensus_pair_chain` (defaults to `source_chain[1:]`). If a second
    source's price differs from the primary by more than
    `consensus_threshold_pct`, the returned quote carries
    `price_uncertain=True` and a `fail_reason` describing the mismatch.
    """

    source_chain: List[Tuple[str, SourceFetcher]]
    consensus_pair_chain: Optional[List[Tuple[str, SourceFetcher]]] = None
    cache_ttl_seconds: float = 60.0
    consensus_threshold_pct: float = 1.0
    monotonic: Callable[[], float] = field(default=time.monotonic)

    def __post_init__(self):
        if not self.source_chain:
            raise ValueError("source_chain cannot be empty")
        self._cache: dict[Tuple[str, str], Tuple[float, PriceQuote]] = {}
        if self.consensus_pair_chain is None:
            self.consensus_pair_chain = list(self.source_chain[1:])

    def get_quote(
        self,
        symbol: str,
        *,
        high_salience: bool = False,
    ) -> PriceQuote:
        cached_quote = self._cache_lookup(symbol)
        if cached_quote is not None and not high_salience:
            return cached_quote

        primary_quote, last_error = self._walk_chain(symbol)
        if primary_quote is None:
            raise PriceOrchestratorError(
                f"all sources failed for {symbol}; last error: {last_error}"
            )

        if high_salience:
            # Always run consensus check when high_salience requested, even if
            # consensus_pair_chain is empty — `_consensus_check` handles that
            # case by marking the quote uncertain.
            return self._consensus_check(primary_quote, symbol)

        return primary_quote

    def _cache_lookup(self, symbol: str) -> Optional[PriceQuote]:
        now = self.monotonic()
        for (cached_source, cached_symbol), (
            cached_at,
            cached_quote,
        ) in self._cache.items():
            if cached_symbol != symbol:
                continue
            if now - cached_at < self.cache_ttl_seconds:
                return cached_quote
        return None

    def _walk_chain(
        self,
        symbol: str,
    ) -> Tuple[Optional[PriceQuote], Optional[Exception]]:
        last_error: Optional[Exception] = None
        for source_name, fetcher in self.source_chain:
            try:
                quote = fetcher(symbol)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            self._cache[(source_name, symbol)] = (self.monotonic(), quote)
            return quote, None
        return None, last_error

    def _consensus_check(
        self,
        primary: PriceQuote,
        symbol: str,
    ) -> PriceQuote:
        """Conditional consensus for high-salience lookups.

        Per Codex review 2026-05-25: if no consensus pair source is reachable,
        return primary marked `price_uncertain=True` with explicit fail_reason.
        Returning primary unflagged would be overconfident when the caller
        explicitly requested a second opinion (e.g., Rule 12.3 chase boundary).
        """
        consensus_attempted = False
        for source_name, fetcher in self.consensus_pair_chain:
            if source_name == primary.source:
                continue
            try:
                secondary = fetcher(symbol)
            except Exception:  # noqa: BLE001
                continue
            if secondary.price <= 0 or primary.price <= 0:
                continue
            consensus_attempted = True
            delta_pct = (
                abs(secondary.price - primary.price) / primary.price * 100.0
            )
            if delta_pct > self.consensus_threshold_pct:
                return PriceQuote(
                    symbol=primary.symbol,
                    price=primary.price,
                    source=primary.source,
                    data_ts=primary.data_ts,
                    wall_ts=primary.wall_ts,
                    data_ts_inferred=primary.data_ts_inferred,
                    fail_reason=(
                        f"consensus mismatch vs {secondary.source}: "
                        f"{delta_pct:.2f}% delta"
                    ),
                    price_uncertain=True,
                )
            return primary
        # Loop ended without successfully comparing against any secondary source.
        if not consensus_attempted:
            return PriceQuote(
                symbol=primary.symbol,
                price=primary.price,
                source=primary.source,
                data_ts=primary.data_ts,
                wall_ts=primary.wall_ts,
                data_ts_inferred=primary.data_ts_inferred,
                fail_reason="consensus unavailable: no valid secondary source",
                price_uncertain=True,
            )
        return primary
