"""PriceQuote schema for the Best-Effort Delayed Price Orchestrator.

Per ADR-0007: this is NOT real-time. Fields are intentionally explicit about
source provenance and freshness so Rule 12.2 stale fail-closed can be enforced
by callers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional


ALLOWED_PRICE_SOURCES = frozenset(
    {
        "yahoo_japan",
        "kabutan",
        "twelvedata",
        "stooq",
        "yfinance",
        "cache",
    }
)


class PriceQuoteValidationError(ValueError):
    """Raised when a PriceQuote fails fail-closed validation."""


@dataclass(frozen=True)
class PriceQuote:
    """A best-effort delayed price quote from one source.

    Fields:
      symbol: normalized 'NNNN.T' form.
      price: positive float (JPY for JP equities).
      source: one of ALLOWED_PRICE_SOURCES.
      data_ts: ISO timestamp claimed for this price. For HTML scrape sources
               that do not expose a source-side timestamp, this falls back to
               wall_ts and `data_ts_inferred` is True. Callers enforcing Rule
               12.2 stale fail-closed MUST treat inferred-ts with skepticism.
      wall_ts: ISO timestamp when the adapter actually fetched it.
      data_ts_inferred: True when data_ts was substituted from wall_ts because
                        the source did not expose its own timestamp. Per Codex
                        review 2026-05-25: scrapers that silently set
                        data_ts=wall_ts can turn stale HTML into a "fresh"
                        quote — this flag makes that uncertainty visible.
      fail_reason: optional human-readable reason if this quote is a fallback
                   or carries a caveat (e.g., consensus mismatch, source
                   timestamp unreadable, consensus unavailable).
      price_uncertain: True when orchestrator's conditional consensus check
                       found a second source disagreed beyond threshold OR
                       when consensus check was unavailable for a high-salience
                       lookup.
    """

    symbol: str
    price: float
    source: str
    data_ts: str
    wall_ts: str
    data_ts_inferred: bool = False
    fail_reason: Optional[str] = None
    price_uncertain: bool = False

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol.endswith(".T"):
            raise PriceQuoteValidationError(
                f"symbol must end with '.T', got {self.symbol!r}"
            )
        head = self.symbol[:-2]
        if len(head) != 4 or not head.isdigit():
            raise PriceQuoteValidationError(
                f"symbol head must be 4 digits, got {self.symbol!r}"
            )
        if not isinstance(self.price, (int, float)) or self.price <= 0:
            raise PriceQuoteValidationError(
                f"price must be positive number, got {self.price!r}"
            )
        if self.source not in ALLOWED_PRICE_SOURCES:
            raise PriceQuoteValidationError(
                f"source must be one of {sorted(ALLOWED_PRICE_SOURCES)}, got {self.source!r}"
            )
        for ts_field in ("data_ts", "wall_ts"):
            ts_value = getattr(self, ts_field)
            try:
                datetime.fromisoformat(ts_value)
            except (TypeError, ValueError) as exc:
                raise PriceQuoteValidationError(
                    f"{ts_field} must be ISO 8601, got {ts_value!r}"
                ) from exc

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriceQuote":
        return cls(
            symbol=data["symbol"],
            price=data["price"],
            source=data["source"],
            data_ts=data["data_ts"],
            wall_ts=data["wall_ts"],
            data_ts_inferred=data.get("data_ts_inferred", False),
            fail_reason=data.get("fail_reason"),
            price_uncertain=data.get("price_uncertain", False),
        )
