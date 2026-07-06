"""External ADR watch schema (P20-02 / Rule 11.15).

A deterministic, dependency-free schema for EXTERNAL semiconductor catalyst
inputs — SK hynix ADR (`SKHY`), its Korea line (`000660.KS`), US memory/AI
peers (`MU`, `NVDA`), a SOX/semi ETF proxy, and `USDJPY=X`. These are catalyst
*context* for Japanese semi candidates; they are NEVER HTR candidate symbols,
JP portfolio/calibration samples, or enabled trade markets (Rule 11.15).

The schema deliberately carries NO probability / win-rate / expected-return /
edge field. Listing status is explicit so a not-yet-listed or stale SKHY feed
surfaces honestly and leaves JP ranking unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping, Optional

ALLOWED_ADR_STATUSES = frozenset({"pending_listing", "active", "stale", "unavailable"})

# Default freshness window (calendar days) — tolerates weekends/holidays between
# the external market's last bar and the JP asof date.
DEFAULT_MAX_AGE_DAYS = 4

__all__ = [
    "ALLOWED_ADR_STATUSES",
    "DEFAULT_MAX_AGE_DAYS",
    "AdrInstrumentSnapshot",
    "is_stale",
    "overnight_return",
]


def overnight_return(last_price: Optional[float], prev_close: Optional[float]) -> Optional[float]:
    """Overnight move ``last/prev - 1`` or None if either is missing / prev<=0."""
    if last_price is None or prev_close is None:
        return None
    try:
        prev = float(prev_close)
        if prev <= 0:
            return None
        return float(last_price) / prev - 1.0
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])  # accepts "YYYY-MM-DD" or ISO ts
    except ValueError:
        return None


def is_stale(asof: str, data_ts: Optional[str], *, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> bool:
    """True when the instrument's last data is missing or older than the window.

    Fail-closed: an unparseable/absent ``data_ts`` is treated as stale (Rule
    11.15 — never manufacture freshness).
    """
    if not data_ts:
        return True
    a = _parse_date(asof)
    d = _parse_date(data_ts)
    if a is None or d is None:
        return True
    delta_days = (a - d).days
    # Fail-closed: future-dated data (delta < 0) is suspect (clock skew / bad feed)
    # → treat as stale, never "fresh" (P20 fix#3).
    return delta_days < 0 or delta_days > max_age_days


@dataclass(frozen=True)
class AdrInstrumentSnapshot:
    """One external instrument's read-only snapshot (no edge fields)."""

    symbol: str
    role: str
    asof: str
    data_ts: Optional[str]
    status: str
    last_price: Optional[float]
    prev_close: Optional[float]
    overnight_return: Optional[float]
    volume: Optional[float]
    volume_z: Optional[float]
    currency: str
    source: str
    stale: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if self.status not in ALLOWED_ADR_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(ALLOWED_ADR_STATUSES)}, got {self.status!r}"
            )
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency must be a non-empty string")
        if not isinstance(self.reasons, tuple):
            raise ValueError("reasons must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdrInstrumentSnapshot":
        return cls(
            symbol=payload["symbol"],
            role=payload.get("role", ""),
            asof=payload["asof"],
            data_ts=payload.get("data_ts"),
            status=payload["status"],
            last_price=payload.get("last_price"),
            prev_close=payload.get("prev_close"),
            overnight_return=payload.get("overnight_return"),
            volume=payload.get("volume"),
            volume_z=payload.get("volume_z"),
            currency=payload.get("currency", "USD"),
            source=payload.get("source", ""),
            stale=bool(payload.get("stale", False)),
            reasons=tuple(payload.get("reasons", ()) or ()),
        )
