"""Rule 14.2 manual-entry validation gates (fail-closed).

Schema gates (.T symbol / side enum / qty int / price positive / fee non-negative /
entry_id integrity / Rule 14.4 correction coupling) are already enforced by
``FillEntry.__post_init__`` and ``CashEvent.__post_init__``. This module adds
the semantic gates that need journal context.

Hard gates (raise ``PortfolioValidationError``):

- ``ts`` must be <= ``now`` (no future-dated entries, PIT discipline)
- ``source`` must be ``manual`` or ``import`` (other sources have separate paths)
- For SELL fills: derived holdings must be >= proposed qty (no short positions)

Soft warnings (returned in ``ValidationResult.warnings``, UI displays but does
not block):

- For BUY fills: derived cash − qty × price − fee < 0 → warn user to add a
  deposit cash_event or confirm overdraft
- For withdrawal cash_events: derived cash + amount < 0 (amount is negative) →
  same warning shape

Magnitude sanity check (Rule 14.5: qty × price > 10% NAV) is NOT this module's
responsibility — it requires current market prices which the journal does not
hold. The manual-entry service (P10-23) computes it before showing preview.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from hot_theme_rotator.portfolio.derive import (
    derive_cash_balance,
    derive_positions,
)
from hot_theme_rotator.portfolio.schema import CashEvent, FillEntry


__all__ = [
    "PortfolioValidationError",
    "ValidationResult",
    "validate_manual_cash_event",
    "validate_manual_fill",
]


_ALLOWED_MANUAL_SOURCES = frozenset({"manual", "import"})


class PortfolioValidationError(ValueError):
    """Raised when a proposed manual entry violates a Rule 14.2 hard gate."""


@dataclass(frozen=True)
class ValidationResult:
    """Result of running Rule 14.2 manual-entry gates.

    Hard-gate failures raise; this object only carries soft warnings. An empty
    ``warnings`` tuple means the entry is fully clean.
    """

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


def validate_manual_fill(
    *,
    proposed: FillEntry,
    journal: Sequence,
    now: datetime,
) -> ValidationResult:
    """Run all Rule 14.2 gates for a proposed manual ``FillEntry``."""
    _check_source_is_manual_or_import(proposed.source)
    _check_ts_not_in_future(proposed.ts, now)

    if proposed.side == "SELL":
        _check_sell_holdings_sufficient(proposed, journal)

    warnings: list[str] = []
    if proposed.side == "BUY":
        warnings.extend(_collect_buy_cash_warnings(proposed, journal))

    return ValidationResult(warnings=tuple(warnings))


def validate_manual_cash_event(
    *,
    proposed: CashEvent,
    journal: Sequence,
    now: datetime,
) -> ValidationResult:
    """Run all Rule 14.2 gates for a proposed manual ``CashEvent``."""
    _check_source_is_manual_or_import(proposed.source)
    _check_ts_not_in_future(proposed.ts, now)

    warnings: list[str] = []
    if proposed.amount < 0:
        warnings.extend(_collect_withdrawal_cash_warnings(proposed, journal))

    return ValidationResult(warnings=tuple(warnings))


# ─── gate implementations ───────────────────────────────────────────────────


def _check_source_is_manual_or_import(source: str) -> None:
    if source not in _ALLOWED_MANUAL_SOURCES:
        raise PortfolioValidationError(
            f"validate_manual_* accepts source in {sorted(_ALLOWED_MANUAL_SOURCES)}, "
            f"got {source!r}. Other sources (paper / migration / correction) have "
            f"their own code paths and skip manual-entry gates."
        )


def _check_ts_not_in_future(ts: str, now: datetime) -> None:
    if now.tzinfo is None:
        raise PortfolioValidationError(
            f"now must be timezone-aware for PIT comparison, got naive {now.isoformat()!r}"
        )
    # Normalize "Z" suffix (UTC) — accepted in Python 3.11+ but not earlier; the
    # rest of the project (alerts.discipline._parse_ts) already does this, so we
    # match that contract here for consistency.
    normalized = ts.replace("Z", "+00:00") if isinstance(ts, str) else ts
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise PortfolioValidationError(
            f"ts must be ISO 8601, got {ts!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise PortfolioValidationError(
            f"ts must carry timezone (PIT discipline), got naive {ts!r}"
        )
    if parsed > now:
        raise PortfolioValidationError(
            f"ts {ts!r} is in the future relative to now={now.isoformat()}; "
            f"PIT discipline (Rule 8.6) forbids future-dated entries"
        )


def _check_sell_holdings_sufficient(fill: FillEntry, journal: Sequence) -> None:
    positions = derive_positions(journal)
    held = positions.get(fill.symbol)
    held_qty = held.qty if held is not None else 0
    if fill.qty > held_qty:
        raise PortfolioValidationError(
            f"SELL of {fill.qty} {fill.symbol} exceeds current holdings of "
            f"{held_qty} (Rule 14.2: no short positions). Adjust qty or add a "
            f"BUY entry first."
        )


def _collect_buy_cash_warnings(fill: FillEntry, journal: Sequence) -> list[str]:
    cash = derive_cash_balance(journal)
    cost = fill.qty * fill.price + fill.fee
    remaining = cash - cost
    if remaining < 0:
        return [
            f"BUY exceeds available cash by ¥{-remaining:,.0f} "
            f"(current cash ¥{cash:,.0f}, this trade ¥{cost:,.0f}). "
            f"Add a deposit cash_event or confirm overdraft."
        ]
    return []


def _collect_withdrawal_cash_warnings(event: CashEvent, journal: Sequence) -> list[str]:
    cash = derive_cash_balance(journal)
    remaining = cash + event.amount  # event.amount is negative for withdrawal
    if remaining < 0:
        return [
            f"withdrawal exceeds available cash by ¥{-remaining:,.0f} "
            f"(current cash ¥{cash:,.0f}, this withdrawal ¥{-event.amount:,.0f}). "
            f"Confirm or reduce amount."
        ]
    return []
