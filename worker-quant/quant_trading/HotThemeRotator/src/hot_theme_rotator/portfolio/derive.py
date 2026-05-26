"""Derive ``positions`` and ``cash_balance`` views from the append-only journal.

Rule 14.1: these views are NEVER persisted — they are computed on demand from
the journal each time. Determinism is mandatory.

Correction handling (Rule 14.4): a ``source='correction'`` entry **invalidates**
the entry it references in the ``corrects`` field. Both the corrected entry
and the correction itself are skipped during derivation — they cancel out
mathematically. The actual replacement (the right values) is a separate fresh
journal entry. This keeps cost-basis math clean: an erroneously-recorded SELL
that reversed at the wrong avg_cost does not contaminate WAC for shares it
never actually touched.

Why not "correction is a reversal trade": treating a correction as a real
trade at the wrong-entry's price re-prices the un-traded shares (e.g., 900
shares at ¥403 get re-averaged to ¥409 by a 400-share correction BUY at
¥417.6). That's wrong — the shares never left the account; the cost basis must
remain ¥403. Skip-both is the only model that preserves cost-basis integrity.

Oversell protection: a SELL that exceeds current holdings raises
``PortfolioDeriveError`` fail-closed rather than producing a phantom short
position. Manual entry validation (P10-21 later cycle) catches this BEFORE the
entry lands in the journal; this is the second-line defense for journals built
via migration or import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hot_theme_rotator.portfolio.schema import CashEvent, FillEntry


__all__ = [
    "PortfolioDeriveError",
    "PositionView",
    "derive_cash_balance",
    "derive_positions",
]


class PortfolioDeriveError(RuntimeError):
    """Raised when derivation encounters an impossible state (e.g., oversell)."""


@dataclass(frozen=True)
class PositionView:
    """Per-symbol current state derived from the journal.

    - ``qty``: cumulative BUY − SELL after all entries (>= 0; oversells raise).
    - ``avg_cost``: weighted average cost basis of currently held shares, JPY.
    - ``realized_pnl``: cumulative realized P&L from all SELLs to date, JPY.
    """

    symbol: str
    qty: int
    avg_cost: float
    realized_pnl: float


def derive_positions(journal: Sequence) -> dict[str, PositionView]:
    """Compute current positions from append-only journal in append order."""
    skip_ids = _collect_skip_ids(journal)
    accumulators: dict[str, dict[str, float]] = {}

    for entry in journal:
        if entry.entry_id in skip_ids:
            continue
        if isinstance(entry, CashEvent):
            continue
        if not isinstance(entry, FillEntry):
            raise PortfolioDeriveError(
                f"unknown journal entry type {type(entry).__name__}"
            )

        acc = accumulators.setdefault(
            entry.symbol,
            {"qty": 0, "total_cost": 0.0, "realized_pnl": 0.0},
        )

        if entry.side == "BUY":
            acc["qty"] = int(acc["qty"]) + entry.qty
            acc["total_cost"] += entry.qty * entry.price + entry.fee
        else:  # SELL
            current_qty = int(acc["qty"])
            if entry.qty > current_qty:
                raise PortfolioDeriveError(
                    f"SELL of {entry.qty} {entry.symbol} exceeds holdings of "
                    f"{current_qty} (entry_id={entry.entry_id}, ts={entry.ts})"
                )
            avg_cost = acc["total_cost"] / current_qty if current_qty > 0 else 0.0
            acc["realized_pnl"] += entry.qty * (entry.price - avg_cost) - entry.fee
            acc["total_cost"] -= entry.qty * avg_cost
            acc["qty"] = current_qty - entry.qty

    result: dict[str, PositionView] = {}
    for symbol, acc in accumulators.items():
        qty = int(acc["qty"])
        if qty > 0:
            avg = acc["total_cost"] / qty
        else:
            avg = 0.0
        if qty == 0 and acc["realized_pnl"] == 0.0:
            # cleanly closed at no realized — drop from view
            continue
        result[symbol] = PositionView(
            symbol=symbol,
            qty=qty,
            avg_cost=avg,
            realized_pnl=acc["realized_pnl"],
        )
    return result


def derive_cash_balance(journal: Sequence) -> float:
    """Compute current cash balance (JPY) from append-only journal in order."""
    skip_ids = _collect_skip_ids(journal)
    cash = 0.0
    for entry in journal:
        if entry.entry_id in skip_ids:
            continue
        if isinstance(entry, CashEvent):
            cash += entry.amount  # signed: + inflow, − outflow
        elif isinstance(entry, FillEntry):
            if entry.side == "BUY":
                cash -= entry.qty * entry.price + entry.fee
            else:  # SELL
                cash += entry.qty * entry.price - entry.fee
        else:
            raise PortfolioDeriveError(
                f"unknown journal entry type {type(entry).__name__}"
            )
    return cash


def _collect_skip_ids(journal: Sequence) -> set[str]:
    """Return entry_ids that must be skipped during derivation (Rule 14.4).

    Includes:
    - every ``source='correction'`` entry (the correction itself is bookkeeping).
    - every entry_id referenced by a correction's ``corrects`` field
      (the corrected entry never happened, mathematically).

    A correction that references a non-existent entry_id has no effect on
    other entries; it is itself still skipped.
    """
    skip: set[str] = set()
    for entry in journal:
        if entry.source == "correction":
            skip.add(entry.entry_id)
            if entry.corrects:
                skip.add(entry.corrects)
    return skip
