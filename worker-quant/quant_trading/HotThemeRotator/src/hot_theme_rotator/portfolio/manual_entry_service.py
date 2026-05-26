"""P10-23 — manual entry service: validate + preview + append (Rule 14.5).

Single entry point for any manual-recording surface (CLI, REST endpoint,
driver-friendly UI modal). Wraps Rule 14.2 validation gates (validation.py) +
Rule 14.5 magnitude sanity check + Rule 14.1 append in one transactional shape.

Rule 3 stance: this module records trades the user has ALREADY executed via
their external broker. It never places orders. Endpoints/CLIs MUST surface this
in user-facing copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from hot_theme_rotator.portfolio.derive import (
    derive_cash_balance,
    derive_positions,
)
from hot_theme_rotator.portfolio.journal_writer import (
    PortfolioJournalError,
    append_cash_event,
    append_fill,
    journal_lock,
    read_all_journal,
)
from hot_theme_rotator.portfolio.schema import (
    CashEvent,
    FillEntry,
    derive_cash_event_id,
    derive_fill_entry_id,
)
from hot_theme_rotator.portfolio.validation import (
    ValidationResult,
    validate_manual_cash_event,
    validate_manual_fill,
)


JST = timezone(timedelta(hours=9), name="JST")
_MAGNITUDE_NAV_FRACTION = 0.10  # Rule 14.5: > 10% NAV requires 2nd confirm


__all__ = [
    "FillPreview",
    "CashEventPreview",
    "build_fill_entry",
    "build_cash_event",
    "preview_fill",
    "preview_cash_event",
    "commit_fill",
    "commit_cash_event",
]


@dataclass(frozen=True)
class FillPreview:
    fill: FillEntry
    validation: ValidationResult
    magnitude_warning: Optional[str]
    before: dict
    after: dict

    @property
    def all_warnings(self) -> tuple[str, ...]:
        ws = list(self.validation.warnings)
        if self.magnitude_warning:
            ws.append(self.magnitude_warning)
        return tuple(ws)


@dataclass(frozen=True)
class CashEventPreview:
    event: CashEvent
    validation: ValidationResult
    before: dict
    after: dict

    @property
    def all_warnings(self) -> tuple[str, ...]:
        return tuple(self.validation.warnings)


def build_fill_entry(
    *,
    side: str,
    symbol: str,
    qty: int,
    price: float,
    ts: str,
    source: str = "manual",
    fee: float = 0.0,
    note: str = "",
) -> FillEntry:
    eid = derive_fill_entry_id(
        ts=ts, symbol=symbol, side=side, qty=qty, price=price, source=source, note=note,
    )
    return FillEntry(
        entry_id=eid, ts=ts, symbol=symbol, side=side, qty=qty, price=price,
        source=source, fee=float(fee), note=note,
    )


def build_cash_event(
    *,
    ts: str,
    amount: float,
    reason: str,
    source: str = "manual",
    note: str = "",
    symbol: Optional[str] = None,
) -> CashEvent:
    eid = derive_cash_event_id(
        ts=ts, reason=reason, amount=float(amount), source=source, note=note, symbol=symbol,
    )
    return CashEvent(
        entry_id=eid, ts=ts, amount=float(amount), reason=reason, source=source,
        note=note, symbol=symbol,
    )


def preview_fill(
    fill: FillEntry,
    *,
    base_dir: str | Path = ".",
    current_prices: Optional[dict[str, float]] = None,
    now: Optional[datetime] = None,
) -> FillPreview:
    """Validate + compute before/after state for a proposed fill. No write."""
    now = now or datetime.now(tz=JST)
    journal = read_all_journal(base_dir)
    validation = validate_manual_fill(proposed=fill, journal=journal, now=now)

    before_positions = derive_positions(journal)
    before_cash = derive_cash_balance(journal)
    before_qty = before_positions[fill.symbol].qty if fill.symbol in before_positions else 0
    before_avg = (
        before_positions[fill.symbol].avg_cost if fill.symbol in before_positions else 0.0
    )

    after_positions = derive_positions(journal + (fill,))
    after_cash = derive_cash_balance(journal + (fill,))
    after_qty = after_positions[fill.symbol].qty if fill.symbol in after_positions else 0
    after_avg = (
        after_positions[fill.symbol].avg_cost if fill.symbol in after_positions else 0.0
    )
    after_realized = (
        after_positions[fill.symbol].realized_pnl if fill.symbol in after_positions else 0.0
    )
    before_realized = (
        before_positions[fill.symbol].realized_pnl if fill.symbol in before_positions else 0.0
    )
    realized_delta = after_realized - before_realized

    notional = fill.qty * fill.price + fill.fee
    magnitude_warning = _magnitude_check(
        notional=notional,
        before_cash=before_cash,
        before_positions=before_positions,
        current_prices=current_prices,
    )

    before = {
        "qty": before_qty,
        "avg_cost": before_avg,
        "cash": before_cash,
    }
    after = {
        "qty": after_qty,
        "avg_cost": after_avg,
        "cash": after_cash,
        "realized_pnl_delta": realized_delta,
    }
    return FillPreview(
        fill=fill,
        validation=validation,
        magnitude_warning=magnitude_warning,
        before=before,
        after=after,
    )


def preview_cash_event(
    event: CashEvent,
    *,
    base_dir: str | Path = ".",
    now: Optional[datetime] = None,
) -> CashEventPreview:
    now = now or datetime.now(tz=JST)
    journal = read_all_journal(base_dir)
    validation = validate_manual_cash_event(proposed=event, journal=journal, now=now)
    before_cash = derive_cash_balance(journal)
    after_cash = derive_cash_balance(journal + (event,))
    return CashEventPreview(
        event=event,
        validation=validation,
        before={"cash": before_cash},
        after={"cash": after_cash},
    )


def commit_fill(
    preview: FillPreview,
    *,
    base_dir: str | Path = ".",
    now: Optional[datetime] = None,
) -> Path:
    """Append the previewed fill to the journal under a serializing lock.

    Re-validates against the current journal state before appending (Rule 14.5
    clarification). Wraps the read-validate-append sequence in a
    ``journal_lock`` to close the TOCTOU window between concurrent commits
    (Rule 14.1 + Rule 14.5).

    Additionally rejects when the current journal already contains a
    correction whose ``corrects`` field targets ``preview.fill.entry_id`` —
    that would mean an earlier-appended correction will silently drop this
    commit via derive_positions' skip-both semantic.

    On rejection: raises ``PortfolioValidationError`` (hard-gate fail) or
    ``PortfolioJournalError`` (correction-targets-us) and does NOT append.
    Callers should re-run ``preview_fill`` and surface a fresh preview.
    """
    now = now or datetime.now(tz=JST)
    with journal_lock(base_dir):
        journal = read_all_journal(base_dir)
        validate_manual_fill(proposed=preview.fill, journal=journal, now=now)
        _reject_if_pre_corrected(preview.fill.entry_id, journal)
        return append_fill(preview.fill, base_dir=base_dir)


def commit_cash_event(
    preview: CashEventPreview,
    *,
    base_dir: str | Path = ".",
    now: Optional[datetime] = None,
) -> Path:
    """Append the previewed cash event with re-validation under a lock."""
    now = now or datetime.now(tz=JST)
    with journal_lock(base_dir):
        journal = read_all_journal(base_dir)
        validate_manual_cash_event(proposed=preview.event, journal=journal, now=now)
        _reject_if_pre_corrected(preview.event.entry_id, journal)
        return append_cash_event(preview.event, base_dir=base_dir)


# ─── internals ──────────────────────────────────────────────────────────────


def _reject_if_pre_corrected(
    proposed_entry_id: str,
    journal,
) -> None:
    """Reject when journal already contains a correction targeting this id.

    Closes the H5 attack: a correction can be crafted with ``corrects`` set to
    the deterministic ``entry_id`` of a *future* fill. If that fill later
    lands, derive_positions skip-both drops it silently. Rejecting at commit
    surface prevents the future fill from being recorded at all.
    """
    for entry in journal:
        if getattr(entry, "source", None) == "correction":
            if getattr(entry, "corrects", None) == proposed_entry_id:
                raise PortfolioJournalError(
                    f"refusing to append entry {proposed_entry_id}: an existing "
                    f"correction {entry.entry_id} already targets it (Rule 14.4 "
                    f"forward-targeting is forbidden); investigate the correction "
                    f"before retrying"
                )


# ─── internals ──────────────────────────────────────────────────────────────


def _magnitude_check(
    *,
    notional: float,
    before_cash: float,
    before_positions: dict,
    current_prices: Optional[dict[str, float]],
) -> Optional[str]:
    """Rule 14.5: warn when notional > 10% NAV. NAV needs current prices."""
    if current_prices is None:
        return None
    holdings_value = sum(
        pv.qty * current_prices.get(sym, 0.0) for sym, pv in before_positions.items()
    )
    nav = before_cash + holdings_value
    if nav <= 0:
        return None
    fraction = notional / nav
    if fraction > _MAGNITUDE_NAV_FRACTION:
        return (
            f"trade notional ¥{notional:,.0f} is {fraction*100:.1f}% of NAV "
            f"(¥{nav:,.0f}) — exceeds Rule 14.5 10% threshold, confirm twice"
        )
    return None
