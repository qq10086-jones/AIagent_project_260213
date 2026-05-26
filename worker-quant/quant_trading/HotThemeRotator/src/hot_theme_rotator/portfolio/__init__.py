"""Portfolio Ledger package (P10-21, Section 14, ADR-0008).

Append-only event journal for HotThemeRotator-owned portfolio state. Binding
on Cutover Day T (recommended 2026-06-08). Until T, ADR-0005 read-only
consumption of Project_optimized stays in force.

Surface so far:

- ``schema``: ``FillEntry`` / ``CashEvent`` dataclasses + deterministic
  entry-id derivation + Rule 14.3 / 14.4 invariants.
- ``journal_writer``: append-only JSONL writer + JST-partitioned files +
  duplicate-id rejection + malformed-line fail-closed (Rule 14.1).
- ``derive``: ``derive_positions`` + ``derive_cash_balance`` pure functions
  over the journal (Rule 14.1 views never persisted, oversell fail-closed).
- ``validation``: Rule 14.2 fail-closed manual-entry gates (future-ts reject,
  oversell reject, source whitelist, BUY/withdrawal cash warnings).

P10-21 implementation surface is now complete. P10-22 (migration snapshot) and
P10-23 (manual entry UI / CLI) consume this package on cutover day T.
"""
from __future__ import annotations

from hot_theme_rotator.portfolio.derive import (
    PortfolioDeriveError,
    PositionView,
    derive_cash_balance,
    derive_positions,
)
from hot_theme_rotator.portfolio.journal_writer import (
    JST,
    JournalEntry,
    PortfolioJournalError,
    append_cash_event,
    append_fill,
    journal_path,
    read_journal,
)
from hot_theme_rotator.portfolio.schema import (
    ALLOWED_CASH_REASONS,
    ALLOWED_SIDES,
    ALLOWED_SOURCES,
    CashEvent,
    FillEntry,
    PortfolioSchemaError,
    derive_cash_event_id,
    derive_fill_entry_id,
)
from hot_theme_rotator.portfolio.validation import (
    PortfolioValidationError,
    ValidationResult,
    validate_manual_cash_event,
    validate_manual_fill,
)

__all__ = [
    "ALLOWED_CASH_REASONS",
    "ALLOWED_SIDES",
    "ALLOWED_SOURCES",
    "JST",
    "CashEvent",
    "FillEntry",
    "JournalEntry",
    "PortfolioDeriveError",
    "PortfolioJournalError",
    "PortfolioSchemaError",
    "PortfolioValidationError",
    "PositionView",
    "ValidationResult",
    "append_cash_event",
    "append_fill",
    "derive_cash_balance",
    "derive_cash_event_id",
    "derive_fill_entry_id",
    "derive_positions",
    "journal_path",
    "read_journal",
    "validate_manual_cash_event",
    "validate_manual_fill",
]
