"""Portfolio Ledger schema — Rule 14.1 / 14.2 / 14.3 / 14.4 invariants.

Two append-only entry types live in the journal: ``FillEntry`` for trade events
and ``CashEvent`` for non-trade cash flow (dividends, deposits, withdrawals,
etc.).

Both carry a deterministic ``entry_id`` derived from their material fields, so
duplicate writes can be rejected (Rule 14.1) and the journal is replay-safe.
``source`` is a mandatory enum (Rule 14.3) and ``correction`` entries MUST
reference the prior entry they correct (Rule 14.4).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Any, Final, Mapping, Optional


ALLOWED_SOURCES: Final[frozenset[str]] = frozenset(
    {"manual", "import", "paper", "migration", "correction"}
)
ALLOWED_SIDES: Final[frozenset[str]] = frozenset({"BUY", "SELL"})
ALLOWED_CASH_REASONS: Final[frozenset[str]] = frozenset(
    {"deposit", "withdrawal", "dividend", "interest", "fee_non_trade", "fx_adjustment", "other"}
)


class PortfolioSchemaError(ValueError):
    """Raised when a journal entry fails schema-level validation."""


# ──────────────────────────────────────────────────────────────────────────
# entry_id derivation (Rule 14.1 determinism)
# ──────────────────────────────────────────────────────────────────────────


def derive_fill_entry_id(
    *,
    ts: str,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    source: str,
    note: str = "",
) -> str:
    """Return a 16-hex deterministic id over the material fields of a fill.

    The ``|`` delimiter is safe because the contributing fields are constrained:
    ``ts`` is ISO 8601, ``symbol`` ends with ``.T``, ``side`` is BUY/SELL,
    ``source`` is enum, ``qty`` is int, ``price`` is float — none of which can
    smuggle a delimiter ambiguously.
    """
    payload = f"{ts}|{symbol}|{side}|{qty}|{price}|{source}|{note}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def derive_cash_event_id(
    *,
    ts: str,
    reason: str,
    amount: float,
    source: str,
    note: str = "",
    symbol: Optional[str] = None,
) -> str:
    """Return a 16-hex deterministic id over a cash event."""
    payload = f"{ts}|{reason}|{amount}|{source}|{note}|{symbol or ''}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────
# shared validators
# ──────────────────────────────────────────────────────────────────────────


def _validate_iso_tz_ts(ts: Any) -> None:
    if not isinstance(ts, str):
        raise PortfolioSchemaError(f"ts must be a string, got {type(ts).__name__}")
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise PortfolioSchemaError(f"ts must be ISO 8601, got {ts!r}") from exc
    if parsed.tzinfo is None:
        raise PortfolioSchemaError(
            f"ts must carry an explicit timezone (PIT discipline), got naive: {ts!r}"
        )


def _validate_jp_symbol(symbol: Any) -> None:
    if not isinstance(symbol, str) or not symbol.endswith(".T"):
        raise PortfolioSchemaError(f"symbol must end with '.T' (Japan equity), got {symbol!r}")
    prefix = symbol[:-2]
    if len(prefix) < 1 or not prefix.replace("_", "").isalnum():
        raise PortfolioSchemaError(f"symbol prefix invalid: {symbol!r}")


def _validate_source(source: Any) -> None:
    if source not in ALLOWED_SOURCES:
        raise PortfolioSchemaError(
            f"source must be one of {sorted(ALLOWED_SOURCES)}, got {source!r}"
        )


# ──────────────────────────────────────────────────────────────────────────
# FillEntry
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FillEntry:
    """A trade event in the portfolio journal.

    Rule 14.2 manual-entry gates (symbol whitelist, side enum, qty int, price
    positive, ts<=now JST, SELL-holdings-sufficient, BUY-cash check) are
    enforced at the **manual entry service layer** in a later cycle. This
    schema only enforces invariants that are universal regardless of source:
    field shape, enum membership, entry_id integrity, and Rule 14.4 correction
    coupling.
    """

    entry_id: str
    ts: str
    symbol: str
    side: str
    qty: int
    price: float
    source: str
    fee: float = 0.0
    note: str = ""
    corrects: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_iso_tz_ts(self.ts)
        _validate_jp_symbol(self.symbol)
        if self.side not in ALLOWED_SIDES:
            raise PortfolioSchemaError(
                f"side must be one of {sorted(ALLOWED_SIDES)}, got {self.side!r}"
            )
        if not isinstance(self.qty, int) or isinstance(self.qty, bool):
            raise PortfolioSchemaError(f"qty must be int (not bool/float), got {self.qty!r}")
        if self.qty <= 0:
            raise PortfolioSchemaError(f"qty must be positive, got {self.qty!r}")
        if not isinstance(self.price, (int, float)) or isinstance(self.price, bool):
            raise PortfolioSchemaError(f"price must be a number, got {self.price!r}")
        if self.price <= 0:
            raise PortfolioSchemaError(f"price must be positive, got {self.price!r}")
        if not isinstance(self.fee, (int, float)) or isinstance(self.fee, bool):
            raise PortfolioSchemaError(f"fee must be a number, got {self.fee!r}")
        if self.fee < 0:
            raise PortfolioSchemaError(f"fee must be non-negative, got {self.fee!r}")
        _validate_source(self.source)

        if self.source == "correction" and not self.corrects:
            raise PortfolioSchemaError(
                "source='correction' requires corrects=<entry_id> (Rule 14.4)"
            )
        if self.source != "correction" and self.corrects:
            raise PortfolioSchemaError(
                f"corrects field is only valid when source='correction', got source={self.source!r}"
            )

        expected = derive_fill_entry_id(
            ts=self.ts,
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            price=self.price,
            source=self.source,
            note=self.note,
        )
        if self.entry_id != expected:
            raise PortfolioSchemaError(
                f"entry_id mismatch (integrity check): got {self.entry_id!r}, expected {expected!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FillEntry":
        return cls(
            entry_id=payload["entry_id"],
            ts=payload["ts"],
            symbol=payload["symbol"],
            side=payload["side"],
            qty=payload["qty"],
            price=payload["price"],
            source=payload["source"],
            fee=payload.get("fee", 0.0),
            note=payload.get("note", ""),
            corrects=payload.get("corrects"),
        )


# ──────────────────────────────────────────────────────────────────────────
# CashEvent
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CashEvent:
    """A non-trade cash flow event (Rule 14.7).

    ``amount`` is signed JPY: positive = inflow (deposit, dividend, interest,
    fx_adjustment crediting cash), negative = outflow (withdrawal,
    fee_non_trade, fx_adjustment debiting cash). Zero amount is rejected.

    ``symbol`` is optional but recommended for events tied to a held
    instrument (e.g., 1306.T dividend).
    """

    entry_id: str
    ts: str
    amount: float
    reason: str
    source: str
    note: str = ""
    symbol: Optional[str] = None
    corrects: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_iso_tz_ts(self.ts)
        if self.reason not in ALLOWED_CASH_REASONS:
            raise PortfolioSchemaError(
                f"reason must be one of {sorted(ALLOWED_CASH_REASONS)}, got {self.reason!r}"
            )
        if not isinstance(self.amount, (int, float)) or isinstance(self.amount, bool):
            raise PortfolioSchemaError(f"amount must be a number, got {self.amount!r}")
        if self.amount == 0:
            raise PortfolioSchemaError("amount must be non-zero (zero events are noise)")
        _validate_source(self.source)
        if self.symbol is not None:
            _validate_jp_symbol(self.symbol)

        if self.source == "correction" and not self.corrects:
            raise PortfolioSchemaError(
                "source='correction' requires corrects=<entry_id> (Rule 14.4)"
            )
        if self.source != "correction" and self.corrects:
            raise PortfolioSchemaError(
                f"corrects field is only valid when source='correction', got source={self.source!r}"
            )

        expected = derive_cash_event_id(
            ts=self.ts,
            reason=self.reason,
            amount=self.amount,
            source=self.source,
            note=self.note,
            symbol=self.symbol,
        )
        if self.entry_id != expected:
            raise PortfolioSchemaError(
                f"entry_id mismatch (integrity check): got {self.entry_id!r}, expected {expected!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CashEvent":
        return cls(
            entry_id=payload["entry_id"],
            ts=payload["ts"],
            amount=payload["amount"],
            reason=payload["reason"],
            source=payload["source"],
            note=payload.get("note", ""),
            symbol=payload.get("symbol"),
            corrects=payload.get("corrects"),
        )
