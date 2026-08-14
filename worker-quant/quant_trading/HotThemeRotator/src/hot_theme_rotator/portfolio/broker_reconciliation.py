"""P37-04 — external reconciliation against the broker, as a separate authority.

The distinction this module enforces
------------------------------------
Rule 14.0 called the portfolio journal "the sole source of truth for live
positions, cash, and realized P&L", with no qualifier. Two different claims got
folded into one sentence:

- **internal accounting authority** — the journal decides what the books say,
  and re-deriving from it is deterministic. This part is true and unchanged.
- **external correctness** — that the books match the account at the broker.
  The journal cannot establish this about itself. Nothing can, except the
  broker.

On 2026-08-14 that gap became concrete. A cash event read off a broker table
that had lost its structure (`譲渡益税還付金 991`) entered canonical cash; the
derived balance became JPY 287,794 against the broker's JPY 287,068; and the
system published an official NAV of JPY 394,724 while its own scorecard said
`reconciled_no_contradicting_evidence`. Every component behaved as designed.
The design never required outside evidence, and "no contradiction" was computed
from the same journal it was supposed to check.

So `reconciled` is defined here as a claim about the outside world that only
outside evidence can support:

1. The broker snapshot must satisfy its OWN identity first --
   ``cash + Σ position_value == total_assets``. A page that does not add up is
   not evidence of anything, and comparing against it would launder a bad read
   into a verdict.
2. Only then are journal cash, per-symbol quantities and same-mark-time NAV
   compared.
3. Anything short of agreement is a diagnostic, never a NAV. A mismatch must
   not write official NAV history, must not compute return metrics, must not be
   called reconciled, and must never propose the entry that would close the
   gap. **The difference is a question, not an instruction** — a system that
   invents the missing JPY 726 to make the books balance has stopped keeping
   books.
4. Cash whose provenance is a screenshot, an OCR pass or reconstructed table
   text is PROVISIONAL until an account-level balance check passes. Provisional
   money is reported, never spent.

Rule 3 / Rule 4: read-only. Nothing here places an order or edits config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping

__all__ = [
    "MISSING_BROKER_SNAPSHOT",
    "STALE",
    "INCOMPARABLE_MARK_TIME",
    "MISMATCH",
    "RECONCILED",
    "STATES_ALLOWING_OFFICIAL_NAV",
    "UNSTRUCTURED_PROVENANCE",
    "BrokerPosition",
    "BrokerSnapshot",
    "BrokerSnapshotError",
    "JournalView",
    "ReconciliationVerdict",
    "canonical_cash",
    "check_broker_identity",
    "reconcile_against_broker",
]

MISSING_BROKER_SNAPSHOT = "missing_broker_snapshot"
STALE = "stale"
INCOMPARABLE_MARK_TIME = "incomparable_mark_time"
MISMATCH = "mismatch"
RECONCILED = "reconciled"

# Deliberately a one-element set. Every other state means the external claim was
# not established, and an unestablished claim must not unlock account metrics.
STATES_ALLOWING_OFFICIAL_NAV = frozenset({RECONCILED})

# Provenances that cannot be trusted to have preserved structure. A number read
# out of a broken table is a reading of a picture of a number.
UNSTRUCTURED_PROVENANCE = frozenset({"screenshot", "ocr", "broken_table_text", "manual_typed"})

# A snapshot whose `source` says it came from our own books is not evidence.
_SELF_DERIVED_SOURCES = frozenset({"derived_from_journal", "journal", "self", "derived"})

_MONEY_TOLERANCE = 0.5  # JPY; broker pages round to the yen


class BrokerSnapshotError(ValueError):
    """The snapshot is internally inconsistent and cannot serve as evidence."""


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    mark: float
    value: float

    def validate(self) -> "BrokerPosition":
        expected = self.qty * self.mark
        if abs(expected - self.value) > _MONEY_TOLERANCE:
            raise BrokerSnapshotError(
                f"{self.symbol}: qty x mark = {expected:,.2f} but the value column "
                f"says {self.value:,.2f}. A value that does not follow from the "
                "quantity and the mark usually means two rows were read as one."
            )
        return self


@dataclass(frozen=True)
class BrokerSnapshot:
    """What the broker's own page says. Typed in or parsed, never computed here."""

    asof: date
    cash: float
    positions: Mapping[str, BrokerPosition]
    total_assets: float
    source: str
    mark_time: str | None = None

    @property
    def positions_value(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def quantities(self) -> dict[str, float]:
        return {sym: p.qty for sym, p in self.positions.items()}

    @property
    def is_self_derived(self) -> bool:
        return self.source.strip().lower() in _SELF_DERIVED_SOURCES


@dataclass(frozen=True)
class JournalView:
    """The books, as derived from the append-only journal."""

    asof: date
    cash: float
    quantities: Mapping[str, float]
    mark_time: str | None = None


@dataclass
class ReconciliationVerdict:
    state: str
    reason: str | None
    differences: list[dict] = field(default_factory=list)
    broker_identity: dict = field(default_factory=dict)
    comparable_fields: list[str] = field(default_factory=list)

    @property
    def may_write_official_nav(self) -> bool:
        return self.state in STATES_ALLOWING_OFFICIAL_NAV

    @property
    def may_compute_return_metrics(self) -> bool:
        return self.state in STATES_ALLOWING_OFFICIAL_NAV

    @property
    def difference_is_not_an_instruction(self) -> bool:
        """Always true, and named so the intent survives a future refactor.

        There is no code path here that turns a difference into a journal entry.
        A balancing adjustment would make every future reconciliation pass by
        construction, which is the opposite of what reconciliation is for.
        """
        return True

    def to_dict(self) -> dict:
        return {
            "_kind": "broker_reconciliation",
            "state": self.state,
            "reason": self.reason,
            "differences": self.differences,
            "broker_identity": self.broker_identity,
            "comparable_fields": self.comparable_fields,
            "may_write_official_nav": self.may_write_official_nav,
            "may_compute_return_metrics": self.may_compute_return_metrics,
            "note": (
                "A difference is a question for the broker record, never an "
                "instruction to adjust the journal. No correction is proposed "
                "here by design."
            ),
        }


def check_broker_identity(snapshot: BrokerSnapshot) -> dict:
    """``cash + Σ position_value == total_assets``, checked before anything else."""
    positions_value = snapshot.positions_value
    residual = snapshot.total_assets - (snapshot.cash + positions_value)
    return {
        "cash": snapshot.cash,
        "positions_value": positions_value,
        "total_assets": snapshot.total_assets,
        "residual": round(residual, 6) if abs(residual) > 1e-9 else 0.0,
        "holds": abs(residual) <= _MONEY_TOLERANCE,
        "tolerance": _MONEY_TOLERANCE,
    }


def _mark_times_comparable(a: str | None, b: str | None) -> bool:
    """Two NAVs are comparable only if struck at the same moment."""
    if a is None or b is None:
        return False
    return a.strip() == b.strip()


def reconcile_against_broker(
    view: JournalView,
    snapshot: BrokerSnapshot | None,
    *,
    max_staleness_days: int = 1,
) -> ReconciliationVerdict:
    """Compare the books against the broker and return what may be claimed."""
    if snapshot is None:
        return ReconciliationVerdict(
            state=MISSING_BROKER_SNAPSHOT,
            reason=(
                "no broker snapshot supplied; the journal cannot establish its own "
                "agreement with the account"
            ),
        )

    if snapshot.is_self_derived:
        return ReconciliationVerdict(
            state=MISMATCH,
            reason=(
                f"snapshot source {snapshot.source!r} is derived from our own books; "
                "evidence generated by the subject cannot test the subject"
            ),
        )

    identity = check_broker_identity(snapshot)
    if not identity["holds"]:
        return ReconciliationVerdict(
            state=MISMATCH,
            reason=(
                "broker identity does not hold: cash + positions != total assets "
                f"(residual {identity['residual']:,.2f}); the snapshot is not "
                "evidence of anything and no comparison was attempted"
            ),
            broker_identity=identity,
        )

    staleness = (view.asof - snapshot.asof).days
    if staleness > max_staleness_days:
        return ReconciliationVerdict(
            state=STALE,
            reason=(
                f"broker snapshot is {staleness} day(s) older than the view "
                f"(limit {max_staleness_days})"
            ),
            broker_identity=identity,
        )

    differences: list[dict] = []
    if abs(view.cash - snapshot.cash) > _MONEY_TOLERANCE:
        differences.append(
            {
                "field": "cash",
                "journal": view.cash,
                "broker": snapshot.cash,
                "difference": round(view.cash - snapshot.cash, 6),
            }
        )

    broker_qty = snapshot.quantities
    for symbol in sorted(set(view.quantities) | set(broker_qty)):
        ours = float(view.quantities.get(symbol, 0))
        theirs = float(broker_qty.get(symbol, 0))
        if abs(ours - theirs) > 1e-9:
            differences.append(
                {
                    "field": f"quantity:{symbol}",
                    "journal": ours,
                    "broker": theirs,
                    "difference": round(ours - theirs, 9),
                }
            )

    comparable = ["cash", "quantity"]
    if not _mark_times_comparable(view.mark_time, snapshot.mark_time):
        return ReconciliationVerdict(
            state=INCOMPARABLE_MARK_TIME,
            reason=(
                f"journal marks at {view.mark_time!r}, broker at "
                f"{snapshot.mark_time!r}; a NAV struck at a different moment "
                "proves nothing about agreement"
            ),
            differences=differences,
            broker_identity=identity,
            comparable_fields=comparable,
        )
    comparable = comparable + ["nav"]

    if differences:
        return ReconciliationVerdict(
            state=MISMATCH,
            reason="journal and broker disagree; see differences",
            differences=differences,
            broker_identity=identity,
            comparable_fields=comparable,
        )

    return ReconciliationVerdict(
        state=RECONCILED,
        reason=None,
        differences=[],
        broker_identity=identity,
        comparable_fields=comparable,
    )


def canonical_cash(
    cash_events: Iterable[Mapping[str, Any]],
    *,
    opening_cash: float,
    broker_cash: float | None = None,
) -> dict:
    """Split cash events into canonical and provisional, and promote only on proof.

    A provisional event is one whose provenance cannot be trusted to have kept
    its structure - a screenshot, an OCR pass, text scraped out of a table that
    had already broken. Such a figure is reported but never spent, because the
    2026-08-14 case is exactly what happens when it is: `991` entered canonical
    cash from a broken table and the books moved away from the account.

    Promotion requires an account-level balance check: applying the provisional
    events must land on the broker's own cash figure. Nothing else promotes.
    """
    canonical_total = float(opening_cash)
    provisional: list[dict] = []
    provisional_total = 0.0

    for event in cash_events:
        amount = float(event.get("amount", 0.0))
        provenance = str(event.get("provenance", "")).strip().lower()
        flagged = bool(event.get("provisional", False))
        if flagged or provenance in UNSTRUCTURED_PROVENANCE:
            provisional.append(dict(event))
            provisional_total += amount
        else:
            canonical_total += amount

    balance_check: dict[str, Any] = {"performed": broker_cash is not None}
    promoted: list[str] = []
    if broker_cash is not None:
        with_provisional = canonical_total + provisional_total
        agrees = abs(with_provisional - float(broker_cash)) <= _MONEY_TOLERANCE
        balance_check.update(
            {
                "broker_cash": float(broker_cash),
                "canonical_only": canonical_total,
                "with_provisional": with_provisional,
                "agrees": agrees,
            }
        )
        if agrees and provisional:
            canonical_total = with_provisional
            promoted = [str(e.get("entry_id")) for e in provisional]
            provisional_total = 0.0
            provisional = []

    return {
        "canonical_cash": canonical_total,
        "provisional_total": provisional_total,
        "provisional_events": provisional,
        "promoted": promoted,
        "balance_check": balance_check,
    }
