"""P37-04 — labelling NAV records with the reconciliation state they were born in.

An NAV that has not been checked against the broker is a computation, not a
measurement of an account. Both look identical once written to a trace, and the
trace is what later analyses read: a bare `nav_jpy` from an unreconciled day is
indistinguishable from a settled one, which is how JPY 394,724 sat in
`reports/observability/risk_mandate/2026-08-14.json` next to a broker account
holding JPY 393,998.

This module does one narrow thing: it stamps a record with the reconciliation
verdict under which it was produced, and refuses to call anything official that
was not externally confirmed.

Two things it deliberately does NOT do:

- **It never edits the figure it labels.** The wrong number stays visible;
  deleting evidence is not a correction, and Rule 14.1's append-only spirit
  applies to derived artifacts as much as to the journal.
- **It never proposes the entry that would close a gap.** See
  ``broker_reconciliation`` - a difference is a question for the broker record,
  not an instruction.
"""
from __future__ import annotations

from typing import Any, Mapping

from .broker_reconciliation import ReconciliationVerdict

__all__ = [
    "NAV_STATUS_KEY",
    "UNRECONCILED_NAV_NOTE",
    "annotate_nav_record",
    "may_publish_official_nav",
    "supersede_record",
]

NAV_STATUS_KEY = "nav_reconciliation"

UNRECONCILED_NAV_NOTE = (
    "not externally reconciled: this NAV is derived from the journal and has "
    "not been confirmed against a broker account snapshot"
)

_RECONCILED_NOTE = (
    "externally reconciled: cash, share counts and a same-mark-time total agree "
    "with an independent broker account snapshot"
)


def may_publish_official_nav(verdict: ReconciliationVerdict) -> bool:
    """Official NAV history and account return metrics require agreement."""
    return verdict.may_write_official_nav


def annotate_nav_record(
    record: dict[str, Any], verdict: ReconciliationVerdict
) -> dict[str, Any]:
    """Stamp a NAV record with its reconciliation state, leaving the figures alone."""
    official = may_publish_official_nav(verdict)
    record[NAV_STATUS_KEY] = {
        "state": verdict.state,
        "official": official,
        "metrics_allowed": verdict.may_compute_return_metrics,
        "reason": verdict.reason,
        "differences": verdict.differences,
        "broker_identity": verdict.broker_identity,
        "comparable_fields": verdict.comparable_fields,
        "note": _RECONCILED_NOTE if official else UNRECONCILED_NAV_NOTE,
    }
    return record


def supersede_record(record: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    """Mark a published record superseded without removing or altering it."""
    out = dict(record)
    out["superseded"] = True
    out["superseded_reason"] = reason
    status = dict(out.get(NAV_STATUS_KEY) or {})
    status["official"] = False
    status["metrics_allowed"] = False
    status.setdefault("state", "mismatch")
    status.setdefault("note", UNRECONCILED_NAV_NOTE)
    out[NAV_STATUS_KEY] = status
    return out
