"""Advice decision queue (P29) — make advice→owner-action delay observable.

The 2026-08-04 retrospective found the execution loop BROKEN for a reason that
had nothing to do with signal quality: advice existed only as disk JSON behind
a dashboard nobody had to open, and it was not a first-class object. With no
ID, no state, and no age, "the owner decided not to act" and "the owner never
saw it" were the same thing to the system — so the delay could not be measured,
and what cannot be measured cannot be improved.

This module gives advice an identity and a lifecycle:

    open -> acknowledged -> executed | declined | expired | superseded

``acknowledged`` is an OPTIONAL observation, not a gate. The owner can act (or
decline) without the system ever having been told they saw it; forcing an
acknowledge-first path would make the ledger unable to record what actually
happened, which is the exact failure being fixed. Its value is that when it IS
recorded, trigger-to-seen and trigger-to-terminal become separable — the first
is a notification problem, the second is a decision problem, and they have
different fixes.

Storage is one append-only JSONL of TRANSITIONS (not states) at
``reports/observability/decision_queue.jsonl``. Current state is a fold over
the rows, so history is never rewritten (§14 append-only discipline). Ages are
counted in JPX sessions, never calendar days — calendar days overstate every
delay across a weekend, and the retrospective's original "9 sessions" figure
came from exactly that class of error.

Rule 3 is untouched: this records decisions, it never places or routes an
order, and it holds no broker identifier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from hot_theme_rotator.data.jpx_calendar import sessions_between

__all__ = [
    "ALLOWED_DECLINE_REASONS",
    "ALLOWED_SEVERITIES",
    "ALLOWED_STATES",
    "TERMINAL_STATES",
    "DecisionQueueError",
    "QueueItem",
    "advice_id",
    "default_queue_path",
    "load_queue",
    "open_item",
    "queue_report",
    "transition",
]


class DecisionQueueError(Exception):
    """Invalid queue operation. Raised, never silently swallowed."""


ALLOWED_STATES = frozenset({
    "open", "acknowledged", "executed", "declined", "expired", "superseded",
})
TERMINAL_STATES = frozenset({"executed", "declined", "expired", "superseded"})

# Structured decline reasons, same discipline as Rule 13.9's bounded rejection
# reasons: a decline is a recorded DECISION. Free text alone is not a reason —
# it cannot be counted, so it cannot show up in the execution scorecard.
ALLOWED_DECLINE_REASONS = frozenset({
    "user_disagrees",
    "already_actioned",
    "insufficient_capital",
    "execution_infeasible",
    "superseded_by_newer_advice",
    "out_of_scope",
    "deferred_by_choice",
})

ALLOWED_SEVERITIES = frozenset({"binding", "advisory", "informational"})

_ID_LEN = 16


def default_queue_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / "reports" / "observability" / "decision_queue.jsonl"


def advice_id(*, source_rule: str, subject: str, created_asof: str) -> str:
    """Deterministic identity for one piece of advice on one session.

    Scoped to (rule, subject, session) so a flag that is still open tomorrow
    keeps its identity and its age, while the same rule firing on a different
    session is a genuinely new item. This is what makes ``open_item``
    idempotent across afterclose reruns.
    """
    payload = f"{source_rule}\x1f{subject}\x1f{created_asof}"
    return sha256(payload.encode("utf-8")).hexdigest()[:_ID_LEN]


@dataclass
class QueueItem:
    advice_id: str
    source_rule: str
    subject: str
    summary: str
    created_asof: str
    severity: str
    evidence_ref: str | None = None
    state: str = "open"
    decline_reason: str | None = None
    note: str | None = None
    acknowledged_asof: str | None = None
    terminal_asof: str | None = None
    history: list[dict] = field(default_factory=list)

    def age_sessions(self, asof: date) -> int | None:
        """Elapsed JPX sessions since creation; ``None`` outside the calendar."""
        try:
            created = date.fromisoformat(self.created_asof)
        except ValueError:
            return None
        return sessions_between(created, asof)

    def _span(self, end_asof: str | None) -> int | None:
        if not end_asof:
            return None
        try:
            return self.age_sessions(date.fromisoformat(end_asof))
        except ValueError:
            return None

    @property
    def trigger_to_seen_sessions(self) -> int | None:
        """Sessions from trigger to the owner being recorded as having seen it.

        ``None`` means UNOBSERVED, never zero: an item that went straight to a
        terminal state was never recorded as seen, and reporting that as
        same-session visibility would hide the notification gap.
        """
        return self._span(self.acknowledged_asof)

    @property
    def trigger_to_terminal_sessions(self) -> int | None:
        return self._span(self.terminal_asof)


def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Parse transition rows; malformed lines warn rather than abort.

    One bad line must never take down the whole queue — an unreadable queue
    reads as "nothing pending", which is the most dangerous possible lie here.
    """
    if not Path(path).exists():
        return [], []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"queue_unreadable:{type(exc).__name__}"]
    rows: list[dict] = []
    warnings: list[str] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            warnings.append(f"malformed_queue_line:{number}")
    return rows, warnings


def _fold(rows: Iterable[Mapping[str, Any]]) -> dict[str, QueueItem]:
    items: dict[str, QueueItem] = {}
    for row in rows:
        aid = row.get("advice_id")
        if not aid:
            continue
        state = row.get("state")
        if aid not in items:
            if state != "open":
                # A transition with no opening row: keep it visible rather than
                # dropping it, but do not invent the missing metadata.
                items[aid] = QueueItem(
                    advice_id=aid,
                    source_rule=row.get("source_rule", "unknown"),
                    subject=row.get("subject", "unknown"),
                    summary=row.get("summary", ""),
                    created_asof=row.get("asof", ""),
                    severity=row.get("severity", "informational"),
                )
            else:
                items[aid] = QueueItem(
                    advice_id=aid,
                    source_rule=row.get("source_rule", "unknown"),
                    subject=row.get("subject", "unknown"),
                    summary=row.get("summary", ""),
                    created_asof=row.get("asof", ""),
                    severity=row.get("severity", "informational"),
                    evidence_ref=row.get("evidence_ref"),
                )
        item = items[aid]
        item.history.append(dict(row))
        if state and state != "open":
            item.state = state
            if state == "acknowledged" and item.acknowledged_asof is None:
                item.acknowledged_asof = row.get("asof")
            if state in TERMINAL_STATES:
                item.terminal_asof = row.get("asof")
            if row.get("reason"):
                item.decline_reason = row["reason"]
            if row.get("note"):
                item.note = row["note"]
    return items


def load_queue(path: Path | str) -> dict[str, QueueItem]:
    rows, _ = _read_rows(Path(path))
    return _fold(rows)


def _append(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def open_item(
    path: Path | str,
    *,
    source_rule: str,
    subject: str,
    summary: str,
    created_asof: str,
    severity: str = "advisory",
    evidence_ref: str | None = None,
) -> str:
    """Open one advice item. Idempotent on (rule, subject, session).

    Reruns of afterclose, and a flag that is simply still open, must not create
    duplicates — a queue that inflates its own open count would recreate the
    row-counting bug that Rule 17.4.7's age had.
    """
    if severity not in ALLOWED_SEVERITIES:
        raise DecisionQueueError(
            f"unknown severity {severity!r}; allowed: {sorted(ALLOWED_SEVERITIES)}")
    path = Path(path)
    aid = advice_id(source_rule=source_rule, subject=subject, created_asof=created_asof)
    if aid in load_queue(path):
        return aid
    _append(path, {
        "advice_id": aid,
        "state": "open",
        "asof": created_asof,
        "source_rule": source_rule,
        "subject": subject,
        "summary": summary,
        "severity": severity,
        "evidence_ref": evidence_ref,
    })
    return aid


def transition(
    path: Path | str,
    item_id: str,
    state: str,
    *,
    asof: str,
    reason: str | None = None,
    note: str | None = None,
) -> str:
    """Record a state transition. Append-only; terminal states are final."""
    if state not in ALLOWED_STATES or state == "open":
        raise DecisionQueueError(
            f"unknown state {state!r}; allowed: {sorted(ALLOWED_STATES - {'open'})}")
    path = Path(path)
    items = load_queue(path)
    item = items.get(item_id)
    if item is None:
        raise DecisionQueueError(f"unknown advice id {item_id!r}")
    if item.state in TERMINAL_STATES:
        raise DecisionQueueError(
            f"advice {item_id} is already terminal ({item.state}); "
            "append a new advice item instead of reopening history")
    if state == "declined" and reason not in ALLOWED_DECLINE_REASONS:
        raise DecisionQueueError(
            "declining requires a structured reason from "
            f"{sorted(ALLOWED_DECLINE_REASONS)} (an optional free-text note may "
            "accompany it, but never replace it)")
    if item.state == state:
        return state  # idempotent: re-recording the current state is a no-op
    _append(path, {
        "advice_id": item_id,
        "state": state,
        "asof": asof,
        "reason": reason,
        "note": note,
    })
    return state


def queue_report(path: Path | str, *, asof: date) -> dict:
    """Open-age distribution + terminal counts for the afterclose surface.

    Empty is reported honestly: ``oldest_open_sessions`` is ``None`` (N/A on a
    zero denominator), never 0, which would read as "everything is fresh".
    """
    rows, warnings = _read_rows(Path(path))
    items = _fold(rows)
    open_items = [i for i in items.values() if i.state not in TERMINAL_STATES]
    terminal_counts: dict[str, int] = {}
    for item in items.values():
        if item.state in TERMINAL_STATES:
            terminal_counts[item.state] = terminal_counts.get(item.state, 0) + 1

    open_rows = sorted(
        (
            {
                "advice_id": i.advice_id,
                "source_rule": i.source_rule,
                "subject": i.subject,
                "summary": i.summary,
                "severity": i.severity,
                "state": i.state,
                "created_asof": i.created_asof,
                "age_sessions": i.age_sessions(asof),
            }
            for i in open_items
        ),
        key=lambda r: (-(r["age_sessions"] or 0), r["subject"]),
    )
    ages = [r["age_sessions"] for r in open_rows if r["age_sessions"] is not None]
    seen = [i.trigger_to_seen_sessions for i in items.values()
            if i.trigger_to_seen_sessions is not None]
    term = [i.trigger_to_terminal_sessions for i in items.values()
            if i.trigger_to_terminal_sessions is not None]
    return {
        "available": True,
        "asof": asof.isoformat(),
        "open_count": len(open_items),
        "open_items": open_rows,
        "oldest_open_sessions": max(ages) if ages else None,
        "terminal_counts": terminal_counts,
        # None = never observed. Distinguishing this from 0 is the whole point:
        # zero would claim same-session visibility the system never had.
        "median_trigger_to_seen_sessions": _median(seen),
        "median_trigger_to_terminal_sessions": _median(term),
        "warnings": warnings,
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0
