"""Low-noise state-transition notifications (P30, Section 12 / Rule 12.7).

The retrospective's execution loop broke partly because a flag could trigger
and the owner not learn of it for sessions. The obvious fix — push the
dashboard — is the wrong one: Section 12 already holds that the push channel
is itself a risk surface, and the alert-fatigue literature (Ancker et al.
2017, DOI 10.1186/s12911-017-0430-8) finds repeated alerts REDUCE acceptance.
A notifier that fires daily on an unchanged state trains the owner to dismiss
the one message that mattered.

So the gate is deliberately stingy:

- only genuine STATE TRANSITIONS notify; an item sitting open for seven
  sessions is one notification, not seven;
- `informational` severity never pushes at all;
- a per-(item, state) dedupe key makes repeats structurally impossible;
- a monthly budget is a hard stop, not an advisory;
- consecutive delivery failures roll the gate back to silent mode rather than
  retry-storming the owner.

Rule 12.7 is untouched and remains the ONLY way to enable a channel: this
class takes the already-resolved enabled set and defaults to none, so the
shipped state is silence. Payloads link a decision ID and carry no order
control, no broker identifier, and no account field (Rule 3).
"""
from __future__ import annotations

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable

from hot_theme_rotator.data.jpx_calendar import sessions_between
from hot_theme_rotator.decision_queue import load_queue

__all__ = [
    "MONTHLY_BUDGET",
    "MAX_CONSECUTIVE_FAILURES",
    "SUBJECT_COOLDOWN_SESSIONS",
    "NotificationGate",
    "dedupe_key",
]

# Deliberately small. Section 12's anti-FOMO discipline treats a high ceiling
# as the failure mode, not a feature.
MONTHLY_BUDGET = 20
MAX_CONSECUTIVE_FAILURES = 3
# Dedupe stops one item repeating; this stops one noisy SUBJECT dominating.
# A sleeve emitting a fresh binding item every couple of sessions would clear
# dedupe every time and still be a daily alarm.
SUBJECT_COOLDOWN_SESSIONS = 5

# Severities that may ever reach a channel. `informational` is excluded by
# design: if it were pushable, every state would eventually be pushed.
_PUSHABLE = {"binding", "advisory"}


def dedupe_key(advice_id: str, state: str) -> str:
    """Stable key for one (item, state) pair — the unit of "already told you"."""
    return sha256(f"{advice_id}\x1f{state}".encode("utf-8")).hexdigest()[:16]


class NotificationGate:
    """Decides what may be delivered, records everything either way.

    Silent mode still writes audit rows. If disabled runs left no trace, P30's
    own metrics would be blind exactly while the channel is off — which is the
    state the system ships in.
    """

    def __init__(
        self,
        *,
        base_dir: Path | str,
        sink: Callable[[dict], bool] | None = None,
        enabled_channels: Iterable[str] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.sink = sink
        self.enabled_channels = set(enabled_channels or ())
        self._audit_path = (
            self.base_dir / "reports" / "observability" / "notifications"
            / "transition_log.jsonl"
        )
        self._state_path = self._audit_path.parent / "gate_state.json"
        state = self._load_state()
        # Rollback and the failure streak MUST outlive the process: afterclose
        # is a fresh interpreter every session, so in-memory state would rearm
        # a broken channel tomorrow — the same retry storm, one day apart.
        self.rolled_back = bool(state.get("rolled_back"))
        self._failures = int(state.get("consecutive_failures") or 0)

    # --- persisted gate state -------------------------------------------

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Unreadable state fails SAFE (silent), not open: a corrupt file
            # must not be a licence to start pushing again.
            return {"rolled_back": True, "consecutive_failures": 0}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps({
                "rolled_back": self.rolled_back,
                "consecutive_failures": self._failures,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def reset_rollback(self, *, asof: str, note: str = "") -> None:
        """Re-arm after an operator has repaired the channel.

        Deliberately explicit: automatic re-arming is what turns a broken
        channel into a daily alarm the owner learns to ignore.
        """
        self.rolled_back = False
        self._failures = 0
        self._save_state()
        self._record({"asof": asof, "event": "rollback_reset", "note": note,
                      "delivered": False, "suppressed_reason": None})

    # --- audit -----------------------------------------------------------

    def audit_rows(self) -> list[dict]:
        if not self._audit_path.exists():
            return []
        rows = []
        for line in self._audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows

    def _record(self, row: dict) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _seen_keys(self) -> set[str]:
        return {r["dedupe_key"] for r in self.audit_rows() if r.get("dedupe_key")}

    def _sent_this_month(self, month: str) -> int:
        return sum(
            1 for r in self.audit_rows()
            if r.get("delivered") and str(r.get("asof", "")).startswith(month)
        )

    def _last_delivered_by_subject(self) -> dict[str, date]:
        last: dict[str, date] = {}
        for row in self.audit_rows():
            if not row.get("delivered"):
                continue
            subject = row.get("subject")
            try:
                when = date.fromisoformat(str(row.get("asof")))
            except (TypeError, ValueError):
                continue
            if subject and (subject not in last or when > last[subject]):
                last[subject] = when
        return last

    def _in_cooldown(self, subject: str, asof: date,
                     last: dict[str, date]) -> bool:
        previous = last.get(subject)
        if previous is None:
            return False
        elapsed = sessions_between(previous, asof)
        if elapsed is None:
            # Outside the verified calendar: do NOT push. Silence is the safe
            # failure here; a spurious alert is the expensive one.
            return True
        return elapsed < SUBJECT_COOLDOWN_SESSIONS

    # --- gating ----------------------------------------------------------

    def notify_transitions(self, queue_path: Path | str, *, asof: date) -> list[dict]:
        """Deliver at most one notification per NEW (item, state) pair."""
        month = asof.strftime("%Y-%m")
        seen = self._seen_keys()
        budget_left = MONTHLY_BUDGET - self._sent_this_month(month)
        last_by_subject = self._last_delivered_by_subject()
        delivered: list[dict] = []
        state_dirty = False

        for item in load_queue(queue_path).values():
            if item.severity not in _PUSHABLE:
                continue
            key = dedupe_key(item.advice_id, item.state)
            base = {
                "asof": asof.isoformat(),
                "dedupe_key": key,
                "advice_id": item.advice_id,
                "state": item.state,
                "severity": item.severity,
                "source_rule": item.source_rule,
                "subject": item.subject,
                "summary": item.summary,
                # Points at the RECORDING surface, never at an execution path.
                "action_hint": "record the outcome: tools/decision_queue_cli.py",
            }

            if key in seen:
                self._record({**base, "delivered": False,
                              "suppressed_reason": "duplicate"})
                continue
            seen.add(key)

            if self.rolled_back:
                self._record({**base, "delivered": False,
                              "suppressed_reason": "rolled_back_error_rate"})
                continue
            if not self.enabled_channels or self.sink is None:
                self._record({**base, "delivered": False,
                              "suppressed_reason": "no_enabled_channel"})
                continue
            if budget_left <= 0:
                self._record({**base, "delivered": False,
                              "suppressed_reason": "monthly_budget_exhausted"})
                continue
            if self._in_cooldown(item.subject, asof, last_by_subject):
                self._record({**base, "delivered": False,
                              "suppressed_reason": "subject_cooldown"})
                continue

            ok = False
            try:
                ok = bool(self.sink(base))
            except Exception:  # noqa: BLE001 - a broken channel must not raise here
                ok = False
            state_dirty = True
            if ok:
                budget_left -= 1
                self._failures = 0
                last_by_subject[item.subject] = asof
                delivered.append(base)
                self._record({**base, "delivered": True, "suppressed_reason": None})
            else:
                self._failures += 1
                self._record({**base, "delivered": False,
                              "suppressed_reason": "delivery_failed"})
                if self._failures >= MAX_CONSECUTIVE_FAILURES:
                    # Stop attempting for the rest of this run and stay silent
                    # until an operator calls reset_rollback(). A retry storm on
                    # a broken channel is the fastest route to a permanently
                    # ignored one.
                    self.rolled_back = True

        if state_dirty:
            self._save_state()
        return delivered

    # --- metrics ---------------------------------------------------------

    def monthly_metrics(self, month: str, *, queue_path: Path | str | None = None) -> dict:
        rows = [r for r in self.audit_rows() if str(r.get("asof", "")).startswith(month)]
        sent = [r for r in rows if r.get("delivered")]
        acknowledged = 0
        seen_spans: list[int] = []
        if queue_path is not None:
            notified = {r["advice_id"] for r in sent if r.get("advice_id")}
            for item in load_queue(queue_path).values():
                if item.advice_id in notified and item.acknowledged_asof:
                    acknowledged += 1
                    span = item.trigger_to_seen_sessions
                    if span is not None:
                        seen_spans.append(span)
        ordered = sorted(seen_spans)
        median = None
        if ordered:
            mid = len(ordered) // 2
            median = (float(ordered[mid]) if len(ordered) % 2
                      else (ordered[mid - 1] + ordered[mid]) / 2.0)
        def count(reason: str) -> int:
            return sum(1 for r in rows if r.get("suppressed_reason") == reason)

        failed = count("delivery_failed")
        return {
            "month": month,
            # attempted = actually handed to a channel (succeeded or not).
            # Reporting `sent` as a synonym for `delivered` hid every failure.
            "attempted": len(sent) + failed,
            "delivered": len(sent),
            "acknowledged": acknowledged,
            "duplicate_suppressed": count("duplicate"),
            "budget_suppressed": count("monthly_budget_exhausted"),
            "cooldown_suppressed": count("subject_cooldown"),
            "channel_disabled_suppressed": count("no_enabled_channel"),
            "rollback_suppressed": count("rolled_back_error_rate"),
            "delivery_failed": failed,
            "rolled_back": self.rolled_back,
            "consecutive_failures": self._failures,
            # None = never acknowledged. Not 0, which would assert a
            # same-session visibility that was never observed.
            "median_trigger_to_seen_sessions": median,
        }
