"""Tests for low-noise state-transition notification gating (P30).

Section 12's premise is that the push channel is itself a risk surface. The
medical alert-fatigue literature (Ancker et al. 2017) says repeated alerts
reduce acceptance — so the gate here is deliberately stingy: only TRANSITIONS
notify, an unchanged open state never repeats, and the budget/cooldown are
hard stops rather than advisories. A notifier that cries every day retrains
the owner to ignore exactly the binding item it exists to surface.

Everything ships DISABLED. Rule 12.7 double confirmation is the only way to
enable a channel, and these tests assert the default really is silence.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.alerts.transition_notifier import (  # noqa: E402
    MONTHLY_BUDGET,
    NotificationGate,
    dedupe_key,
)
from hot_theme_rotator.decision_queue import open_item, transition  # noqa: E402


class _Sink:
    """Test double for a delivery channel; records instead of delivering."""

    def __init__(self, fail: bool = False):
        self.sent: list[dict] = []
        self.fail = fail

    def __call__(self, payload: dict) -> bool:
        self.sent.append(payload)
        return not self.fail


def _gate(tmp_path: Path, sink=None, *, enabled=("desktop",)) -> NotificationGate:
    return NotificationGate(
        base_dir=tmp_path,
        sink=sink or _Sink(),
        enabled_channels=set(enabled),
    )


def _queue(tmp_path: Path) -> Path:
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    open_item(path, source_rule="17.4.6", subject="sleeve_C",
              summary="exit bracket breached on close",
              created_asof="2026-07-24", severity="binding")
    return path


# --- default silence ------------------------------------------------------

def test_no_enabled_channel_means_no_delivery(tmp_path):
    """Rule 12.7.1: every channel defaults disabled. Shipping must be silent."""
    sink = _Sink()
    gate = _gate(tmp_path, sink, enabled=())
    sent = gate.notify_transitions(_queue(tmp_path), asof=date(2026, 7, 24))
    assert sent == []
    assert sink.sent == []


def test_disabled_gate_still_records_what_it_would_have_sent(tmp_path):
    """Silent mode must remain measurable, or P30's own metrics are blind."""
    gate = _gate(tmp_path, enabled=())
    gate.notify_transitions(_queue(tmp_path), asof=date(2026, 7, 24))
    audit = gate.audit_rows()
    assert len(audit) == 1
    assert audit[0]["delivered"] is False
    assert audit[0]["suppressed_reason"] == "no_enabled_channel"


# --- transitions only -----------------------------------------------------

def test_a_new_transition_notifies_once(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    sent = gate.notify_transitions(_queue(tmp_path), asof=date(2026, 7, 24))
    assert len(sent) == 1 and len(sink.sent) == 1
    assert sink.sent[0]["severity"] == "binding"


def test_an_unchanged_open_state_never_repeats(tmp_path):
    """The 8035.T item stayed open 7 sessions. That is ONE notification."""
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = _queue(tmp_path)
    for day in (24, 27, 28, 29, 30):
        gate.notify_transitions(path, asof=date(2026, 7, day))
    assert len(sink.sent) == 1


def test_a_real_state_change_notifies_again(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = _queue(tmp_path)
    item_id = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["advice_id"]
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    transition(path, item_id, "executed", asof="2026-08-04")
    gate.notify_transitions(path, asof=date(2026, 8, 4))
    assert [p["state"] for p in sink.sent] == ["open", "executed"]


def test_dedupe_key_is_stable_per_item_and_state():
    a = dedupe_key("abc123", "open")
    assert a == dedupe_key("abc123", "open")
    assert a != dedupe_key("abc123", "executed")
    assert a != dedupe_key("def456", "open")


# --- content red-lines ----------------------------------------------------

def test_payload_links_the_decision_id_and_carries_no_order_control(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    gate.notify_transitions(_queue(tmp_path), asof=date(2026, 7, 24))
    payload = sink.sent[0]
    assert payload["advice_id"]
    assert "decision_queue_cli.py" in payload["action_hint"]
    blob = json.dumps(payload).lower()
    for forbidden in ("order_id", "account", "submit", "place_order", "quantity_to_sell"):
        assert forbidden not in blob


def test_informational_severity_is_never_pushed(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    open_item(path, source_rule="17.4", subject="sleeve_C", summary="fyi",
              created_asof="2026-07-24", severity="informational")
    assert gate.notify_transitions(path, asof=date(2026, 7, 24)) == []


# --- budget, cooldown, rollback ------------------------------------------

def test_monthly_budget_is_a_hard_stop(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    for n in range(MONTHLY_BUDGET + 3):
        open_item(path, source_rule=f"17.{n}", subject=f"s{n}", summary="x",
                  created_asof="2026-07-24", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    assert len(sink.sent) == MONTHLY_BUDGET
    reasons = {r.get("suppressed_reason") for r in gate.audit_rows()}
    assert "monthly_budget_exhausted" in reasons


def test_budget_resets_in_a_new_month(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    for n in range(MONTHLY_BUDGET):
        open_item(path, source_rule=f"17.{n}", subject=f"s{n}", summary="x",
                  created_asof="2026-07-24", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    open_item(path, source_rule="17.99", subject="august", summary="x",
              created_asof="2026-08-03", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 8, 3))
    assert len(sink.sent) == MONTHLY_BUDGET + 1


def test_delivery_failures_roll_back_to_silent_mode(tmp_path):
    """Rather than retry-storm a broken channel into the owner's face."""
    sink = _Sink(fail=True)
    gate = _gate(tmp_path, sink)
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    for n in range(4):
        open_item(path, source_rule=f"17.{n}", subject=f"s{n}", summary="x",
                  created_asof="2026-07-24", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    assert gate.rolled_back is True
    assert len(sink.sent) <= 3            # stopped early, did not attempt all
    assert any(r.get("suppressed_reason") == "rolled_back_error_rate"
               for r in gate.audit_rows())


def test_same_subject_is_cooled_down_across_sessions(tmp_path):
    """Dedupe stops the SAME item repeating; cooldown stops one noisy subject.

    Both are needed: a sleeve that generates a fresh binding item every couple
    of sessions would otherwise push every time with dedupe fully satisfied.
    """
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    open_item(path, source_rule="17.4.6", subject="sleeve_C", summary="first",
              created_asof="2026-07-24", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 24))

    open_item(path, source_rule="17.4.4", subject="sleeve_C", summary="second",
              created_asof="2026-07-27", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 27))
    assert len(sink.sent) == 1
    assert any(r.get("suppressed_reason") == "subject_cooldown"
               for r in gate.audit_rows())


def test_cooldown_expires_and_a_different_subject_is_unaffected(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    open_item(path, source_rule="17.4.6", subject="sleeve_C", summary="first",
              created_asof="2026-07-24", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 24))

    # A different subject is never held back by sleeve_C's cooldown.
    open_item(path, source_rule="17.2", subject="portfolio", summary="band",
              created_asof="2026-07-27", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 27))
    assert len(sink.sent) == 2

    # Past the cooldown window, sleeve_C may speak again.
    open_item(path, source_rule="17.4.4", subject="sleeve_C", summary="later",
              created_asof="2026-08-04", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 8, 4))
    assert len(sink.sent) == 3


def _binding(path: Path, n: int, *, asof: str = "2026-07-24") -> None:
    for i in range(n):
        open_item(path, source_rule=f"17.{i}", subject=f"s{i}", summary="x",
                  created_asof=asof, severity="binding")


def test_rollback_survives_a_new_process(tmp_path):
    """afterclose is a fresh process every session. In-memory rollback would
    silently rearm a broken channel tomorrow — the retry storm, one day apart.
    """
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    _binding(path, 4)
    first = _gate(tmp_path, _Sink(fail=True))
    first.notify_transitions(path, asof=date(2026, 7, 24))
    assert first.rolled_back is True

    revived = _gate(tmp_path, _Sink())          # new process, healthy sink
    assert revived.rolled_back is True
    _binding(path, 1, asof="2026-08-04")
    assert revived.notify_transitions(path, asof=date(2026, 8, 4)) == []


def test_failure_streak_accumulates_across_processes(tmp_path):
    """Two failures yesterday plus one today is still three in a row."""
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    _binding(path, 2)
    _gate(tmp_path, _Sink(fail=True)).notify_transitions(path, asof=date(2026, 7, 24))

    _binding(path, 1, asof="2026-08-04")
    second = _gate(tmp_path, _Sink(fail=True))
    assert second.rolled_back is False          # loaded streak = 2, not yet tripped
    second.notify_transitions(path, asof=date(2026, 8, 4))
    assert second.rolled_back is True


def test_a_success_clears_the_persisted_streak(tmp_path):
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    _binding(path, 2)
    _gate(tmp_path, _Sink(fail=True)).notify_transitions(path, asof=date(2026, 7, 24))

    _binding(path, 1, asof="2026-08-04")
    healthy = _gate(tmp_path, _Sink())
    healthy.notify_transitions(path, asof=date(2026, 8, 4))

    _binding(path, 1, asof="2026-08-11")
    third = _gate(tmp_path, _Sink(fail=True))
    third.notify_transitions(path, asof=date(2026, 8, 11))
    assert third.rolled_back is False           # streak restarted at 1, not 3


def test_rollback_requires_an_explicit_reset(tmp_path):
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    _binding(path, 4)
    gate = _gate(tmp_path, _Sink(fail=True))
    gate.notify_transitions(path, asof=date(2026, 7, 24))

    healed = _gate(tmp_path, _Sink())
    healed.reset_rollback(asof="2026-08-04", note="channel repaired")
    assert healed.rolled_back is False
    _binding(path, 1, asof="2026-08-04")
    assert len(healed.notify_transitions(path, asof=date(2026, 8, 4))) == 1


def test_metrics_separate_attempts_from_deliveries(tmp_path):
    """'sent' and 'delivered' being one number hides every failed attempt."""
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    _binding(path, 2)
    gate = _gate(tmp_path, _Sink(fail=True))
    gate.notify_transitions(path, asof=date(2026, 7, 24))

    metrics = gate.monthly_metrics("2026-07")
    assert metrics["attempted"] == 2
    assert metrics["delivered"] == 0
    assert metrics["delivery_failed"] == 2


def test_metrics_count_cooldown_suppression_separately(tmp_path):
    path = tmp_path / "reports" / "observability" / "decision_queue.jsonl"
    open_item(path, source_rule="17.4.6", subject="sleeve_C", summary="a",
              created_asof="2026-07-24", severity="binding")
    gate = _gate(tmp_path, _Sink())
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    open_item(path, source_rule="17.4.4", subject="sleeve_C", summary="b",
              created_asof="2026-07-27", severity="binding")
    gate.notify_transitions(path, asof=date(2026, 7, 27))

    metrics = gate.monthly_metrics("2026-07")
    assert metrics["cooldown_suppressed"] == 1
    assert metrics["attempted"] == 1 and metrics["delivered"] == 1


def test_monthly_metrics_separate_suppressed_from_never_seen(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = _queue(tmp_path)
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    gate.notify_transitions(path, asof=date(2026, 7, 27))   # duplicate suppressed

    metrics = gate.monthly_metrics("2026-07")
    assert metrics["attempted"] == 1
    assert metrics["delivered"] == 1
    assert metrics["duplicate_suppressed"] == 1
    # Nobody acknowledged: unobserved, not zero-latency.
    assert metrics["acknowledged"] == 0
    assert metrics["median_trigger_to_seen_sessions"] is None


def test_acknowledgement_flows_into_the_metrics(tmp_path):
    sink = _Sink()
    gate = _gate(tmp_path, sink)
    path = _queue(tmp_path)
    item_id = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["advice_id"]
    gate.notify_transitions(path, asof=date(2026, 7, 24))
    transition(path, item_id, "acknowledged", asof="2026-07-27")

    metrics = gate.monthly_metrics("2026-07", queue_path=path)
    assert metrics["acknowledged"] == 1
    assert metrics["median_trigger_to_seen_sessions"] == 1.0
