"""Tests for the advice decision queue (P29, Rule 3 / Rule 13.9-analogous).

The queue exists because the 2026-08-04 retrospective found the advice→owner
interface unmeasurable: "decided not to act" and "never saw it" were
indistinguishable, so the delay could not be observed, let alone improved.
Every invariant here serves that: deterministic identity (a flag that fires
twice on one session is ONE item), append-only transitions (history is never
rewritten), structured decline reasons (a decline is a decision, not a gap),
and JPX-session ages (calendar days would overstate every delay).
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

import pytest  # noqa: E402

from hot_theme_rotator.decision_queue import (  # noqa: E402
    ALLOWED_DECLINE_REASONS,
    TERMINAL_STATES,
    DecisionQueueError,
    advice_id,
    load_queue,
    open_item,
    queue_report,
    transition,
)


def _q(tmp_path: Path) -> Path:
    return tmp_path / "decision_queue.jsonl"


def _open_one(path: Path, *, created_asof: str = "2026-07-24") -> str:
    return open_item(
        path,
        source_rule="17.4.6",
        subject="8035.T",
        summary="exit bracket lower bound breached on close",
        created_asof=created_asof,
        severity="binding",
        evidence_ref="reports/observability/risk_mandate/2026-07-24.json",
    )


# --- identity -------------------------------------------------------------

def test_advice_id_is_deterministic_and_scoped_to_the_session():
    a = advice_id(source_rule="17.4.6", subject="8035.T", created_asof="2026-07-24")
    b = advice_id(source_rule="17.4.6", subject="8035.T", created_asof="2026-07-24")
    c = advice_id(source_rule="17.4.6", subject="8035.T", created_asof="2026-07-27")
    d = advice_id(source_rule="17.4.7", subject="8035.T", created_asof="2026-07-24")
    assert a == b
    assert a != c and a != d
    assert len(a) == 16


def test_reopening_the_same_advice_is_idempotent(tmp_path):
    """An afterclose rerun, or a flag still open tomorrow, must not duplicate."""
    path = _q(tmp_path)
    first = _open_one(path)
    second = _open_one(path)
    assert first == second
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(load_queue(path)) == 1


# --- append-only transitions ---------------------------------------------

def test_transition_appends_and_never_rewrites_history(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    transition(path, item_id, "acknowledged", asof="2026-07-27")
    transition(path, item_id, "executed", asof="2026-08-04",
               note="S-kabu market order, opening match")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["state"] for r in rows] == ["open", "acknowledged", "executed"]
    assert all(r["advice_id"] == item_id for r in rows)
    assert load_queue(path)[item_id].state == "executed"


def test_open_may_go_straight_to_a_terminal_state(tmp_path):
    """The owner can act without the system ever having been acknowledged.

    Forcing acknowledge-first would make the ledger unable to record what
    actually happened — which is precisely the 8035.T failure this queue exists
    to measure. `acknowledged` is an optional observation, not a gate.
    """
    path = _q(tmp_path)
    item_id = _open_one(path)
    transition(path, item_id, "executed", asof="2026-08-04")
    assert load_queue(path)[item_id].state == "executed"


@pytest.mark.parametrize("terminal", sorted(TERMINAL_STATES))
def test_terminal_states_are_final(tmp_path, terminal):
    path = _q(tmp_path)
    item_id = _open_one(path)
    kwargs = {"reason": "user_disagrees"} if terminal == "declined" else {}
    transition(path, item_id, terminal, asof="2026-07-27", **kwargs)
    with pytest.raises(DecisionQueueError, match="terminal"):
        transition(path, item_id, "acknowledged", asof="2026-07-28")


def test_unknown_state_and_unknown_item_are_rejected(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    with pytest.raises(DecisionQueueError, match="unknown state"):
        transition(path, item_id, "done", asof="2026-07-27")
    with pytest.raises(DecisionQueueError, match="unknown advice"):
        transition(path, "0" * 16, "acknowledged", asof="2026-07-27")


def test_repeating_the_current_state_is_idempotent_not_an_error(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    transition(path, item_id, "acknowledged", asof="2026-07-27")
    transition(path, item_id, "acknowledged", asof="2026-07-27")
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2  # open + one acknowledged


# --- decline discipline ---------------------------------------------------

def test_decline_requires_a_structured_reason(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    with pytest.raises(DecisionQueueError, match="structured reason"):
        transition(path, item_id, "declined", asof="2026-07-27")


def test_decline_rejects_a_free_text_only_reason(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    with pytest.raises(DecisionQueueError, match="structured reason"):
        transition(path, item_id, "declined", asof="2026-07-27",
                   reason="didn't feel like it")


def test_decline_records_reason_and_optional_note(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    transition(path, item_id, "declined", asof="2026-07-27",
               reason="user_disagrees", note="holding through the verdict date")
    item = load_queue(path)[item_id]
    assert item.state == "declined"
    assert item.decline_reason == "user_disagrees"
    assert item.note == "holding through the verdict date"
    assert "user_disagrees" in ALLOWED_DECLINE_REASONS


# --- session ages ---------------------------------------------------------

def test_item_age_counts_jpx_sessions_not_calendar_days(tmp_path):
    """8035.T: 2026-07-24 -> 2026-08-04 is 11 calendar days, 7 elapsed sessions."""
    path = _q(tmp_path)
    item_id = _open_one(path)
    item = load_queue(path)[item_id]
    assert item.age_sessions(date(2026, 8, 4)) == 7
    assert item.age_sessions(date(2026, 7, 24)) == 0


def test_age_is_none_outside_the_verified_calendar(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path, created_asof="2026-12-30")
    assert load_queue(path)[item_id].age_sessions(date(2027, 1, 8)) is None


def test_trigger_to_seen_and_trigger_to_terminal_are_separately_measurable(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    transition(path, item_id, "acknowledged", asof="2026-07-27")
    transition(path, item_id, "executed", asof="2026-08-04")
    item = load_queue(path)[item_id]
    assert item.trigger_to_seen_sessions == 1
    assert item.trigger_to_terminal_sessions == 7


def test_trigger_to_seen_is_none_when_never_acknowledged(tmp_path):
    """Unobserved, not zero — the 8035.T case must not read as 'seen same day'."""
    path = _q(tmp_path)
    item_id = _open_one(path)
    transition(path, item_id, "executed", asof="2026-08-04")
    item = load_queue(path)[item_id]
    assert item.trigger_to_seen_sessions is None
    assert item.trigger_to_terminal_sessions == 7


# --- reporting ------------------------------------------------------------

def test_queue_report_counts_terminal_states_and_open_ages(tmp_path):
    path = _q(tmp_path)
    stale = _open_one(path)
    fresh = open_item(path, source_rule="17.2", subject="portfolio",
                      summary="exposure below band", created_asof="2026-08-03",
                      severity="binding")
    done = open_item(path, source_rule="17.4.7", subject="8035.T",
                     summary="flag sunset", created_asof="2026-07-30",
                     severity="advisory")
    transition(path, done, "declined", asof="2026-07-31", reason="out_of_scope")

    report = queue_report(path, asof=date(2026, 8, 4))
    assert report["open_count"] == 2
    assert report["terminal_counts"]["declined"] == 1
    assert report["oldest_open_sessions"] == 7
    ages = {row["advice_id"]: row["age_sessions"] for row in report["open_items"]}
    assert ages[stale] == 7
    assert ages[fresh] == 1
    assert done not in ages


def test_queue_report_on_empty_queue_is_honest_not_zero_filled(tmp_path):
    report = queue_report(_q(tmp_path), asof=date(2026, 8, 4))
    assert report["open_count"] == 0
    assert report["oldest_open_sessions"] is None   # N/A, not 0
    assert report["available"] is True


def test_corrupt_line_degrades_visibly_rather_than_silently_dropping(tmp_path):
    path = _q(tmp_path)
    item_id = _open_one(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not-json\n")
    report = queue_report(path, asof=date(2026, 8, 4))
    assert report["warnings"] == ["malformed_queue_line:2"]
    assert item_id in {row["advice_id"] for row in report["open_items"]}
