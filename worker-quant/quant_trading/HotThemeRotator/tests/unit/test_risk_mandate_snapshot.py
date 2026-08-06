"""Tests for the Rule 17.4.7 flag-sunset session counter (risk mandate snapshot).

The counter must age flags by DISTINCT covered JPX sessions, never by trace
rows: a same-date rerun must not advance age, and duplicate historical rows
must not inflate it. A missing eligible session is ``unobserved`` — it neither
increments nor resets age, but it degrades confidence visibly. Continuity ends
only when a row explicitly observes the flag absent. Malformed or uncovered
history warns rather than silently disabling the escalation (Rule 11.9.4
honest degradation, not fabricated continuity).
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import tools.risk_mandate_snapshot as rms


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(asof: str, present: bool = True) -> dict:
    return {
        "asof": asof,
        "flags": {"C": ["exit_triggered"]} if present else {},
    }


def _panel() -> dict:
    return {
        "navJpy": 384_321.0,
        "cashJpy": 283_463.0,
        "exposure": {
            "betaAdjustedJpy": 159_586.0,
            "ratio": 0.415,
            "bandStatus": "below_band",
        },
        "killSwitch": {
            "bufferJpy": 284_321.0,
            "bufferPct": 73.98,
            "breached": False,
        },
        "mandate": {"flagSunsetSessions": 7},
        "sleeves": [{"id": "C", "flags": ["exit_triggered"], "holdings": []}],
        "sectorLookThrough": [],
    }


def test_flag_ages_deduplicates_same_asof_and_counts_prior_sessions(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [
        _row("2026-07-24"),
        _row("2026-07-27"),
        _row("2026-07-28"),
        _row("2026-07-28"),
        _row("2026-07-29"),
        _row("2026-07-29"),
    ])

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2026-07-30",
    )

    assert ages == {
        ("C", "exit_triggered"): rms.FlagAge(
            prior_observed_sessions=4,
            observation_gap_sessions=0,
            first_observed_asof="2026-07-24",
        )
    }
    assert warnings == ()


def test_flag_ages_ignores_existing_current_asof_on_rerun(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [
        _row("2026-07-24"),
        _row("2026-07-27"),
        _row("2026-07-28"),
        _row("2026-07-29"),
        _row("2026-07-30"),
    ])

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2026-07-30",
    )

    assert ages[("C", "exit_triggered")].prior_observed_sessions == 4
    assert ages[("C", "exit_triggered")].observation_gap_sessions == 0
    assert warnings == ()


def test_flag_ages_skips_missing_session_without_increment_or_reset(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [
        _row("2026-07-27"),
        _row("2026-07-29"),
    ])

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2026-07-30",
    )

    assert ages[("C", "exit_triggered")] == rms.FlagAge(
        prior_observed_sessions=2,
        observation_gap_sessions=1,
        first_observed_asof="2026-07-27",
    )
    assert warnings == (
        "flag_age_degraded:C:exit_triggered:missing_sessions=1",
    )


def test_first_observed_asof_is_stable_while_the_flag_stays_open(tmp_path):
    """Queue identity depends on this: a persistent condition must keep ONE id.

    If the first-observed date drifted forward each session, every afterclose
    would mint a new advice item for the same unresolved condition and the open
    age would reset daily — the failure the queue exists to prevent.
    """
    trace = tmp_path / "trace.jsonl"
    _write(trace, [_row(d) for d in ("2026-07-24", "2026-07-27", "2026-07-28")])

    for asof in ("2026-07-29", "2026-07-30", "2026-07-31"):
        ages, _ = rms._flag_ages(trace, {"C": ["exit_triggered"]}, current_asof=asof)
        assert ages[("C", "exit_triggered")].first_observed_asof == "2026-07-24"


def test_first_observed_asof_is_none_when_today_is_the_first_sighting(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [_row("2026-07-29", present=False)])

    ages, _ = rms._flag_ages(trace, {"C": ["exit_triggered"]}, current_asof="2026-07-30")
    assert ages[("C", "exit_triggered")].prior_observed_sessions == 0
    assert ages[("C", "exit_triggered")].first_observed_asof is None


def test_flag_ages_stops_at_closed_flag_before_reopen(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [
        _row("2026-07-28"),
        _row("2026-07-29", present=False),
    ])

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2026-07-30",
    )

    assert ages[("C", "exit_triggered")].prior_observed_sessions == 0
    assert warnings == ()


def test_flag_ages_skips_corrupt_line_and_warns_instead_of_silencing(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(_row("2026-07-29")) + "\nnot-json\n",
        encoding="utf-8",
    )

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2026-07-30",
    )

    assert ages[("C", "exit_triggered")].prior_observed_sessions == 1
    assert warnings == ("malformed_trace_line:2",)


def test_flag_ages_warns_and_disables_escalation_when_current_calendar_uncovered(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [_row("2027-01-04")])

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2027-01-05",
    )

    assert ages == {}
    assert warnings == ("flag_age_calendar_uncovered:2027-01-05",)


def test_unrelated_future_uncovered_row_does_not_poison_current_path(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [
        _row("2026-07-29"),
        _row("2027-01-04"),
    ])

    ages, warnings = rms._flag_ages(
        trace,
        {"C": ["exit_triggered"]},
        current_asof="2026-07-30",
    )

    assert ages[("C", "exit_triggered")].prior_observed_sessions == 1
    assert warnings == ()


def test_main_surfaces_and_persists_degraded_age_warning(tmp_path, monkeypatch, capsys):
    trace = tmp_path / "reports" / "observability" / "risk_mandate_trace.jsonl"
    trace.parent.mkdir(parents=True)
    _write(trace, [_row("2026-07-27"), _row("2026-07-29")])
    monkeypatch.setattr(rms, "_positions_dict", lambda: {})
    monkeypatch.setattr(rms, "build_risk_mandate_panel", lambda *_args, **_kwargs: _panel())

    assert rms.main(["--asof", "2026-07-30", "--base-dir", str(tmp_path)]) == 0

    assert "WARNING flag_age_degraded:C:exit_triggered:missing_sessions=1" in capsys.readouterr().out
    written = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert written[-1]["age_warnings"] == [
        "flag_age_degraded:C:exit_triggered:missing_sessions=1"
    ]


def test_queue_sync_opens_one_binding_item_per_unresolved_condition(tmp_path):
    """P29 auto-open: binding mandate conditions become first-class advice."""
    trace = tmp_path / "trace.jsonl"
    _write(trace, [_row(d) for d in ("2026-07-24", "2026-07-27", "2026-07-28")])
    queue = tmp_path / "decision_queue.jsonl"

    ages, _ = rms._flag_ages(trace, {"C": ["exit_triggered"]}, current_asof="2026-07-29")
    opened = rms._queue_sync(
        queue, trace,
        asof="2026-07-29",
        flags={"C": ["exit_triggered"]},
        ages=ages,
        band_status="below_band",
    )

    from hot_theme_rotator.decision_queue import load_queue
    items = load_queue(queue)
    assert len(opened) == 2 and len(items) == 2
    by_rule = {i.source_rule: i for i in items.values()}
    # The flag item is keyed to when the condition FIRST appeared, not today.
    assert by_rule["17.4.6"].created_asof == "2026-07-24"
    assert by_rule["17.4.6"].severity == "binding"
    # The standing band breach is the retrospective's "third state".
    assert by_rule["17.2"].subject == "portfolio"
    assert by_rule["17.2"].severity == "binding"


def test_queue_sync_is_idempotent_across_sessions_while_condition_persists(tmp_path):
    """The 8035.T case: 7 sessions open must be ONE item aging, not 7 items."""
    trace = tmp_path / "trace.jsonl"
    rows = ["2026-07-24", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30"]
    _write(trace, [_row(d) for d in rows])
    queue = tmp_path / "decision_queue.jsonl"

    from hot_theme_rotator.decision_queue import load_queue
    for asof in ("2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"):
        ages, _ = rms._flag_ages(trace, {"C": ["exit_triggered"]}, current_asof=asof)
        rms._queue_sync(queue, trace, asof=asof, flags={"C": ["exit_triggered"]},
                        ages=ages, band_status="within_band")

    items = load_queue(queue)
    assert len(items) == 1
    item = next(iter(items.values()))
    assert item.created_asof == "2026-07-24"
    assert item.age_sessions(_dt.date(2026, 7, 31)) == 5


def test_queue_sync_skips_advisory_flags_and_healthy_band(tmp_path):
    """Not every state is advice. Queueing noise trains the owner to ignore it."""
    trace = tmp_path / "trace.jsonl"
    _write(trace, [{"asof": "2026-07-28", "flags": {"C": ["thesis_missing"]},
                    "band_status": "within_band"}])
    queue = tmp_path / "decision_queue.jsonl"

    ages, _ = rms._flag_ages(trace, {"C": ["thesis_missing"]}, current_asof="2026-07-29")
    opened = rms._queue_sync(queue, trace, asof="2026-07-29",
                             flags={"C": ["thesis_missing"]}, ages=ages,
                             band_status="within_band")
    assert opened == []
    assert not queue.exists()


def test_queue_sync_failure_never_breaks_the_snapshot(tmp_path, monkeypatch, capsys):
    """Rule 15.5: a diagnostic must never block collection."""
    trace = tmp_path / "trace.jsonl"
    _write(trace, [_row("2026-07-28")])
    queue = tmp_path / "decision_queue.jsonl"
    monkeypatch.setattr(
        rms, "open_item",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire")))

    ages, _ = rms._flag_ages(trace, {"C": ["exit_triggered"]}, current_asof="2026-07-29")
    assert rms._queue_sync(queue, trace, asof="2026-07-29",
                           flags={"C": ["exit_triggered"]}, ages=ages,
                           band_status="below_band") == []
    assert "queue_sync_failed" in capsys.readouterr().out


def test_band_breach_item_keys_to_first_out_of_band_session(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write(trace, [
        {"asof": "2026-07-24", "flags": {}, "band_status": "within_band"},
        {"asof": "2026-07-27", "flags": {}, "band_status": "below_band"},
        {"asof": "2026-07-28", "flags": {}, "band_status": "below_band"},
    ])
    queue = tmp_path / "decision_queue.jsonl"

    rms._queue_sync(queue, trace, asof="2026-07-29", flags={}, ages={},
                    band_status="below_band")

    from hot_theme_rotator.decision_queue import load_queue
    item = next(iter(load_queue(queue).values()))
    assert item.created_asof == "2026-07-27"
    assert item.age_sessions(_dt.date(2026, 7, 29)) == 2


def test_append_trace_row_suppresses_identical_same_asof(tmp_path):
    trace = tmp_path / "trace.jsonl"
    row = _row("2026-07-30")

    assert rms._append_trace_row(trace, row) == "appended"
    assert rms._append_trace_row(trace, row) == "unchanged"
    assert len(trace.read_text(encoding="utf-8").splitlines()) == 1


def test_append_trace_row_records_changed_same_asof_as_revision(tmp_path):
    trace = tmp_path / "trace.jsonl"
    first = _row("2026-07-30")
    changed = {**first, "nav_jpy": 123.0}

    assert rms._append_trace_row(trace, first) == "appended"
    assert rms._append_trace_row(trace, changed) == "revised"

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["asof_revision"] == 1
    assert rows[1]["asof_revision"] == 2
    assert rows[1]["supersedes_revision"] == 1
