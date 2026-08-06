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
    )
    assert warnings == (
        "flag_age_degraded:C:exit_triggered:missing_sessions=1",
    )


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
