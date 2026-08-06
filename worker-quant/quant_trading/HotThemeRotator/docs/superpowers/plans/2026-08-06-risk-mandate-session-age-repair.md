# Risk-Mandate Session-Age Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Rule 17.4.7 sunset age count distinct consecutive JPX sessions and make same-date risk-snapshot reruns idempotent without rewriting historical trace rows.

**Architecture:** Keep the repair inside `tools/risk_mandate_snapshot.py`. Reduce the append-only trace to the last effective row per parseable `asof`, then count backward through only the covered JPX path needed by the current flag. A missing eligible-session row is `unobserved`, not `closed`: skip it without incrementing or resetting age, emit and persist a degraded-history warning, and reset only on an explicit observed row where the flag is absent. Same-date persistence suppresses identical reruns and records changed rows as explicit revisions. Preserve the existing `openSessions` inclusive field for compatibility; the later P29 decision queue will use age-zero elapsed-session semantics.

**Tech Stack:** Python 3, stdlib JSON/date/pathlib, existing `hot_theme_rotator.data.jpx_calendar`, pytest.

---

## Scope boundary

This is the first executable repair extracted from the broader P28-P33 design. It does not reconcile the owner-reported 8035.T fill, choose a band-breach response, build the P29 decision queue, change the mandate, or rewrite old trace lines. Those are separate operational or implementation plans because they have different inputs and approval gates.

Semantic decision for Rule 17.4.7: continuity ends only when a trace row explicitly observes the flag closed. Missing rows neither add age nor erase previously observed open sessions. Any missing, malformed, or uncovered history actually encountered by the backward walk degrades confidence and must be visible in stdout and the new trace row. An uncovered current `asof` disables age escalation for that run with a warning; an unrelated future or out-of-path row cannot poison the full history.

Execution order: start this zero-input code repair immediately. P28 remains the highest governance priority, but its final ledger reconciliation and band-breach disposition wait for the owner-reported fill and owner choice; that dependency must not block this plan.

### Task 1: Lock the distinct-session behavior with unit tests

**Files:**
- Create: `tests/unit/test_risk_mandate_snapshot.py`
- Read: `tools/risk_mandate_snapshot.py:36-56`
- Read: `src/hot_theme_rotator/data/jpx_calendar.py`

- [ ] **Step 1: Write the failing counter tests**

```python
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
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```powershell
python -m pytest tests/unit/test_risk_mandate_snapshot.py -q
```

Expected: failures reporting that `_flag_ages()` does not accept `current_asof` and still counts duplicate rows.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add -- tests/unit/test_risk_mandate_snapshot.py
git commit -m "test: lock risk flag session-age semantics"
```

### Task 2: Implement calendar-aware effective-row counting

**Files:**
- Modify: `tools/risk_mandate_snapshot.py:19-56`
- Test: `tests/unit/test_risk_mandate_snapshot.py`

- [ ] **Step 1: Import the existing calendar helpers**

Add beside the current project imports:

```python
from dataclasses import dataclass

from hot_theme_rotator.data.jpx_calendar import (  # noqa: E402
    calendar_covers,
    is_trading_day,
)
```

- [ ] **Step 2: Replace `_flag_ages` with effective-row and previous-session helpers**

```python
@dataclass(frozen=True)
class FlagAge:
    prior_observed_sessions: int
    observation_gap_sessions: int


def _effective_trace_rows(trace_path: Path) -> tuple[dict[_dt.date, dict], tuple[str, ...]]:
    """Return the last parseable row per asof plus non-fatal diagnostics."""
    if not trace_path.exists():
        return {}, ()
    try:
        raw_lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, (f"flag_age_trace_unreadable:{type(exc).__name__}",)

    effective: dict[_dt.date, dict] = {}
    warnings: list[str] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            warnings.append(f"malformed_trace_line:{line_number}")
            continue
        try:
            asof = _dt.date.fromisoformat(str(row["asof"]))
        except (KeyError, TypeError, ValueError):
            warnings.append(f"invalid_trace_asof:{line_number}")
            continue
        effective[asof] = row
    return effective, tuple(warnings)


def _previous_trading_day(d: _dt.date) -> _dt.date:
    prior = d - _dt.timedelta(days=1)
    for _ in range(10):
        if is_trading_day(prior):
            return prior
        prior -= _dt.timedelta(days=1)
    return prior


def _flag_ages(
    trace_path: Path,
    current_flags: dict,
    *,
    current_asof: str,
) -> tuple[dict[tuple[str, str], FlagAge], tuple[str, ...]]:
    """Count prior observed-open sessions; gaps degrade but do not reset."""
    if not current_flags:
        return {}, ()
    try:
        current_date = _dt.date.fromisoformat(current_asof)
    except ValueError:
        return {}, (f"flag_age_invalid_current_asof:{current_asof}",)
    if not calendar_covers(current_date) or not is_trading_day(current_date):
        return {}, (f"flag_age_calendar_uncovered:{current_asof}",)

    effective, load_warnings = _effective_trace_rows(trace_path)
    past_dates = [
        d for d in effective
        if d < current_date and calendar_covers(d) and is_trading_day(d)
    ]
    earliest = min(past_dates) if past_dates else current_date

    ages: dict[tuple[str, str], FlagAge] = {}
    warnings = list(load_warnings)
    for sid, flags in current_flags.items():
        for flag in flags:
            count = 0
            gaps = 0
            cursor = _previous_trading_day(current_date)
            while cursor >= earliest:
                if not calendar_covers(cursor):
                    warnings.append(f"flag_age_calendar_uncovered:{cursor.isoformat()}")
                    break
                row = effective.get(cursor)
                if row is None:
                    gaps += 1
                    cursor = _previous_trading_day(cursor)
                    continue
                row_flags = ((row.get("flags") or {}).get(sid) or [])
                if flag not in row_flags:
                    break
                count += 1
                cursor = _previous_trading_day(cursor)
            ages[(sid, flag)] = FlagAge(
                prior_observed_sessions=count,
                observation_gap_sessions=gaps,
            )
            if gaps:
                warnings.append(
                    f"flag_age_degraded:{sid}:{flag}:missing_sessions={gaps}"
                )
    return ages, tuple(dict.fromkeys(warnings))
```

- [ ] **Step 3: Pass `asof` from `main`**

Replace the existing call with:

```python
    ages, age_warnings = _flag_ages(
        obs / "risk_mandate_trace.jsonl",
        {k: v for k, v in flags.items() if v},
        current_asof=asof,
    )
    for warning in age_warnings:
        print(f"  WARNING {warning}")
```

Replace the escalation loop with:

```python
    escalations = []
    for (sid, flag), age in ages.items():
        open_sessions = age.prior_observed_sessions + 1
        if open_sessions >= sunset_n:
            escalation = {
                "sleeve": sid,
                "flag": flag,
                "openSessions": open_sessions,
                "observationGapSessions": age.observation_gap_sessions,
                "ageQuality": "degraded" if age.observation_gap_sessions else "complete",
            }
            escalations.append(escalation)
            print(
                f"  SUNSET [{sid}] '{flag}' open {open_sessions} observed sessions "
                f">= {sunset_n} (gaps={age.observation_gap_sessions}) - "
                "Rule 17.4.7 demands resolve"
            )
```

When constructing the trace `row`, persist warnings only when present:

```python
        if age_warnings:
            row["age_warnings"] = list(age_warnings)
```

`openSessions` remains the inclusive count of sessions where the flag was actually observed open. Missing sessions appear separately and do not increment or reset it. Do not relabel this as the age-zero P29 queue age.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/unit/test_risk_mandate_snapshot.py tests/unit/test_jpx_calendar.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the counter repair**

```powershell
git add -- tools/risk_mandate_snapshot.py tests/unit/test_risk_mandate_snapshot.py
git commit -m "fix: count risk flags by JPX session"
```

### Task 3: Make trace persistence idempotent and revision-aware

**Files:**
- Modify: `tools/risk_mandate_snapshot.py:139-155`
- Modify: `tests/unit/test_risk_mandate_snapshot.py`

- [ ] **Step 1: Add failing persistence tests**

Append:

```python
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
```

- [ ] **Step 2: Run the two tests and confirm RED**

Run:

```powershell
python -m pytest tests/unit/test_risk_mandate_snapshot.py -q
```

Expected: failures because `_append_trace_row` does not exist.

- [ ] **Step 3: Add the append helper**

```python
def _semantic_trace_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"asof_revision", "supersedes_revision"}
    }


def _append_trace_row(trace_path: Path, row: dict) -> str:
    """Append once per semantic state; preserve changed same-date revisions."""
    existing: list[dict] = []
    if trace_path.exists():
        try:
            existing = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError):
            existing = []

    same_date = [item for item in existing if item.get("asof") == row.get("asof")]
    prior_revision = max(
        (int(item.get("asof_revision", 1)) for item in same_date),
        default=0,
    )
    if same_date and _semantic_trace_row(same_date[-1]) == _semantic_trace_row(row):
        return "unchanged"

    output = {**row, "asof_revision": prior_revision + 1}
    if prior_revision:
        output["supersedes_revision"] = prior_revision
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(output, ensure_ascii=False) + "\n")
    return "revised" if prior_revision else "appended"
```

This deliberately does not delete or rewrite the duplicate historical lines. `_effective_trace_rows` makes the last row authoritative for counting; the later retrospective correction explains the old inflated values.

- [ ] **Step 4: Route `main` through the helper**

Replace the direct append block with:

```python
        trace_result = _append_trace_row(obs / "risk_mandate_trace.jsonl", row)
        print(
            f"wrote {obs / 'risk_mandate' / (asof + '.json')} "
            f"+ trace {trace_result}"
        )
```

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/unit/test_risk_mandate_snapshot.py tests/unit/test_daily_routine.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the persistence repair**

```powershell
git add -- tools/risk_mandate_snapshot.py tests/unit/test_risk_mandate_snapshot.py
git commit -m "fix: make risk trace reruns idempotent"
```

### Task 4: Recompute the golden timeline and document the correction

**Files:**
- Modify: `docs/proposals/retrospective_review_2026-08-04.md:110-126`
- Modify: `docs/01_TASKS.md` after P27
- Modify: `PROJECT_STATUS.md` current-state section and change log
- Reference: `reports/observability/risk_mandate_trace.jsonl`
- Reference: `docs/superpowers/specs/2026-08-06-evidence-based-retrospective-remediation-design.md:53-86`

- [ ] **Step 1: Correct the retrospective delay table**

Use these exact distinctions:

```markdown
| Item | Trigger/creation | Terminal/as-of | Elapsed eligible JPX sessions | Inclusive sessions |
|---|---:|---:|---:|---:|
| 8035 bracket exit | 2026-07-24 | 2026-08-04 | 7 | 8 |
| Sleeve A tranche 2+ | 2026-07-13 | 2026-08-03 | 14 | 15 |
| Sleeve B deployment | 2026-07-13 | 2026-08-03 | 14 | 15 |
```

Replace the provisional arithmetic with `JPY 62,800 - JPY 54,990 = JPY 7,810`, retaining the `provisional` label until the actual fill and fees enter the Section 14 journal.

- [ ] **Step 2: Add P28-P33 as proposed tasks**

Append this backlog block after P27. Keep every item `proposed - awaiting owner activation`; do not mark P28 complete while the sell fill or mandate-state decision is absent.

```markdown
## Milestone P28-P33: Evidence-Based Retrospective Remediation (proposed)

Status for all tasks: **proposed - awaiting owner activation**. No task authorizes broker execution, capital deployment, signal promotion, or a mandate-parameter change.

### P28 - Ledger, Retrospective, and Mandate-State Closure
- Record the actual 8035.T sell fill through Section 14 and remove the stale holding from the next snapshot.
- Reconcile NAV, realized/unrealized P&L, benchmark return, and active return to a documented tolerance.
- Report implementation shortfall using decision, eligible execution, actual execution, fees, and provisional/final status.
- Recompute delay tables using age zero on creation and label inclusive counts separately.
- Recompute exposure after 8035.T leaves the ledger.
- Obtain a dated owner decision: deploy, submit a Rule 4 band proposal, or approve a time-bounded exception with expiry.
- Mark the withdrawn low-exposure interpretation superseded in `PROJECT_STATUS.md`.

### P29 - Decision Queue and Execution Observability
- Persist deterministic advice IDs and append-only state transitions: `open -> acknowledged -> executed | declined | expired | superseded`.
- Record source rule, timestamp, JPX-session age, severity, evidence pointer, and structured decline reason.
- Make `_flag_ages` distinct-session based and make same-date snapshots idempotent or explicitly superseding.
- Report open-age distribution, terminal counts, trigger-to-seen, and trigger-to-terminal after close.
- Keep CLI/afterclose recording ahead of UI mutation; any HTTP write first amends Rule 11.5.

### P30 - Low-Noise State-Transition Notifications
- Enable one owner-selected Rule 12.7 double-confirmed channel.
- Notify state transitions only; deduplicate unchanged open states.
- Record severity, cooldown, monthly budget, delivery audit, and decision ID without order controls.
- Report sent, delivered, acknowledged, duplicate-suppressed, and trigger-to-seen metrics.
- Roll back automatically to silent mode on a predeclared error or duplicate-rate breach.

### P31 - Locked 63D Evidence Review Protocol
- Treat 2026-08-26 as the earliest review and emit `confirm`, `fail`, or `insufficient`.
- Count all attempted variants and report independent date clusters, rows, maturity, and missingness.
- Emit PIT, survivorship, costs, purge, embargo, DSR, PBO/CPCV, and t-stat checks.
- Report E/P and B/P independently and separate signal from deployment verdicts.
- Do not change capital or configuration from the report alone.

### P32 - Risk-Mandate Decision Memo
- Reproduce the ADR arithmetic and its line-34 error in one short memo.
- Show the 1.2731x-versus-1.4x growth and floor-hit trade-off as model outputs.
- Apply LETF drag and an as-of verified official fee only to the leveraged allocation.
- Show parameter-uncertainty assumptions and sensitivity.
- Present at least three owner alternatives: retain 1.4x and withdraw the 10% claim; align target with the fractional-Kelly bound; or abandon Kelly provenance and re-justify the band as owner preference.
- Defer bootstrap and jump/regime simulation unless the owner selects a risk-calibrated band, intends to occupy it, and states what decision simulation could change.
- Do not edit the active mandate.

### P33 - Three-Ledger KPI and Rule-Sunset Review
- Publish separate account, research, and execution scorecards with defined numerators, denominators, unavailable states, and as-of dates.
- Measure ledger lag and band compliance only with valid prices and reconciled positions.
- Scan runtime references from rules to config, code, tests, and reports.
- Put rules unused for six months on an owner-review list; never delete automatically.
- Preserve audit history when rules are merged or retired.
```

- [ ] **Step 3: Propagate the withdrawn status claim**

Add a current-state correction and annotate the 2026-07-18 entry:

```markdown
**SUPERSEDED INTERPRETATION (2026-08-06):** The favorable July outcome under low exposure is observed, but it is not a second empirical validation of low exposure as a strategy. The account was outside its declared exposure band, so policy compliance and outcome must be reported separately.
```

Preserve the original entry for audit history. Do not delete it.

- [ ] **Step 4: Run documentation checks**

Run:

```powershell
rg -n "9 sessions|15\+ sessions|approximately JPY 7,700|second empirical protection" docs/proposals/retrospective_review_2026-08-04.md PROJECT_STATUS.md
git diff --check -- docs/proposals/retrospective_review_2026-08-04.md docs/01_TASKS.md PROJECT_STATUS.md
```

Expected: the first command returns no unqualified live claims; historical text may remain only beside an explicit `SUPERSEDED` annotation. `git diff --check` returns no errors.

- [ ] **Step 5: Run the focused and non-slow regression lanes**

Run:

```powershell
python -m pytest tests/unit/test_risk_mandate_snapshot.py tests/unit/test_jpx_calendar.py tests/unit/test_daily_routine.py -q
python -m pytest -m "not slow" --basetemp .runtime/pytest-session-age -q
```

Expected: both commands exit 0; the second preserves the existing five slow deselections and introduces no failures.

- [ ] **Step 6: Commit the recomputation and proposed backlog**

```powershell
git add -- docs/proposals/retrospective_review_2026-08-04.md docs/01_TASKS.md PROJECT_STATUS.md
git commit -m "docs: correct retrospective session counts"
```

## Completion evidence

The implementation is complete only when:

- the duplicate 2026-07-28 and 2026-07-29 rows remain auditable but no longer inflate age;
- a replay for 2026-07-30 yields `openSessions=5`, not 7;
- a replay for 2026-08-05 yields `openSessions=9`, not 11;
- an unobserved eligible session neither increments nor resets age, and produces a persisted `flag_age_degraded` warning;
- an explicit observed flag absence resets continuity;
- an uncovered current calendar disables escalation with a visible warning, while an unrelated future row cannot poison current history;
- repeating the same `asof` with unchanged data appends no trace row;
- a changed same-date snapshot appends an explicit revision and does not add a session;
- the retrospective uses seven elapsed/eight inclusive sessions for the 8035.T delay;
- no broker execution, position, or mandate parameter changes occur.
