# M10 Daily Tasklist - 2026-03-13

- Date: 2026-03-13
- Owners: QA / Architecture
- Scope: M10 mainline closeout under the approved limited enforced cohort
- Status: READY FOR EXECUTION

---

## 1. Today's Objective

Today's objective is not to add new product capability.

Today's objective is to move M10 from:

- "partial technical recovery has been observed"

to:

- "the current M10 state is governed by one authoritative baseline, one auditable evidence set, and one decision-ready conclusion"

In practical terms, today must achieve all of the following:

1. lock one authoritative M10 runtime and evidence baseline
2. reconcile `T-32` status with its actual artifacts
3. start and structure `T-33` failure-injection validation
4. eliminate document / governance / artifact drift

---

## 2. What Must Be Achieved Today

By the end of today, the project should have:

1. one frozen authoritative snapshot for current M10 assessment
2. one explicit mainline execution lane that is not mixed with provider triage lanes
3. one clear `T-32` verdict backed by artifacts rather than narrative interpretation
4. one defined and started `T-33` chaos / failure-injection track
5. one synchronized set of progress, tasklist, and governance notes

If these are not achieved, M10 remains in-progress and is not yet in a sign-off-ready state.

---

## 3. Execution Rules For Today

- no cohort widening
- no new autonomy expansion
- no new feature surface
- no mixed evidence between stable mainline validation and `Qwen on opencode` triage
- no "architectural pass" claim without a matching authoritative artifact
- every conclusion must point to one explicit snapshot and one explicit artifact set

---

## 4. Priority Tasklist

## P0 - Freeze The Authoritative Baseline

### T-01 Authoritative Snapshot

**Goal**

Freeze one authoritative M10 assessment baseline for today's work.

**Required outcome**

- capture current commit / workspace state
- capture current runtime config posture
- capture current execution lane, provider, and model identity
- capture the authoritative artifact roots used for today's assessment
- define which evidence is current and which evidence is only historical

**Done when**

- one human-readable snapshot note exists
- one machine-reviewable snapshot block exists
- all later updates reference that snapshot

---

### T-02 Mainline Execution Lane Decision

**Goal**

Ensure that today's M10 validation runs against one explicit mainline execution lane.

**Required outcome**

- declare one primary lane for current M10 validation
- declare any fallback or recovery lane explicitly
- keep `Qwen on opencode` as an isolated triage lane unless proven otherwise
- prevent evidence from multiple lanes being merged into one pass claim

**Done when**

- the documentation names:
  - `primary_execution_lane`
  - `allowed_validation_lane`
  - `triage_only_lane`

---

## P0 - Close The T-32 Evidence Gap

### T-03 Existing T-32 Artifact Triage

**Goal**

Separate historical failing load-test artifacts from the current candidate authoritative result.

**Required outcome**

- classify existing `m10_load_test` artifacts into:
  - `historical_failure`
  - `diagnostic_intermediate`
  - `candidate_authoritative`
- identify which previously observed failures were:
  - architecture bottlenecks
  - release-pack / `go_no_go` evidence gaps
  - unsettled workflow runs
  - runtime/provider lane issues

**Done when**

- one triage summary exists for all current `T-32` artifacts
- old `FAIL` artifacts are no longer mistaken for the latest truth

---

### T-04 Authoritative T-32 Rerun Or Reclassification

**Goal**

Produce one decision-ready `T-32` status.

**Required outcome**

- either run the authoritative `T-32` baseline again under the frozen snapshot
- or formally reclassify `T-32` as still in progress if the evidence set is not yet closure-grade
- verify:
  - terminal-state settlement
  - duplicate-finalization absence
  - release-pack completeness
  - `go_no_go_result.json` presence or an explicitly approved alternative
  - no unrecovered stale result backlog

**Done when**

- `T-32` is represented by one authoritative result:
  - `PASS`
  - or `FAIL/BLOCKED`
- project documents stop carrying dual interpretations

---

## P0 - Start T-33 Failure Injection

### T-05 T-33 Fault Matrix

**Goal**

Define the exact M10 resilience checks that must still be proven.

**Required outcome**

- define at least these scenarios:
  1. main-workspace drift immediately before promote
  2. worker crash or timeout during enforced parallel execution
  3. result-consumer recovery disturbance during `XAUTOCLAIM`-based stale-message handling
- define for each:
  - injection point
  - expected system behavior
  - expected artifact / observability signal
  - pass / fail oracle

**Done when**

- one fault matrix note exists and is reviewable

---

### T-06 First-Pass T-33 Execution

**Goal**

Prove that M10 resilience is not only happy-path deep.

**Required outcome**

- execute at least one first-pass failure-injection run
- verify:
  - `PROMOTION_CONFLICT` triggers deterministically on injected drift
  - branch failure collapses safely under worker disruption
  - stale-result recovery remains bounded and observable

**Done when**

- one first-pass fault-injection artifact exists
- any failure is registered as an explicit M10 blocker

---

## P1 - Governance And Documentation Alignment

### T-07 Update `PROGRESS_LATEST.md`

**Goal**

Make the latest progress document reflect only artifact-backed truth.

**Required outcome**

- state the real current `T-32` status
- state the real current `T-33` status
- name the current blocker explicitly if closure is incomplete
- distinguish historical failed runs from current authoritative evidence

**Done when**

- progress documentation matches today's authoritative snapshot and evidence set

---

### T-08 Update M10 Tasklist State

**Goal**

Ensure the M10 tasklist is a true execution board rather than an optimistic summary.

**Required outcome**

- reconcile real status for:
  - `T-21`
  - `T-32`
  - `T-33`
- mark incomplete work as incomplete
- make conditional interpretations explicit

**Done when**

- the tasklist can be used directly in review without extra oral correction

---

### T-09 QA / Architecture Sign-off Note

**Goal**

Create a one-page assessment for today's review boundary.

**Required outcome**

- define:
  - today's review scope
  - frozen baseline
  - validated facts
  - unresolved risks
  - next approved step
- if sign-off is not granted, name the blocker and owner clearly

**Done when**

- one sign-off note exists and can feed the next go / no-go review

---

## P1 - Keep Provider Triage Out Of Mainline Truth

### T-10 `Qwen on opencode` Triage Isolation

**Goal**

Prevent unresolved provider-path issues from contaminating mainline validation status.

**Required outcome**

- keep `Qwen on opencode` explicitly labeled as a separate triage track
- keep mainline M10 validation tied to the known-good lane only
- require one explicit provider-triage conclusion instead of leaving the status implicit

**Done when**

- mainline evidence and provider triage evidence are no longer mixed

---

## 5. Required Outputs By End Of Day

The day is only complete if the following outputs exist:

1. authoritative M10 snapshot
2. `T-32` evidence triage summary
3. authoritative `T-32` verdict artifact or explicit blocked-state note
4. `T-33` fault matrix
5. at least one first-pass `T-33` execution artifact
6. updated `PROGRESS_LATEST.md`
7. updated M10 tasklist state
8. QA / Architecture sign-off note

---

## 6. Completion Criteria

Today is successful only if all of the following are true:

1. one authoritative M10 baseline is frozen
2. one mainline execution lane is clearly defined
3. `T-32` status is artifact-backed and unambiguous
4. `T-33` has moved from planned to actively evidenced
5. the project's tasklist, progress note, and governance posture no longer contradict each other

---

## 7. One-Line Summary

Today's work is to turn M10 from "apparently recovered" into "auditably validated and decision-ready."
