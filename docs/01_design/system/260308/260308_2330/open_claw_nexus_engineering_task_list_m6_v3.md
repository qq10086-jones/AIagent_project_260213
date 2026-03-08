# OpenClaw Nexus
## Engineering Task List - Milestone 6 (v3)
## Date: 2026-03-08
## Focus: Production Parallel Rollout Readiness + Staging Validation
## Supersedes: open_claw_nexus_engineering_task_list_m6_v_2.md

---

## Changelog from v2

| Section | Change |
|---------|--------|
| Section 1.1 | Added quantitative go/no-go thresholds to success metrics |
| Section 4 / WS-23-01 | Added minimum replay corpus size and per-category coverage floor |
| Section 4 / WS-24.5-02 | Added runtime FE-safe guard requirement (structural impossibility detection) |
| Section 4 / WS-24-04 | Added negative test: intentionally misconfigured FE-safe workflow must be caught at runtime |
| Section 4 / WS-25-01 | Added single-command exposure state query requirement for on-call operability |
| Section 4 / WS-25-02.5 | Added pre-defined quantitative decision thresholds for go/no-go |
| Section 4 / WS-25-05 (NEW) | Added automated circuit-breaker for limited production exposure |
| Section 5 | Adjusted execution order to include new task and added per-phase time budget guidance |
| Section 8 | Added R-13 automated circuit-breaker risk |
| Section 9 | Added circuit-breaker rollback scenario |

---

## 0. Current System State Summary

Milestone 5 is closed and provides the baseline for M6.

### Entry governance condition

This task list is an `M6 candidate execution list` until the corresponding M6 design update is approved.

Before implementation begins, the following must exist and be approved:

- `OpenClaw Nexus Design Document v3.2` or an explicit M6 design addendum
- governance confirmation that M6 is an approved rollout-readiness milestone
- confirmation that M5 close-state remains the production baseline until an explicit M6 gate opens exposure

### Implemented / established by M5

1. `Structured patch / diff-first execution` is available under governance
- structured patch bundle schema and validation exist
- BE / FE steps support diff-first execution with fallback path
- diff-first is feature-gated and can be disabled without rolling back code

2. `Context budget observability` is available
- per-step context budget reporting exists
- overflow / warning conditions are measurable rather than inferred
- release artifacts can expose budget-related execution state

3. `Workflow DAG primitive and parallel readiness` are available
- DAG metadata and scheduling primitive exist
- BE / FE parallel feasibility can be evaluated
- at least synthetic / gated readiness has been demonstrated

4. `Governance position remains conservative`
- production `coding_team_v0` remains intentionally sequential by default
- parallel capability is treated as `ready + gated`, not broadly opened to real users
- milestone closure governance requires a newly approved milestone before additional production rollout work begins

### Why M6 exists

M5 completed the capability loop, but not the production exposure loop.

The core remaining gap is no longer "can the system support controlled parallel execution?" but rather:

- under real Discord-originated workflows, is the system stable enough?
- are rollout gates, rollback gates, and user-visible failure behaviors defined?
- do we have enough baseline production-like evidence to justify limited exposure?

Therefore M6 is not a pure implementation milestone. It is a production-readiness and rollout-governance milestone.

---

## 1. Objective

Milestone 6 prepares `coding_team_v0` for controlled real-world production exposure of parallel execution, without violating closure governance and without assuming default-open rollout.

M6 is explicitly split into two governed phases:

1. `M6-A — Staging / canary validation with real traffic patterns`
2. `M6-B — Gated production exposure for approved workflow classes only`

This milestone is about admission control, evidence collection, exposure policy, and rollback safety.

### 1.1 Success Metrics

M6 success is measured against rollout-governance outcomes, not just code completion.

| Metric | Pre-M6 State | M6 Target | Measurement Method | Go/No-Go Threshold |
|--------|--------------|-----------|-------------------|---------------------|
| Real Discord workflow replay coverage | none / ad hoc | approved replay set exists and passes staging pipeline | replay manifest + staging run report | minimum 50 cases total; minimum 7 per workflow category |
| Step-level context budget baseline | synthetic / limited | baseline distributions collected for PM / Arch / BE / FE / QA | `metrics/baseline_context_budget.json` | p90 values documented; overflow-risk tails explicitly flagged |
| Diff-first hit rate visibility | partially known | diff-first hit / fallback rate measurable on real replay corpus | `metrics/diff_first_baseline.json` | hit rate ≥ 60% on clean-repo cases to proceed with exposure |
| Patch anchor mismatch observability | synthetic only | mismatch rate measurable and thresholded | `metrics/patch_reliability.json` | mismatch rate ≤ 15% on replay corpus to proceed with exposure |
| FE-safe parallel eligibility visibility | limited | eligibility hit rate measurable by workflow type | `metrics/parallel_eligibility.json` | at least one workflow type qualifies with ≥ 80% eligibility rate |
| Parallel vs sequential outcome comparison | not established | side-by-side staging comparison exists for approved replay set | `metrics/parallel_vs_sequential.json` | parallel success rate within 5% of sequential baseline |
| Rollback gate operability | unproven | one-command / one-config rollback validated in staging and canary prod | rollback drill log | rollback completes in < 30 seconds without code deploy |
| Limited production exposure | not enabled | whitelist-only production rollout completed under gate | gated production report | partial failure rate ≤ 10%; zero uncontrolled scope expansion events |

**Threshold governance:** These thresholds are pre-committed baselines for the go/no-go decision. If any threshold is not met, the default decision is `STAY_GATED` or `DEFER_EXPOSURE`. Thresholds may be revised only through a documented governance review before the go/no-go meeting, not during it.

### 1.2 User Value Summary

Although M6 is governance-focused, it directly improves user trust and production safety:

- **Real-input staging** validates the system against the actual Discord prompt distribution instead of idealized prompts.
- **Gated exposure** prevents premature broad rollout of parallel execution before completion contracts and failure handling are stable.
- **Measured rollback readiness** reduces operational risk when new execution paths are exposed to real users.
- **Baseline data collection** creates the factual foundation required for later decisions such as broader rollout or adaptive routing.

---

## 2. Embedded Design Decisions

### D0 — M6 requires an approved design delta before task execution

Because M6 introduces new governed objects that were not fully specified inside the closed M5 implementation scope, no M6 task should be implemented until the matching design delta is approved.

That design delta must define at minimum:

- production rollout state model
- replay corpus data governance
- FE-safe completion contract boundaries
- failure and rollback semantics for limited production exposure

### D1 — M6 is a rollout-governance milestone, not a capability-expansion milestone

M6 does not define success as "more parallel features shipped."
It defines success as "parallel execution becomes governable, observable, and safely exposable for limited approved cases."

### D2 — Real Discord-originated workflow replay is mandatory

Synthetic canaries remain useful, but M6 requires replay of real human-language Discord input patterns in staging, because the production boundary condition is the natural-language entrypoint.

### D3 — Production remains sequential by default unless an explicit gate approves otherwise

Even after M6, production parallel execution is never assumed globally. Exposure is controlled by whitelist, project type eligibility, and runtime gate policy.

### D4 — Completion contract and failure-handling contract must be finalized before exposure

Parallel dispatch may not be production-exposed unless:
- completion contract is explicit
- artifact merge / release behavior is explicit
- one-sided failure behavior is explicit
- rollback path is explicit

### D5 — M6 must produce baseline evidence for future milestones

M6 baseline metrics are not optional reporting overhead. They are prerequisites for later roadmap items, including wider rollout decisions and any future adaptive routing work.

### D6 — Deferred roadmap items remain deferred

M6 does not introduce:
- Brain Router LLM classification
- adaptive model routing
- vector / semantic memory expansion
- new agent teams or new product domains
- dashboard-first productization unrelated to rollout governance

### D7 — Runtime structural guards take precedence over policy correctness (NEW)

The system must not rely solely on the correctness of policy configuration to prevent structurally invalid execution paths. Where a structural invariant can be verified at runtime (e.g., FE-safe completion feasibility), the runtime must enforce it independently of what the policy file declares.

---

## 3. Workstream Overview

| ID | Name | Type | Blocks |
|----|------|------|--------|
| WS-23 | Real Workflow Replay & Staging Validation | A | WS-24, WS-25 |
| WS-24 | Parallel Exposure Contract Finalization | A | WS-24.5, WS-25 |
| WS-24.5 | Production Parallel Enablement Wiring | A | WS-25 |
| WS-25 | Gated Production Exposure & Rollback Governance | A | E2E closure |
| WS-26 | Baseline Metrics & Approval Artifacts | A | E2E closure |

All workstreams are Type A because they directly govern production admission of the Coding Team pipeline.

---

## 4. Detailed Task List

---

## WS-23 Real Workflow Replay & Staging Validation

**Type:** Type A / Critical Path  
**Pipeline node:** Discord entrypoint → Brain Router → Coding Team end-to-end flow

---

### WS-23-01 Define Replay Corpus Contract

**Deliverables:**
- `orchestrator/contracts/workflow_replay_manifest.schema.json`
- `orchestrator/replay/manifests/m6_staging_replay_manifest.json`
- sanitized replay fixtures under `orchestrator/replay/fixtures/`

**Required fields:**
- `replay_id`
- `source_channel_type`
- `input_class`
- `raw_prompt_ref`
- `sanitized_prompt`
- `expected_route`
- `workflow_type`
- `project_type`
- `fe_parallel_eligible_expected`
- `notes`

**Requirements:**
- replay corpus must include real Discord-originated prompt patterns, sanitized if needed
- corpus must cover at least: PM-heavy, Architect-heavy, BE-led, FE-led, QA-heavy, mixed ambiguous prompts
- corpus must include both FE-safe and non-FE-safe cases
- corpus must include at least one dirty-repo / complex-file case where diff-first is likely stressed

**Minimum corpus size and coverage floor:**
- total replay corpus: minimum 50 cases
- each workflow category (PM-heavy, Architect-heavy, BE-led, FE-led, QA-heavy, mixed ambiguous): minimum 7 cases
- FE-safe cases: minimum 10 cases
- non-FE-safe cases: minimum 10 cases
- dirty-repo / complex-file cases: minimum 5 cases

**Acceptance criteria:**
- schema validates replay manifest
- minimum corpus size and per-category coverage floor are met
- every replay item has an explicit expected route and workflow type
- corpus distribution is documented in a coverage summary file alongside the manifest

---

### WS-23-01.5 Define Replay Data Governance

**Deliverables:**
- `docs/governance/replay_data_governance_m6.md`
- sanitization rules reference under `orchestrator/replay/README.md` or equivalent

**Requirements:**
- define whether raw Discord prompts may be stored, and under what boundary
- define mandatory sanitization for:
  - user identifiers
  - channel identifiers
  - links / attachments
  - secrets / environment hints
  - repository-specific sensitive paths if present
- define who is allowed to generate and review replay fixtures
- define retention and redaction rules for:
  - replay fixtures
  - staging artifacts
  - approval / closure packages

**Acceptance criteria:**
- replay corpus governance is explicit enough for repeatable staging use
- approval artifacts do not depend on unsanitized raw prompts
- fixture generation and retention rules are reviewable and enforceable

---

### WS-23-02 Build Staging Replay Runner

**Deliverable:** `orchestrator/scripts/run_m6_staging_replay.js`

**Requirements:**
- execute replay corpus end-to-end against staging configuration
- record per-run route, step outcomes, execution mode, fallback events, context budget, and final workflow state
- support sequential and gated-parallel execution modes for comparison
- ensure artifacts are isolated per replay case

**Acceptance criteria:**
- replay runner exits 0 on healthy staging suite
- outputs written under `orchestrator/artifacts/m6_staging_replay/`
- each replay case produces a structured result bundle

---

### WS-23-03 Add Parallel vs Sequential Comparison Harness

**Deliverables:**
- comparison mode inside replay runner
- `metrics/parallel_vs_sequential.json`

**Requirements:**
- for approved replay cases, run both sequential and gated-parallel paths
- compare:
  - workflow success / failure
  - partial failure incidence
  - execution duration
  - diff-first fallback frequency
  - patch anchor mismatch incidence
  - release artifact validity
- comparison must never silently discard failed branches

**Acceptance criteria:**
- side-by-side comparison exists for each approved comparison case
- metrics file is machine-readable and approval-ready
- parallel success rate variance from sequential baseline is explicitly reported

---

### WS-23-04 Staging Validation Canary

**Deliverable:** `orchestrator/scripts/canary_m6_staging.js`

**Coverage:**
- replay manifest loads successfully
- at least one sequential case passes end-to-end
- at least one FE-safe case passes gated-parallel path
- at least one non-FE-safe case is forced back to sequential path
- artifacts and metrics are emitted correctly

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/m6_staging/`

---

## WS-24 Parallel Exposure Contract Finalization

**Type:** Type A / Critical Path  
**Pipeline node:** Workflow engine + release behavior + user-visible outcome handling

---

### WS-24-01 Finalize FE-safe Completion Contract

**Deliverables:**
- `docs/contracts/fe_safe_completion_contract.md`
- workflow contract updates if needed

**Requirements:**
- define when BE and FE may be considered independently complete
- define conditions under which QA may start after parallel execution
- define required artifact merge order after both branches succeed
- define what "completion" means for partial-output states

**Acceptance criteria:**
- contract is explicit, versioned, and reviewable
- QA start conditions are testable
- no ambiguity remains around merge readiness

---

### WS-24-02 Finalize Failure-Handling Contract

**Deliverables:**
- `docs/contracts/parallel_failure_handling_contract.md`
- workflow state / release policy updates

**Requirements:**
- define user-visible behavior for:
  - BE success + FE failure
  - BE failure + FE success
  - branch timeout
  - patch failure after diff-first attempt
  - rollback-triggering incident
- define retry policy for failed branch only
- define whether partial artifacts are retained, quarantined, or discarded

**Acceptance criteria:**
- each failure mode has explicit handling rule
- branch-specific retry policy is documented and testable
- partial artifacts are not silently merged into release output

---

### WS-24-03 Finalize Exposure Eligibility Policy

**Deliverables:**
- `orchestrator/configs/parallel_exposure_policy.json`
- policy schema if needed

**Requirements:**
- whitelist workflow types eligible for production gated exposure
- whitelist project types eligible for FE-safe parallel mode
- define deny conditions:
  - ambiguous routing
  - high context overflow risk
  - dirty repo / anchor instability above threshold
  - unapproved workflow class
- policy must be runtime-configurable, not hardcoded

**Acceptance criteria:**
- policy is machine-readable and test-covered
- both allow and deny paths are observable
- changing whitelist does not require service logic edits

---

## WS-24.5 Production Parallel Enablement Wiring

**Type:** Type A / Critical Path  
**Pipeline node:** Runtime gate -> workflow dispatch -> step validation -> QA admission

This workstream exists because the M5 production baseline deliberately keeps `coding_team_v0` sequential.

M6 may not assume exposure merely from policy documents. It must explicitly bridge the current runtime from `production sequential lock` to `gated parallel enablement`.

---

### WS-24.5-01 Replace Production Sequential Lock with Policy-Driven Gate

**Deliverables:**
- updates to runtime gate logic
- test coverage proving deny-by-default behavior remains intact

**Requirements:**
- current production lock must remain the default until M6 rollout policy explicitly allows exposure
- runtime gate must read approved machine-readable policy rather than hardcoded production lock
- deny reason must stay observable and typed

**Acceptance criteria:**
- with no approved rollout policy, production remains sequential
- with approved but non-matching policy, production remains sequential
- with approved matching policy, workflow becomes eligible for gated parallel path

---

### WS-24.5-02 Finalize FE Validation Path for Parallel Completion

**Deliverables:**
- validator / handoff contract updates if needed
- test coverage for FE-safe completion without invalid completion assumptions
- **runtime structural guard that detects completion impossibility independently of policy configuration**

**Requirements:**
- explicitly define whether FE may complete without `be_to_fe` handoff in approved FE-safe cases
- if FE-safe completion still requires upstream BE material, parallel exposure must be denied
- validation rules must exactly match the approved completion contract
- **the runtime must include a structural feasibility check that evaluates whether completion is actually possible for the dispatched workflow, regardless of what the eligibility policy declares; if the structural check detects that completion requires material that is not available under the current execution path, the workflow must be forced back to sequential with a typed denial reason**

**Acceptance criteria:**
- no production path exists where dispatch is allowed but completion is structurally impossible
- FE validation semantics match documented FE-safe contract
- **runtime structural guard is verified by a dedicated negative test (see WS-24-04)**

---

### WS-24.5-03 Finalize QA Admission and Release Gating for Parallel Branches

**Deliverables:**
- workflow engine / validator updates if needed
- integration coverage for QA admission and release blocking

**Requirements:**
- QA start conditions after parallel branch execution must be deterministic
- release pack generation must not start from partial branch success
- branch merge order and artifact visibility must follow the completion contract

**Acceptance criteria:**
- QA cannot start from ambiguous or partial merge state
- release gating is deterministic and test-covered

---

### WS-24-04 Contract Validation Integration Tests

**Deliverable:** integration coverage for completion / failure / exposure contracts

**Coverage:**
- FE-safe case proceeds under parallel path
- non-FE-safe case is denied and remains sequential
- branch failure prevents improper QA / release progression
- rollback-trigger condition is emitted and logged
- runtime gate path matches actual validator and QA admission behavior
- **NEGATIVE TEST (NEW): intentionally misconfigured FE-safe workflow — a workflow class that requires `be_to_fe` handoff is marked as FE-safe in the eligibility policy; the runtime structural guard must detect completion impossibility and force the workflow back to sequential, emitting a typed denial reason such as `structural_completion_impossible`; this test verifies that the system does not rely solely on policy correctness to prevent invalid dispatch**

**Acceptance criteria:**
- integration suite passes
- logs and state transitions match contract definitions
- **negative misconfiguration test passes and produces the expected structural denial**

---

## WS-25 Gated Production Exposure & Rollback Governance

**Type:** Type A / Critical Path  
**Pipeline node:** Production admission control

---

### WS-25-01 Define Production Gate and Rollback Switches

**Deliverables:**
- `orchestrator/configs/production_parallel_rollout.json`
- updated runtime gate loading logic if needed
- **`orchestrator/scripts/exposure_state_query.js` — single-command exposure state diagnostic tool**

**Required controls:**
- global enable / disable
- whitelist by workflow type
- whitelist by project type
- canary cohort / limited exposure ratio if supported
- force-sequential override
- emergency rollback switch

**Required operational tooling:**
- a single CLI command or script that outputs:
  - current rollout master state (enabled / disabled / force-sequential)
  - current eligibility policy summary (whitelisted workflow types, whitelisted project types, active deny conditions)
  - effective exposure decision distribution for the last N runs (configurable, default 100)
  - timestamp of last policy change
- this tool must be usable by on-call engineers within 30 seconds of receiving an alert, without requiring them to manually read and cross-reference two JSON files

**Acceptance criteria:**
- all controls are runtime-editable
- production can revert to sequential without code rollback
- control state is logged at run start
- control state and effective eligibility decision are queryable after the run starts
- **exposure state diagnostic tool returns accurate results and is documented in the rollback runbook**

---

### WS-25-02 Run Pre-Exposure Rollback Drill

**Deliverables:**
- rollback drill log
- operator checklist under `docs/runbooks/m6_parallel_rollback_runbook.md`

**Requirements:**
- simulate gated exposure enabled
- trigger rollback to sequential-only mode
- verify in-flight and subsequent runs behave according to policy
- verify operators can identify current gate state quickly
- **drill must include a timed measurement: time from "alert received" to "rollback confirmed effective" must be recorded and must be under 30 seconds for pass**

**Acceptance criteria:**
- rollback completes without code deploy
- **rollback confirmed effective in under 30 seconds**
- runbook is complete enough for on-call usage
- drill result is attached to approval package

---

### WS-25-02.5 Exposure Go / No-Go Approval

**Deliverables:**
- explicit exposure approval record under `docs/governance/m6_exposure_go_no_go.md`
- approved decision snapshot referenced by rollout artifacts

**Requirements:**
- review staging replay evidence, comparison metrics, rollback drill result, and open risks before any limited production exposure begins
- produce one explicit decision:
  - `GO_LIMITED_EXPOSURE`
  - `STAY_GATED`
  - `ROLLBACK_TO_SEQUENTIAL_ONLY`
  - `DEFER_EXPOSURE`
- decision must name:
  - approved workflow whitelist
  - approved project-type whitelist
  - reviewer sign-off boundary
  - rollback trigger threshold if applicable

**Pre-defined quantitative decision criteria:**

The following thresholds are pre-committed and must be evaluated against replay-derived and staging evidence. They may only be revised through a documented governance review before the go/no-go meeting.

| Criterion | GO_LIMITED_EXPOSURE requires | STAY_GATED if | ROLLBACK_TO_SEQUENTIAL_ONLY if |
|-----------|------------------------------|---------------|--------------------------------|
| Replay corpus size | ≥ 50 cases, coverage floor met | corpus exists but floor not met | corpus does not exist or is trivial |
| Parallel vs sequential success rate | parallel within 5% of sequential baseline | delta between 5%–15% | delta > 15% |
| Partial failure rate (parallel path) | ≤ 10% | between 10%–25% | > 25% |
| Diff-first hit rate (clean-repo) | ≥ 60% | between 40%–60% | < 40% |
| Patch anchor mismatch rate | ≤ 15% | between 15%–30% | > 30% |
| Rollback drill time | < 30 seconds | 30–60 seconds | > 60 seconds or drill not completed |
| FE-safe eligibility qualification | ≥ 1 workflow type at ≥ 80% eligibility | eligibility exists but below 80% | no workflow type qualifies |
| Structural guard negative test | passes | — | fails or not executed |

**DEFER_EXPOSURE** is the appropriate decision when evidence is insufficient to evaluate thresholds (e.g., replay runner did not complete, metrics files are missing or malformed).

**Acceptance criteria:**
- limited production exposure does not begin without a recorded go/no-go decision
- approval record references the pre-defined thresholds and states the measured value for each
- approval record is reviewable and referenced by the production exposure report

---

### WS-25-03 Execute Limited Production Exposure

**Deliverables:**
- gated production exposure report
- production metrics snapshot

**Requirements:**
- enable parallel path only for approved whitelist scope
- keep exposure limited to approved workflow / project types
- use the recorded `GO_LIMITED_EXPOSURE` approval as the only valid production exposure entry gate
- monitor:
  - success rate
  - partial failure rate
  - fallback rate
  - rollback trigger events
  - user-visible incident count
  - circuit-breaker activation events (see WS-25-05)
- exposure must be reversible immediately

**Acceptance criteria:**
- limited exposure completes with no uncontrolled scope expansion
- production report exists and references gate settings used
- any triggered rollback (manual or automated) is documented rather than hidden

---

### WS-25-04 Production Exposure Canary

**Deliverable:** `orchestrator/scripts/canary_m6_rollout_gate.js`

**Coverage:**
- denied workflow remains sequential
- approved FE-safe workflow can enter gated-parallel path
- emergency rollback switch forces subsequent runs to sequential
- gate state is logged and queryable
- circuit-breaker triggers force-sequential when threshold is breached (see WS-25-05)

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/m6_rollout_gate/`

---

### WS-25-05 Automated Circuit-Breaker for Production Exposure (NEW)

**Deliverables:**
- circuit-breaker logic integrated into runtime gate evaluation
- circuit-breaker configuration in `orchestrator/configs/production_parallel_rollout.json`
- test coverage for circuit-breaker activation and recovery

**Requirements:**
- during limited production exposure, the runtime must monitor a rolling window of recent gated-parallel runs
- if the partial failure rate or rollback-trigger event rate within the rolling window exceeds a configurable threshold, the system must automatically activate force-sequential mode
- circuit-breaker activation must:
  - be logged with timestamp, trigger metric, threshold, and observed value
  - emit an operator-visible alert
  - persist the force-sequential state until an operator explicitly resets it
- circuit-breaker must not auto-recover; manual reset is required to prevent oscillation
- circuit-breaker configuration must include:
  - rolling window size (number of recent runs)
  - partial failure rate threshold (default: same as go/no-go ROLLBACK threshold, i.e., 25%)
  - rollback-trigger event threshold
  - alert destination configuration

**Acceptance criteria:**
- circuit-breaker activates correctly when threshold is breached in test
- circuit-breaker does not auto-recover
- operator receives alert on activation
- circuit-breaker state is visible in exposure state diagnostic tool (WS-25-01)
- canary test (WS-25-04) covers circuit-breaker activation path

---

## WS-26 Baseline Metrics & Approval Artifacts

**Type:** Type A / Protective  
**Pipeline node:** Cross-cutting governance and milestone closure evidence

---

### WS-26-01 Generate Context Budget Baseline

**Deliverables:**
- `metrics/baseline_context_budget.json`
- approval-ready summary in release / milestone artifact pack

**Requirements:**
- aggregate PM / Arch / BE / FE / QA distributions from replay corpus
- record p50 / p90 / max values where practical
- identify overflow-risk tails and affected workflow classes

**Acceptance criteria:**
- baseline is derived from replay results, not hand-waved estimates
- high-risk tails are explicitly called out

---

### WS-26-02 Generate Diff-first Reliability Baseline

**Deliverables:**
- `metrics/diff_first_baseline.json`
- `metrics/patch_reliability.json`

**Requirements:**
- measure diff-first hit rate
- measure fallback rate
- measure patch anchor mismatch rate
- separate clean-repo vs dirty-repo / complex-file cases if feasible

**Acceptance criteria:**
- reliability metrics are tied to replay corpus and staging results
- mismatch rate is thresholdable for rollout decisions

---

### WS-26-03 Generate Parallel Eligibility Baseline

**Deliverables:**
- `metrics/parallel_eligibility.json`

**Requirements:**
- measure how often real replay cases qualify for FE-safe parallel path
- break down by workflow type and project type
- identify over-conservative vs over-permissive gating patterns

**Acceptance criteria:**
- eligibility rate is measurable rather than anecdotal
- denial reasons are categorized and inspectable

---

### WS-26-04 Produce M6 Approval / Closure Package

**Deliverables:**
- `docs/governance/m6_approval_note.md`
- `docs/governance/m6_closure_note.md` (template prepared during M6; finalized at close)

**Requirements:**
- approval note must summarize:
  - scope
  - replay corpus coverage and distribution summary
  - replay data governance and sanitization boundary
  - baseline metrics with measured values against pre-defined thresholds
  - exposure policy
  - rollback readiness (including drill time measurement)
  - circuit-breaker configuration and test evidence
  - open risks
- closure note template must include a clear decision section:
  - stay gated
  - expand exposure
  - rollback to sequential-only
  - defer broader rollout to future milestone

**Acceptance criteria:**
- approval and closure materials are review-ready
- every go/no-go threshold from WS-25-02.5 has a corresponding measured value in the approval note
- milestone outcome can be decided from evidence, not narrative optimism

---

## 5. Suggested Execution Order

### Per-phase time budget guidance

To prevent governance documentation from consuming disproportionate milestone time, each phase has an indicative time budget. If a phase exceeds its budget by more than 50%, this should be treated as a scope signal requiring review, not a quality signal requiring more time.

| Phase | Indicative Budget | Hard Escalation Trigger |
|-------|-------------------|------------------------|
| Phase 0 — Approval entry gate | 3 days | 5 days |
| Phase 1 — Replay and contracts | 2 weeks | 3 weeks |
| Phase 2 — Runtime bridge | 2 weeks | 3 weeks |
| Phase 3 — Staging execution and comparison | 1.5 weeks | 2.5 weeks |
| Phase 4 — Rollout governance | 1.5 weeks | 2.5 weeks |
| Phase 5 — Evidence and limited exposure | 1 week | 2 weeks |

```text
Phase 0 — Approval entry gate
  approve M6 design delta / design addendum
  confirm M5 remains production baseline until explicit M6 gate opens

Phase 1 — Replay and contracts
  WS-23-01  Replay corpus contract (with minimum size and coverage floor)
  WS-23-01.5 Replay data governance
  WS-24-01  FE-safe completion contract
  WS-24-02  Failure-handling contract
  WS-24-03  Exposure eligibility policy

Phase 2 — Runtime bridge
  WS-24.5-01 Replace production sequential lock with policy-driven gate
  WS-24.5-02 Finalize FE validation path for parallel completion (including structural guard)
  WS-24.5-03 Finalize QA admission and release gating
  WS-24-04  Contract validation integration tests (including negative misconfiguration test)

Phase 3 — Staging execution and comparison
  WS-23-02  Staging replay runner
  WS-23-03  Parallel vs sequential comparison harness
  WS-23-04  Staging validation canary

Phase 4 — Rollout governance
  WS-25-01  Production gate and rollback switches (including exposure state diagnostic tool)
  WS-25-05  Automated circuit-breaker for production exposure
  WS-25-02  Pre-exposure rollback drill (including timed measurement)
  WS-25-02.5 Exposure go / no-go approval (with pre-defined quantitative thresholds)
  WS-25-04  Production exposure canary (including circuit-breaker path)

Phase 5 — Evidence and limited exposure
  WS-26-01  Context budget baseline
  WS-26-02  Diff-first reliability baseline
  WS-26-03  Parallel eligibility baseline
  WS-25-03  Limited production exposure
  WS-26-04  Approval / closure package
```

Phase 0 must finish before any implementation begins.
Phases 1 and 2 must finish before any staging comparison is considered trustworthy.
No production exposure should begin until contracts, runtime bridge work, and replay-driven staging evidence all exist.
No limited production exposure should begin without a recorded M6 exposure go / no-go approval.

---

## 6. Definition of Done for Milestone 6

Milestone 6 is complete when:

- an approved M6 design delta exists before implementation began
- a governed replay corpus of real Discord-originated workflow patterns exists, meeting minimum size and per-category coverage floor
- replay data governance and sanitization rules are explicit and enforced
- staging replay runner executes replay cases end-to-end and emits structured artifacts
- sequential vs gated-parallel comparison data exists for approved replay cases
- FE-safe completion contract is finalized and versioned
- failure-handling contract is finalized and test-covered
- runtime production lock has been replaced by a policy-driven deny-by-default gate
- runtime structural guard prevents dispatch-without-completion independently of policy configuration
- no production path allows gated parallel dispatch while validator or QA admission still makes completion impossible
- runtime exposure policy exists and controls whitelist / deny behavior
- production rollout gates and emergency rollback switches are runtime-operable
- automated circuit-breaker is configured, tested, and active during limited exposure
- rollback drill has been executed, documented, and meets the 30-second time target
- an explicit exposure go / no-go decision exists before limited production exposure, with measured values for all pre-defined thresholds
- limited production exposure has been performed only within approved scope
- context budget baseline exists from replay-derived data
- diff-first fallback / reliability baseline exists from replay-derived data
- FE-safe parallel eligibility baseline exists from replay-derived data
- approval package is complete, references all threshold measurements, and is sufficient for milestone review
- production parallel execution remains explicitly gated rather than silently becoming default behavior

---

## 7. Non-Scope for Milestone 6

- no broad default-on rollout of parallel execution for all production workflows
- no Brain Router LLM classification
- no adaptive model routing
- no expansion into new agent teams or new product domains
- no vector memory or semantic retrieval expansion
- no dashboard-first work that is disconnected from rollout governance
- no assumption that M6 automatically implies M7 readiness

---

## 8. Risks

| Risk | Severity | M6 Disposition |
|------|----------|----------------|
| Team reframes M6 as more parallel feature-building instead of rollout governance | High | scope wording explicitly centers admission, metrics, rollback, and whitelist exposure |
| Synthetic canaries hide real Discord prompt distribution edge cases | High | replay corpus with real prompt patterns is mandatory, with minimum size and coverage floor |
| Replay artifacts accidentally retain sensitive Discord or user data | High | mandatory replay data governance, sanitization, and retention rules |
| FE-safe eligibility policy is too permissive | High | deny-by-default policy with whitelist and explicit thresholds; runtime structural guard as safety net |
| FE-safe eligibility policy is too conservative | Medium | collect denial reason baseline and tune from evidence |
| Production runtime still contains sequential-lock assumptions after policy approval | High | explicit runtime bridge workstream required before exposure |
| Patch anchor instability spikes on dirty repos | High | treat anchor mismatch rate as rollout-governance metric, not just implementation detail |
| Partial failures create confusing user-visible outcomes | High | finalize failure-handling contract before production exposure |
| Rollback exists in theory but is slow in practice | High | mandatory rollback drill with timed measurement and 30-second target |
| M6 produces evidence but no clear go / no-go decision mechanism | Medium | pre-defined quantitative thresholds committed before go/no-go meeting |
| Governance documentation work consumes disproportionate milestone time | Medium | per-phase time budgets with escalation triggers |
| On-call engineers cannot determine exposure state quickly during incidents | Medium | single-command exposure state diagnostic tool required |
| Limited production exposure has no automated safety net; relies entirely on human monitoring and manual rollback | High | automated circuit-breaker with rolling-window threshold, operator alert, and manual-reset-only recovery (R-13) |

---

## 9. Rollback Strategy

| Scenario | Rollback Action | Impact |
|----------|----------------|--------|
| Limited production exposure shows instability | Flip emergency rollback switch / force-sequential override | All subsequent runs revert to sequential-only mode |
| Eligibility policy is too broad | Tighten `parallel_exposure_policy.json` whitelist | Exposure narrows without code changes |
| Diff-first reliability degrades under real traffic | Disable diff-first feature gate while preserving M6 governance artifacts | Workflow remains operable with reduced optimization |
| Parallel path causes unacceptable incident rate | Disable production parallel rollout config entirely | Revert to M5-style governed sequential production |
| Circuit-breaker activates automatically | Force-sequential persists until operator explicitly resets; investigate trigger cause before re-enabling | Automatic protection; no human latency in initial response |

---

## 10. Next Step

This task list should be reviewed and approved before any M6 production exposure work begins.

Approval of M6 authorizes a controlled rollout-readiness program, not unconditional production-wide parallel execution.
