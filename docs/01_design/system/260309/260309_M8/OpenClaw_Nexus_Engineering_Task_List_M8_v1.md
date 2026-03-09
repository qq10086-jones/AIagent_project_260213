# OpenClaw Nexus — M8 Engineering Task List v1

- Version: 1.0
- Date: 2026-03-09
- Milestone: M8 — Staging Evidence and Live Routing Validation
- Status: DRAFT — Awaiting Architect Approval
- Depends on: M7 CLOSED WITH DEVIATION (`docs/governance/m7_closure_note.md`)

---

## Entry Criteria

Before any M8 task begins:

| Criterion | Status |
|-----------|--------|
| M7 formally closed | ✅ DONE — `m7_closure_note.md` v1.1 |
| M8 Engineering Task List approved by Architect | ⏸ This document |
| M8 Design Delta (if architecture changes) | TBD — Phase 0 is debt-only, no new architecture |
| P1 pre-requisite tasks from PROGRESS_LATEST.md acknowledged | ✅ Listed as Phase 0 |

---

## M8 Scope Summary

M8 delivers the evidence required to validate M7 dynamic routing infrastructure in production-like conditions, resolve the technical debt inherited from M7, and establish the foundation for cohort expansion.

**M8 is NOT a new feature milestone.** It is a validation and stabilization milestone.

### What M8 includes

1. **Phase 0 — Technical Debt (M7 Carryover):** Brain test infrastructure + workflow_engine decomposition. No new features; no architecture changes. Must complete before Phase 1.
2. **Phase 1 — Live Trial Execution:** Isolated staging environment with real DB + service + LLM calls. Executes the live dynamic routing trial deferred from M7 WS-31-02.
3. **Phase 2 — Evidence Review:** Populate routing_decision_log and waterfall_stage_log with real data. Generate live routing evaluation report. Compare vs M6 baseline.
4. **Phase 3 — Closure and Decisions:** M6 STAY_GATED → GO_LIMITED_EXPOSURE decision (independent). M8 Go/No-Go package. Optional: cohort expansion readiness decision.

### What M8 excludes

- New routing features or classifier model changes
- Brain API boundary decoupling (tracked as BLOCK-02 risk; architectural plan only)
- Cohort expansion beyond `fe_led` (gated on Phase 2 evidence showing fe_led stability)
- Classifier `model_tier` acting on model selection (currently advisory-only; remains advisory)

---

## Governance Constraints

| Rule | Source |
|------|--------|
| No new features in Phase 0 — debt cleanup only | M7 deviation rationale, PROGRESS_LATEST.md |
| M6 activation decision must not be conflated with M7 master_enabled | `m7_go_no_go.md` § 9 deviation rationale |
| `workflow_engine.js` must stay below 600-line governance budget | `OpenClaw_Execution_Governance_Scope_Control_v3.md` Section 12 |
| Target for decomposition: below 520 lines | PROGRESS_LATEST.md BLOCK-03 |
| Any brain/ classifier changes require pytest coverage first | PROGRESS_LATEST.md BLOCK-02 |
| Live trial requires isolated staging DB + service | `m7_go_no_go.md` § 9 architectural rationale |

---

## Phase 0 — Technical Debt (Pre-Phase-1 Gate)

**Objective:** Resolve M7 carryover technical debt. These tasks are BLOCKERS for Phase 1.

### WS-33 — M7 Technical Debt Resolution

#### WS-33-01: Brain Directory Test Infrastructure

**Owner:** Engineer
**Depends on:** Nothing
**Blocks:** WS-33-02 (partially), WS-34-01

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| WS-33-01-01: pytest setup | Create `brain/pytest.ini`, `brain/conftest.py`, `brain/tests/__init__.py` | `pytest brain/tests/` runs with 0 errors |
| WS-33-01-02: Mock infrastructure | `conftest.py`: fixtures for mocked psycopg2 connection, mocked LLM (no API calls), mocked HTTP requests | All test isolation verified — no real DB or network calls |
| WS-33-01-03: Classifier pipeline tests | `brain/tests/test_supervisor_routing.py`: happy path (correct routing per mode), unavailability path (DB error in poll_for_fact degrades gracefully), low-confidence path (missing facts → writer fallback) | Minimum 6 tests; 100% pass |

**Key files:**
- `brain/pytest.ini`
- `brain/conftest.py`
- `brain/tests/__init__.py`
- `brain/tests/test_supervisor_routing.py`

**Test scope (minimum):**
1. `supervisor_node` coding mode, no facts → routes to "coder"
2. `supervisor_node` coding mode, coder fact present → routes to "writer"
3. `supervisor_node` discovery mode, no facts → routes to "discovery"
4. `supervisor_node` analysis mode, no facts → routes to "quant"
5. `poll_for_fact` with DB exception → returns None (no crash)
6. `get_llm` with no API key set → returns None (no crash)

---

#### WS-33-02: `workflow_engine.js` Decomposition

**Owner:** Engineer
**Depends on:** Nothing
**Target:** Bring `workflow_engine.js` from 577 lines to below 520 lines

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| WS-33-02-01: Extract `workflow_step_artifacts.js` | Move `buildContextBudgetArtifactPath`, `writeContextBudgetReport`, `applyStructuredPatchIfPresent` to `src/domain/workflow_step_artifacts.js` with factory pattern | workflow_engine.js line count reduced; all existing tests pass |
| WS-33-02-02: Extract `workflow_checkpoint.js` | Move `createCheckpoint` logic to `src/domain/workflow_checkpoint.js` with `createWorkflowCheckpointService({ pool })` factory | workflow_engine.js line count below 520; all existing tests pass |
| WS-33-02-03: Verify test suite | `node --test test/*.test.js` must pass 127/127 | Zero regressions |

**Key files:**
- `src/domain/workflow_step_artifacts.js` (NEW)
- `src/domain/workflow_checkpoint.js` (NEW)
- `src/workflow_engine.js` (MODIFIED — imports only, function bodies removed)

---

## Phase 1 — Live Trial Execution

**Objective:** Execute the live dynamic routing trial deferred from M7 WS-31-02. Requires isolated staging environment.

**Gate:** WS-33-01 and WS-33-02 must be COMPLETE before Phase 1 begins.

### WS-34 — Staging Environment and Live Trial

#### WS-34-01: Staging Environment Setup

**Owner:** Engineer / Ops
**Depends on:** WS-33-01 (brain test infra must exist)

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| WS-34-01-01: Staging DB | Provision isolated PostgreSQL instance (not shared with dev/prod) with schema migrated | `routing_decision_log` and `waterfall_stage_log` tables exist and are writable |
| WS-34-01-02: Service instance | Run orchestrator service instance pointing to staging DB with `master_enabled: true`, `dynamic_routing_enabled: true` | Service starts, health endpoint responds |
| WS-34-01-03: Cohort sign-off | Architect approves `cohort_enabled: true` in `configs/m7_exposure_cohorts.json` | Written approval recorded in Go/No-Go package |

#### WS-34-02: Live Dynamic Routing Trial

**Owner:** Engineer
**Depends on:** WS-34-01

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| WS-34-02-01: Execute trial | Run `scripts/run_m7_dynamic_routing_trial.js` against staging service with real LLM calls | Trial completes without fatal errors; `mode: live_trial` in result artifact |
| WS-34-02-02: Trial evidence bundle | Produce `artifacts/m8_trial/live_trial_result.json` | Contains routing_decision_log entries, classifier health snapshot, threshold evaluation |
| WS-34-02-03: Circuit-breaker validation | Confirm CB thresholds are not breached under normal load | `circuit_breaker_evaluation.breached: false` in trial result |

---

## Phase 2 — Evidence Review

**Objective:** Populate routing_decision_log and waterfall_stage_log from live runs; generate live routing evaluation report.

**Gate:** WS-34-02 must be COMPLETE.

### WS-35 — Live Routing Evidence

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| WS-35-01: Routing decision log review | Query `routing_decision_log`; verify all 8 normalized sources represented; check high-risk misroute rate | high_risk_misroute < 2% (live) |
| WS-35-02: Waterfall latency attribution | P50/P95 latency per stage from `waterfall_stage_log` | Values populated; routing stage < 50ms P95 |
| WS-35-03: Live routing evaluation report | Run `scripts/generate_routing_evaluation_report.js` with live data | Report includes live counterfactual comparison and latency delta vs M6 baseline |
| WS-35-04: Low-confidence fallback ratio | Measure live low-confidence fallback ratio from routing_decision_log | Within 10–40% target range |

---

## Phase 3 — Closure and Decisions

**Objective:** Make formal Go/No-Go decisions based on Phase 2 evidence.

**Gate:** WS-35-03 complete; WS-35-04 within bounds.

### WS-36 — M8 Closure

| Task | Description | Acceptance Criteria |
|------|-------------|---------------------|
| WS-36-01: M6 activation decision | Independent Architect review: M6 STAY_GATED → GO_LIMITED_EXPOSURE | Separate written decision; must not reference M7 `master_enabled` |
| WS-36-02: Cohort expansion readiness | Review fe_led live stability; decide whether to expand to be_fe_simple or similar | Written recommendation with evidence basis |
| WS-36-03: M8 Go/No-Go package | Document Phase 1+2 results; record any deviations; formal Architect approval | `docs/governance/m8_go_no_go.md` |
| WS-36-04: PROGRESS_LATEST.md | Update with M8 status | M8 status = CLOSED or STAY_GATED with rationale |

---

## Success Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| workflow_engine.js line count | < 520 lines | Phase 0 |
| brain/ pytest tests | ≥ 6 tests, 100% pass | Phase 0 |
| Total orchestrator test suite | 127/127 (no regression) | Phase 0 |
| Live trial executes without fatal errors | 1 run | Phase 1 |
| High-risk misroute rate (live) | < 2% | Phase 2 |
| Waterfall routing stage P95 | < 50ms | Phase 2 |
| Low-confidence fallback ratio (live) | 10–40% | Phase 2 |

---

## Risk Register (M8)

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R-NEW-01 | Brain test infra absent before M8 classifier changes | High | WS-33-01 is Phase 0 blocker |
| R-NEW-02 | workflow_engine.js budget breach on next feature add | Medium | WS-33-02 brings back to 512 lines |
| R-NEW-03 | Staging environment not isolated — shares data with dev | High | WS-34-01-01 explicitly provisions separate DB |
| R-NEW-04 | M6/M7 activation coupled via master_enabled | High | WS-36-01 explicitly separates M6 decision |
| R-NEW-05 | Low-confidence fallback ratio outside 10–40% in live conditions | Medium | WS-35-04 gates WS-36 decisions |

---

## Approval Record

| Role | Decision | Date |
|------|----------|------|
| PM | Submitted for Architect review | 2026-03-09 |
| Architect | Pending | — |
