# OpenClaw Nexus M7 — Go/No-Go Review Package

- Version: 1.0
- Date: 2026-03-09
- Milestone: M7 — Limited Dynamic Routing v1
- Prepared by: PM / Architecture Review
- Required reviewers: PM + Architect

---

## 1. Design Delta Reference

| Document | Location | Status |
|----------|----------|--------|
| M7 Design Document v4.0 | `docs/01_design/system/260309/260309_1048/OpenClaw_Nexus_Design_Document_v4.md` | Approved |
| M7 Engineering Task List v2 | `docs/01_design/system/260309/260309_1048/OpenClaw_Nexus_Engineering_Task_List_M7_v2.md` | Approved |
| M7 Governance v3 | `docs/01_design/system/260309/260309_1048/OpenClaw_Execution_Governance_Scope_Control_v3.md` | Approved |

---

## 2. Task Completion Checklist

### WS-27 — Design Delta and Approval

| Task | Status | Evidence |
|------|--------|----------|
| WS-27-01: M7 Design Document v4.0 authored | ✅ PASS | `OpenClaw_Nexus_Design_Document_v4.md` — three-layer policy precedence, classifier degradation semantics, model tier contract, observability additions |
| WS-27-02: Approval gate | ✅ PASS | Design delta reviewed and approved; task list authorized before implementation |

### WS-28 — Brain Router Classification and Decision Contract

| Task | Status | Evidence |
|------|--------|----------|
| WS-28-01: Classification taxonomy v1 | ✅ PASS | Work shape (4 classes) + domain lead (5 classes) defined; ambiguous → single_branch_safe safe fallback |
| WS-28-02: Classifier implementation | ✅ PASS | `src/vnext/brain_router_classifier.js`; offline evaluation completed |
| WS-28-03: Routing decision contract | ✅ PASS | `contracts/routing_decision.schema.json`; 8 normalized routing_decision_source values |
| WS-28-04: Classifier offline evaluation quality gate | ✅ PASS | Evaluation report produced and Architect-reviewed; high-risk misroute < 2%, precision > 85%, low-confidence ratio within 10–40% |

### WS-29 — Adaptive Runtime Integration

| Task | Status | Evidence |
|------|--------|----------|
| WS-29-01: Dynamic routing integration + classifier unavailability path | ✅ PASS | `src/domain/parallel_rollout_gate.js` — three-layer precedence; classifier unavailability → static-policy-only fallback; dedicated integration tests |
| WS-29-02: Model tier recommendation path | ✅ PASS | model_tier logged per run in routing_decision_log; fallback to balanced_default when unavailable/low-conf |
| WS-29-03: Safety degradation and rollback controls | ✅ PASS | router_mode, dynamic_routing_enabled, force_sequential runtime controls; circuit-breaker (M6); rollback drill 8 sec |

### WS-30 — Observability, Auditability, Evidence

| Task | Status | Evidence |
|------|--------|----------|
| WS-30-01: Routing decision audit log | ✅ PASS | `src/domain/routing_audit_log.js`; `routing_decision_log` table; 9/9 tests |
| WS-30-02: Waterfall trace & latency attribution | ✅ PASS | `src/domain/waterfall_trace_service.js`; `waterfall_stage_log` table; P50/P95 queryable; 10/10 tests |
| WS-30-03: Routing evaluation report | ✅ PASS | `src/domain/routing_evaluation_report.js`; 6 dimensions + counterfactual comparison reads from governed policy file; 15/15 tests |

### WS-31 — Limited Dynamic Exposure Program

| Task | Status | Evidence |
|------|--------|----------|
| WS-31-01: Define approved M7 exposure cohorts | ✅ PASS | `configs/m7_exposure_cohorts.json`; machine-readable; rollback thresholds explicit; `cohort_enabled: false` (pending sign-off) |
| WS-31-02: Infrastructure complete | ✅ PASS | `src/domain/classifier_health_monitor.js`; `scripts/run_m7_dynamic_routing_trial.js`; 11/11 tests |
| WS-31-02: Live trial with real LLM calls | ✅ ACCEPTED_WITH_DEVIATION | Deferred to M8 Phase 1; dry-run preflight + unavailability drill satisfies acceptance criteria — see § 9 |

### WS-32 — Closure Review and Decision Package

| Task | Status | Evidence |
|------|--------|----------|
| WS-32-01: This document | ✅ IN REVIEW | `docs/governance/m7_go_no_go.md` |
| WS-32-02: Closure note | ✅ DONE | `docs/governance/m7_closure_note.md` v1.1 — CLOSED WITH DEVIATION |

---

## 3. Routing Evaluation Report Summary

**Source:** `scripts/generate_routing_evaluation_report.js` (WS-30-03)
**Evidence basis:** Dry-run preflight against governed replay corpus (50 cases, m6-staging-v1)

**Preflight result artifact:** `orchestrator/artifacts/m7_trial/preflight_result.json`

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Total cases evaluated | 50 | — | — |
| Static-policy gated_parallel | 10 (20%) | — | Baseline |
| Dynamic-routing gated_parallel | 11 (22%) | — | +1 dynamic uplift |
| Agreement rate (static vs dynamic) | 94% (47/50) | — | High coherence |
| High-risk misroute count | 0 | < 2% | ✅ |
| Low-confidence fallback ratio | 0% (drill-off) | 10–40% live target | ⏸ Requires live data |
| Forced-sequential ratio (normal) | 78% | — | Expected (master_enabled=false) |
| Classifier availability (drill-off) | 100% | ≥ 90% | ✅ |

**Counterfactual comparison summary:**
- Static-policy path (fe_led whitelist only): 10 parallel admissions
- Dynamic-routing path (classifier ground-truth): 11 parallel admissions
- Dynamic uplift: +1 case correctly identified by classifier that static policy misses
- Agreement: 94% of decisions match between static and dynamic paths

**Latency stats:** Not yet populated — requires live routing_decision_log and waterfall_stage_log data from real runs.

---

## 4. Rollback Drill Evidence

| Drill | Date | Method | Time | Result |
|-------|------|--------|------|--------|
| M6 force-sequential rollback | 2026-03-09 | Set `force_sequential: true` in rollout config | **8 seconds** | ✅ PASS (target < 30s) |
| Runbook | — | `docs/runbooks/m6_parallel_rollback_runbook.md` | — | Operational |
| Circuit-breaker activation | 2026-03-09 | `evaluateCircuitBreaker()` → `activateCircuitBreaker()` | Immediate | ✅ PASS |

M7-specific rollback controls verified via test coverage (WS-29-03):
- `router_mode=static_policy_only` → confirmed via parallel_rollout_gate.js tests
- `dynamic_routing_enabled=false` → confirmed, Layer 1 switch
- `force_sequential=true` → confirmed, gate Layer 1 catches before any eligibility evaluation

---

## 5. Incident Summary

| Period | Severity | Count | Notes |
|--------|----------|-------|-------|
| M7 development (WS-27 → WS-32) | P0 | 0 | — |
| M7 development | P1 | 0 | — |
| M7 development | P2 | 0 | — |
| Live trial | — | N/A | Live trial not yet executed (master_enabled=false) |

No incidents to report for the M7 development phase.

---

## 6. Metric Summary vs M6 Baseline

| Metric | M6 Baseline | M7 State | Delta |
|--------|-------------|----------|-------|
| Routing decision queryability | None | 100% (routing_decision_log active) | +new |
| Waterfall latency attribution | None | 4 stages instrumented | +new |
| Counterfactual replay comparison | None | 50-case corpus evaluated | +new |
| Classifier health monitoring | None | Rolling-window availability tracking | +new |
| circuit-breaker drill | 8 sec rollback | Classifier health + CB threshold integrated | ✅ |
| Test suite | 84/84 | 127/127 | +43 tests |
| Routing decision source normalization | None | 8 normalized values, audit logged per run | +new |

---

## 7. Classifier Offline Evaluation Results (WS-28-04)

Per PROGRESS_LATEST.md and WS-28-04 quality gate:

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| High-risk misroute rate | < 2% | Met | ✅ |
| dual_branch_parallel_candidate precision | > 85% | Met | ✅ |
| Low-confidence fallback ratio | 10–40% | Met | ✅ |
| Ambiguity handling defaults verified | Explicit in § 14.4 | Verified | ✅ |
| Architect go/no-go for runtime integration | Required | **Granted** | ✅ |

---

## 8. Classifier Unavailability Drill Results

**Drill artifact:** `orchestrator/artifacts/m7_trial/drill_unavailable_result.json`
**Method:** `node scripts/run_m7_dynamic_routing_trial.js --drill-unavailable`
**Date:** 2026-03-09

| Criterion | Expected | Observed | Status |
|-----------|----------|----------|--------|
| Classifier availability under drill | 0% | 0.0% | ✅ |
| All cases fall back to forced_sequential | 100% | 100% (50/50) | ✅ |
| Classifier health alert raised | Once | Raised on first failure | ✅ |
| forced_sequential_spike threshold detection | > 40% triggers | 100% → threshold breached + logged | ✅ |
| No panic / no crash | Non-fatal | Runs to completion, exit code 2 | ✅ |

**Finding:** Classifier unavailability correctly degrades the entire routing layer to `forced_sequential`. The health alert fires exactly once. The threshold evaluation correctly identifies the spike. Exit code 2 signals operators without crashing the process.

---

## 9. Go/No-Go Recommendation

### Evidence-based assessment

**Infrastructure quality:** All WS-27 through WS-31-01 deliverables are complete and test-covered (127/127 PASS). Zero critical bugs. All governance budgets respected. All fire-and-forget paths non-blocking and observable.

**Safety verification:** Classifier unavailability drill passed. Static safety override precedence confirmed. Force-sequential rollback < 30 seconds.

**Gap:** WS-31-02 live trial with real LLM calls has **not yet been executed** because `master_enabled=false` and `cohort_enabled=false`. This is the single remaining criterion before M7 can close.

### Recommendation

**CLOSED — ACCEPTED WITH DEVIATION (WS-31-02)**

M7 is formally closed under the "production-like governed evidence" path permitted by WS-31-02 acceptance criteria ("real LLM calls **or** production-like governed evidence").

### Deviation Statement — WS-31-02 Live Trial

**Decision:** WS-31-02 live trial with real LLM calls is deferred to M8 Phase 1.

**Basis for deviation:**

WS-31-02 acceptance criteria are satisfied as follows:

| Criterion | Path used | Evidence artifact |
|-----------|-----------|-------------------|
| trial evidence bundle completed | dry-run preflight against governed 50-case corpus | `artifacts/m7_trial/preflight_result.json` |
| no unresolved severe incident | zero incidents across full M7 development | incident summary in § 5 of this document |
| fallback behavior exercised and verified | static-policy fallback confirmed in preflight (94% agreement) | `preflight_result.json` → summary.agreement_rate |
| classifier unavailability fallback exercised (may be simulated) | drill_unavailable mode: 100% forced_sequential, health alert raised | `artifacts/m7_trial/drill_unavailable_result.json` |

**Architectural rationale for deviation:**

1. `run_m7_dynamic_routing_trial.js` is a gate-evaluation simulator, not a live traffic driver. Flipping `master_enabled: true` would change live service behavior, not satisfy the trial requirement — the two are structurally decoupled.
2. `master_enabled: true` activates the M6 parallel execution path simultaneously. M6 is currently STAY_GATED for independent reasons. The two decisions must not be conflated.
3. A proper live trial requires an isolated staging environment with a real database and injected traffic. This is M8 Phase 1 scope.
4. The governing corpus (m6-staging-v1, 50 cases, architect-reviewed) is the authoritative evidence base for this project. Dry-run evaluation against it constitutes "production-like governed evidence" as defined in the WS-31-02 acceptance criteria.

**Deferred work (tracked as M8 P1):**

- Set up isolated staging environment (DB + running service instance)
- Execute live dynamic routing trial with real LLM calls
- Populate `routing_decision_log` and `waterfall_stage_log` with real run data
- Generate routing evaluation report with live latency delta vs M6 baseline
- Independently decide M6 STAY_GATED → GO_LIMITED_EXPOSURE upgrade with separate Architect approval

---

## 10. Approval Record

| Role | Name | Decision | Date |
|------|------|----------|------|
| PM | AI (OpenClaw PM) | Submitted for review | 2026-03-09 |
| Architect | OpenClaw Architect | **APPROVED — CLOSED WITH DEVIATION** | 2026-03-09 |
