# M7 Controlled Enablement Plan

- Version: 1.0
- Date: 2026-03-10
- Scope: M7 dynamic routing controlled enablement
- Prepared by: Architecture / QA
- Required reviewers: PM + Architect

---

## 1. Objective

This document defines the controlled enablement plan for `M7 dynamic routing` after the completion of:

- M7 implementation and closure
- M8 staging evidence and live validation
- 2026-03-09 compressed validation replacing long-window baseline observation

The goal is not to immediately open dynamic routing globally. The goal is to:

1. define an approved production enablement path
2. constrain blast radius
3. preserve explicit rollback authority
4. attach machine-readable thresholds to each enablement step

---

## 2. Current State

Status note:

- This document remains the original 2026-03-10 controlled enablement plan.
- Operational state later advanced beyond this snapshot.
- The authoritative transition into limited enforced mode is recorded in `docs/governance/2026-03-12_m10_phase_b_limited_enforced_signoff.md`.

Current production state:

- `master_enabled=true`
- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_advisory`

Current production conclusion:

- M6 static-policy limited exposure is active
- M7 dynamic routing infrastructure is implemented and validated
- M7 Phase A advisory-only package is prepared and partially validated
- final live activation confirmation is pending a controlled restart of the local orchestrator runtime so the updated advisory routing code is actually loaded

Current evidence base:

- M7 offline evaluation completed
- M8 staging live trial completed
- classifier unavailability drill completed
- live runtime validation completed
- 30-minute compressed validation completed
- Phase A advisory canary completed
- Phase A live runtime validation after enablement completed

Reference artifacts:

- `docs/governance/m7_go_no_go.md`
- `docs/governance/m8_go_no_go.md`
- `docs/governance/m6_accelerated_validation_go_no_go_2026-03-09.md`
- `orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json`
- `orchestrator/artifacts/m7_phase_a/phase_a_enablement_plan.json`
- `orchestrator/artifacts/canary/m7_phase_a_advisory/canary_m7_phase_a_advisory.json`
- `orchestrator/artifacts/m7_trial/preflight_result_20260310_phase_a_enabled.json`

---

## 3. Enablement Principles

M7 controlled enablement must obey these principles:

1. **No global switch-on**
   Dynamic routing may not be enabled for all eligible traffic at once.

2. **Cohort-first rollout**
   Enablement must be limited to an explicit approved cohort.

3. **Static-policy remains the safety baseline**
   Any classifier degradation, ambiguity, or threshold breach must degrade back to static-policy-only or forced-sequential.

4. **Rollback must be operator-simple**
   Rollback must be achievable by:
   - `dynamic_routing_enabled=false`
   - or `router_mode=static_policy_only`
   - or `force_sequential=true`

5. **Decision must remain evidence-driven**
   Every enablement step requires fresh evidence from:
   - `routing_decision_log`
   - `waterfall_stage_log`
   - workflow run status outcomes

---

## 4. Proposed Rollout Strategy

### Phase A. Advisory-Only Production Enablement

Purpose:

- enable classifier output logging in production
- keep execution behavior unchanged
- compare dynamic recommendations against current static outcome

Required config shape:

- `master_enabled=true`
- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_advisory`
- `cohort_enabled=true`

Execution effect:

- classifier runs
- audit/logging runs
- runtime still behaves conservatively
- no broad behavior change is allowed outside approved cohort

Recommended cohort:

- `workflow_type = coding_team_v0`
- `project_type = crm`
- `input_class = fe_led`

Success condition:

- no high-risk misroutes
- classifier availability stable
- no rollback trigger breach
- sufficient sample size to justify Phase B

### Phase B. Limited Execution Enablement

Purpose:

- allow dynamic routing to influence execution only for the approved cohort

Required config shape:

- `master_enabled=true`
- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_enforced`
- `cohort_enabled=true`

Execution effect:

- classifier recommendation may admit `gated_parallel_allowed`
- static-policy fallback remains mandatory
- high-risk and low-confidence deny conditions remain mandatory

Blast-radius control:

- only approved cohort
- no expansion to `be_fe_simple`
- no high-risk work shapes
- no ambiguous / unavailable classifier cases

### Phase C. Post-Enablement Expansion Review

Purpose:

- evaluate whether to expand cohort or keep scope fixed

This phase is not automatic.

It requires a separate review for:

- cohort expansion
- project-type expansion
- model-tier execution influence

---

## 5. Recommended Production Cohort

Recommended first production M7 cohort:

- `workflow_type`: `coding_team_v0`
- `project_type`: `crm` / `webapp_crm`
- `input_class`: `fe_led`

Explicitly excluded:

- `high_risk_release_sensitive`
- `architectural_orchestration_required`
- low-confidence classifications
- classifier unavailable cases
- non-approved project types

Rationale:

- this is the smallest cohort already aligned with existing M6 policy reality
- it is consistent with the historical M7 cohort intent
- it preserves the most conservative useful production slice

---

## 6. Enablement Gates

### Gate G1. Preconditions Before Phase A

All of the following must be true:

1. compressed validation evidence approved
2. live runtime validation passes
3. rollback controls validated
4. `routing_decision_log` and `waterfall_stage_log` writing confirmed
5. PM + Architect approval recorded

### Gate G2. Advisory-Only Exit Criteria

Minimum advisory evidence required:

- routing sample size `>= 60`
- classifier availability `>= 90%`
- high-risk misroute count `= 0`
- low-confidence or unavailable events degrade correctly
- no P0/P1 production incident

### Gate G3. Enforced Limited Enablement Entry Criteria

All of the following must be true:

- G2 complete
- `forced_sequential_ratio` does not show unexplained spike
- `execution_dispatch` and downstream workflow states remain observable
- advisory recommendation uplift is measurable and not risk-dominant
- Architect explicitly signs off Phase B

---

## 7. Rollback Conditions

Immediate rollback is mandatory if any of the following occurs:

1. `high_risk_misroute_rate_pct > 2`
2. `tier_misroute_rate_pct > 5`
3. `forced_sequential_spike_pct > 40` without explained cohort artifact
4. classifier circuit breaker opens
5. workflow failure pattern indicates routing-induced instability
6. operators lose confidence in observability accuracy

Rollback actions in priority order:

1. set `router_mode=static_policy_only`
2. if needed, set `dynamic_routing_enabled=false`
3. if needed, set `force_sequential=true`

Target rollback expectation:

- under 30 seconds after operator action and service restart / config reload path

---

## 8. Configuration Plan

### 8.1 Production Before M7 Enablement

Current:

- `dynamic_routing_enabled=false`
- `router_mode=static_policy_only`

### 8.2 Phase A Advisory-Only

Proposed:

- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_advisory`
- `limited_exposure_cohort_pct=100` only within approved cohort
- `m7_exposure_cohorts.json.runtime_controls.cohort_enabled=true`

### 8.3 Phase B Limited Enforced

Proposed:

- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_enforced`
- cohort file remains constrained

No further widening is allowed in the same approval package.

---

## 9. Required Monitoring

During M7 controlled enablement, the following must be reviewed per validation session:

1. total routing samples
2. `gated_parallel_allowed` count
3. `forced_sequential` count and ratio
4. decision source distribution
5. deny / degrade reason distribution
6. workflow status distribution
7. `execution_dispatch` P50 / P95
8. approval / rejection paths if triggered

Operational report inputs:

- `orchestrator/scripts/generate_accelerated_validation_report.js`
- `orchestrator/scripts/generate_routing_evaluation_report.js`

---

## 10. Approval Model

### Required for Phase A

- PM approval
- Architect approval

### Required for Phase B

- PM approval
- Architect approval
- explicit note that enablement is limited and reversible

### Not delegated

The following may not be changed without explicit review:

- widening cohort scope
- enabling high-risk work shapes
- disabling fallback behavior
- disabling observability requirements

---

## 11. Recommended Next Action

As of 2026-03-10, the recommended next action is:

**Pause evidence expansion, restart the local orchestrator on the updated runtime, confirm advisory-only decisions are being logged, then resume Phase A evidence collection**

Reason:

- the Phase A package is ready
- the remaining gap is runtime activation certainty, not design completeness
- it is unsafe to continue collecting evidence from a process that may still be running pre-fix code

This is the fastest path that:

- moves the program forward
- preserves rollback simplicity
- keeps static-policy as the real safety floor

---

## 12. Final Recommendation

Recommended formal decision:

> Record that M7 dynamic routing Phase A is active under advisory-only production mode for the narrow `coding_team_v0 / crm / fe_led` cohort. Keep rollback controls unchanged. Require a separate Architect sign-off before any transition from advisory-only to enforced execution.

---

## 13. Activation Status Update

Phase A was applied on 2026-03-10 with:

- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_advisory`
- `m7_exposure_cohorts.json.runtime_controls.cohort_enabled=true`

Initial post-enable checks:

- `live_validate_vnext_runtime.js` -> `PASS`
- `canary_m7_phase_a_advisory.js` -> `PASS`
- `run_m7_dynamic_routing_trial.js` -> `live_trial`, `cohort_cases=10`, `agreement_rate=0.94`, no threshold breach

Operational interpretation:

- Phase A is active
- rollback remains available through runtime config
- next action is sample collection and observation reporting, not broader rollout

### 13.1 2026-03-12 Status Update

The separate Architect/PM sign-off required above has now been recorded.

See:

- `docs/governance/2026-03-12_m10_phase_b_limited_enforced_signoff.md`

Updated interpretation as of 2026-03-12:

- the narrow cohort has entered **Phase B limited enforced** mode
- rollback expectations and cohort restrictions remain unchanged
- no cohort widening is authorized by this update

---

## 14. Evidence Index

- `docs/governance/m7_go_no_go.md`
- `docs/governance/m8_go_no_go.md`
- `docs/governance/m6_accelerated_validation_go_no_go_2026-03-09.md`
- `docs/03_feature_development/2026-03-09_qa_test_summary.md`
- `docs/03_feature_development/2026-03-09_30min_accelerated_validation_plan.md`
- `orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json`
- `orchestrator/artifacts/canary/live_vnext_runtime/live_vnext_runtime_report.json`
- `orchestrator/artifacts/m7_phase_a/phase_a_enablement_plan.json`
- `orchestrator/artifacts/canary/m7_phase_a_advisory/canary_m7_phase_a_advisory.json`
- `orchestrator/artifacts/m7_trial/preflight_result_20260310_phase_a_enabled.json`
