# OpenClaw Nexus vNext
## Design Document v3.2
## Date: 2026-03-08
## Type: Design Addendum for M6
## Supplements: docs/01_design/system/260308/260308_2053/OpenClaw_Nexus_Design_Document_v3.1.md

---

## Changelog from v3.1

| Section | Change |
|---------|--------|
| Section 9.5 | Added M6 production parallel exposure model and FE-safe completion boundary |
| Section 16 | Added replay data governance and production rollout guardrails |
| Section 17 | Added rollout gate observability requirements |
| Section 20 | Clarified rollout-state interaction with `partial_failure` and sequential fallback |
| Section 22 | Added M6 rollout-governance success criteria |
| Section 23 | Added M6 risks for replay data handling and production exposure mismatch |
| Section 24 | Reclassified parallel execution from M5 readiness to M6 controlled exposure program |
| Section 27 (NEW) | Production Parallel Exposure Model |
| Section 28 (NEW) | Replay Corpus Governance |

Review source: `docs/01_design/system/260308/260308_2053/open_claw_nexus_engineering_task_list_m6_v_2.md`

---

## 1. Scope Clarification

This document is an M6 design addendum. It does not replace the full v3.1 baseline text.

This addendum keeps the M5 closed implementation baseline intact and defines the M6 design delta only.

M6 is not a broad feature-expansion milestone. It is a rollout-readiness and controlled-exposure milestone for the existing Coding Team pipeline.

M6 does not change these baseline truths:

- M5 remains the production baseline before any explicit M6 rollout gate is enabled
- production `coding_team_v0` remains sequential by default
- BE/FE parallel execution is not considered production-open merely because DAG readiness exists

---

## 2. M6 Objective

The purpose of M6 is to decide whether production parallel exposure can be safely opened for a limited approved subset of workflows.

M6 therefore focuses on:

- explicit exposure contracts
- replay-driven staging evidence
- deny-by-default runtime gating
- rollback operability
- approval-ready baseline metrics

M6 does not authorize broad default-on parallel rollout.

---

## 3. Design Principles for M6

### 3.1 Sequential is still the safe baseline

Production must remain sequential unless an approved machine-readable exposure policy explicitly permits a workflow to enter the gated parallel path.

### 3.2 Dispatch and completion must match

No workflow may be allowed onto a gated parallel path unless:

- dispatch eligibility is satisfied
- FE/BE completion semantics are explicit
- QA admission conditions are explicit
- release gating is explicit

If dispatch is possible but completion is structurally impossible, the design is invalid.

### 3.3 Real replay evidence outranks synthetic optimism

Synthetic canaries remain required, but real Discord-originated replay patterns are the primary evidence source for M6 decisions.

### 3.4 Rollback must be operational, not theoretical

Exposure is only acceptable if operators can revert to sequential execution through runtime controls without code rollback.

### 3.5 Replay data is a governed artifact

Replay fixtures, reports, and approval packages are governed operational artifacts and must obey sanitization, retention, and review rules.

---

## 4. Coding Team Update for M6

### 4.1 Production Parallel Exposure Boundary

`coding_team_v0` now has two distinct states in design:

1. `readiness state`
   - DAG capability exists
   - synthetic workflows may validate parallel behavior
   - no production exposure is implied

2. `exposure state`
   - production workflow may enter gated parallel execution only if runtime policy allows it
   - the policy decision must be explicit, logged, and queryable

### 4.2 FE-safe Completion Boundary

FE-safe parallel exposure is valid only if one of the following is true:

1. FE completion is explicitly defined to not require `be_to_fe` handoff for the approved workflow class
2. FE completion still requires BE material, in which case that workflow class is not FE-safe and must remain sequential

This rule prevents the system from exposing a dispatch path that cannot actually complete.

### 4.3 QA Admission Boundary

QA may start after parallel execution only when:

- both required implementation branches have reached contract-valid completion
- branch artifacts are in a merge-ready state
- no branch is in timeout, failed, quarantined, or ambiguous status

Partial branch success must not unlock QA.

### 4.4 Release Boundary

Release pack generation must not proceed from:

- `partial_failure`
- branch quarantine
- unresolved patch application failure
- ambiguous merge readiness

Release is permitted only from a deterministic merge-ready workflow state.

---

## 5. Production Parallel Exposure Model

### 5.1 Runtime State Model

Production parallel exposure is governed by a deny-by-default runtime state machine:

`sequential_locked`
-> `policy_evaluated`
-> `gated_parallel_allowed` or `forced_sequential`

The default state is `sequential_locked`.

### 5.2 Required Runtime Controls

The runtime exposure model must support:

- global parallel enable/disable
- whitelist by workflow type
- whitelist by project type
- force-sequential override
- emergency rollback switch
- optional limited-exposure or cohort controls

### 5.2.1 Policy Precedence

M6 uses two machine-readable policy layers with distinct responsibilities:

1. `production_parallel_rollout.json`
   - rollout master switch
   - global enable or disable
   - force-sequential override
   - limited exposure controls

2. `parallel_exposure_policy.json`
   - workflow-type and project-type eligibility rules
   - explicit deny conditions
   - whitelist logic for FE-safe admission

The required evaluation order is:

1. evaluate `production_parallel_rollout.json`
2. if rollout is disabled or force-sequential is enabled, final decision is `forced_sequential`
3. only if rollout layer permits evaluation, evaluate `parallel_exposure_policy.json`
4. if eligibility policy denies, final decision is `forced_sequential`
5. only if both layers allow, final decision is `gated_parallel_allowed`

No implementation may invert this order.

### 5.3 Queryability

For every workflow run, the system must persist:

- rollout control state used at run start
- eligibility decision
- deny reason or allow reason
- execution path actually used
- `effective_exposure_decision`
- `effective_exposure_decision_source`

Required normalized values:

- `effective_exposure_decision`
  - `gated_parallel_allowed`
  - `forced_sequential`
- `effective_exposure_decision_source`
  - `rollout_master_disabled`
  - `force_sequential_override`
  - `eligibility_policy_denied`
  - `eligibility_policy_allowed`

This is required for incident review and closure decisions.

---

## 6. Replay Corpus Governance

### 6.1 Purpose

Replay corpus artifacts exist to test the real Discord prompt distribution without exposing unsafe raw production data in milestone evidence.

### 6.2 Governance Rules

Replay governance must define:

- whether raw prompts may be retained
- where raw prompts may be stored if retention is allowed
- how user identifiers, channel identifiers, links, attachments, and secrets are sanitized
- who may create or review replay fixtures
- how replay artifacts are retained, redacted, and referenced in approval materials

### 6.3 Approval Boundary

No M6 approval package may depend on unsanitized raw prompts as its primary evidence source.

Approval-ready evidence must be reproducible from sanitized replay fixtures and structured result bundles.

---

## 7. Observability Additions for M6

In addition to existing M4/M5 observability, M6 requires:

- rollout gate decision log per run
- effective exposure policy snapshot per run
- sequential vs gated-parallel comparison artifacts
- diff-first hit/fallback/mismatch metrics from replay corpus
- FE-safe eligibility metrics with categorized denial reasons
- rollback drill evidence

These records must be machine-readable and suitable for milestone approval review.

---

## 8. State Machine Update

The existing workflow state machine remains valid, with these M6 clarifications:

- `partial_failure` remains a valid branch-mixed outcome state
- `partial_failure` is never release-eligible
- `forced_sequential` is an exposure decision, not a workflow failure state
- rollback to sequential mode affects subsequent admission behavior without redefining the workflow success schema

---

## 9. Guardrails for M6

M6 introduces these additional guardrails:

1. No production exposure work begins before the M6 design delta and task list are both approved.
2. No production workflow may enter a gated parallel path while validator or QA rules still make completion impossible.
3. Replay artifacts must follow explicit sanitization and retention policy.
4. Exposure policy must be runtime-configurable, not hardcoded.
5. Emergency rollback must not require code rollback.

---

## 10. Definition of Success for M6

M6 is successful only when:

1. A governed replay corpus of real Discord-originated prompt patterns exists.
2. Replay-driven staging runs produce structured, isolated, machine-readable results.
3. Sequential versus gated-parallel comparison exists for approved replay cases.
4. FE-safe completion semantics are explicit and testable.
5. Failure-handling semantics for mixed branch outcomes are explicit and testable.
6. Production exposure remains deny-by-default and policy-driven.
7. Rollback controls are runtime-operable and validated by drill evidence.
8. Context budget, diff-first reliability, patch mismatch, and parallel eligibility baselines are derived from replay evidence.
9. Approval materials are sufficient to support one of these decisions:
   - stay gated
   - expand exposure
   - rollback to sequential-only
   - defer wider rollout

---

## 11. Risk Register Update for M6

| Risk ID | Risk | Severity | M6 Mitigation | Status |
|---------|------|----------|---------------|--------|
| R-9 | Production exposure mismatch: runtime allows dispatch but validator/QA blocks completion | High | explicit FE-safe completion contract + runtime bridge workstream + integration tests | Open until M6 closure |
| R-10 | Replay data leakage through fixtures or approval artifacts | High | replay governance, sanitization, retention, and redaction rules | Open until M6 closure |
| R-11 | Rollback exists in config but is not operationally fast enough | High | mandatory rollback drill and runbook evidence | Open until M6 closure |
| R-12 | Eligibility policy is too permissive or too conservative | Medium | replay-derived denial baseline + whitelist tuning from evidence | Open until M6 closure |

---

## 12. Future Roadmap Update

The roadmap status is now:

| Item | Target | Status | Reference |
|------|--------|--------|-----------|
| Structured diff / patch execution | M5 | Complete and governed | M5 WS-19 |
| Context budget tracking | M5 | Complete and governed | M5 WS-20 |
| BE + FE parallel execution readiness | M5 | Complete as readiness only; production remains sequential by default | M5 WS-21 |
| Controlled production parallel exposure | M6 | Candidate, approval pending | M6 WS-23 to WS-26 |
| Brain Router LLM classification | Future | Deferred | Future milestone |
| Adaptive model routing | M7 | Deferred | Future milestone |

---

## 13. Approval Requirement

This design addendum becomes authoritative for M6 only after Architect review and approval, together with the corresponding M6 engineering task list.

Until then:

- M5 closed state remains authoritative
- production `coding_team_v0` remains sequential by default
- no M6 implementation should begin
