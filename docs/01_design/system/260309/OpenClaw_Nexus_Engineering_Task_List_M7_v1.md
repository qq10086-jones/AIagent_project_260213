# OpenClaw Nexus vNext
## Engineering Task List — Milestone 7 (M7)
## Date: 2026-03-09
## Type: Engineering Task List
## Author: PM / Architecture Review Draft

---

## 1. Scope Positioning

M7 is **not** a broad “turn everything on” milestone.

M7 exists to move the Coding Team pipeline from **static, policy-gated parallel exposure** toward **limited dynamic routing v1**, while preserving the core governance principles established in M6:

- production remains **deny-by-default**
- sequential execution remains the safe fallback baseline
- dispatch eligibility and completion eligibility must stay aligned
- rollback must remain operational through runtime controls
- milestone approval must be based on machine-readable evidence, not optimism

M7 therefore focuses on one bounded objective:

> **Introduce explainable, auditable, limited-scope adaptive routing for approved workflow classes without removing the existing safety override model.**

---

## 2. M7 Objective

The purpose of M7 is to prove that the system can make **dynamic routing decisions**—including execution-path selection and model-tier recommendation—more effectively than the current static exposure policy, **without increasing release risk**.

M7 specifically aims to:

- add Brain Router classification and routing-decision contracts
- preserve runtime deny-by-default safety behavior
- introduce dynamic routing only for approved workflow cohorts
- improve routing precision and latency observability
- keep rollback to sequential behavior immediate and operational
- produce approval-ready evidence for either wider rollout or continued restriction

M7 does **not** authorize:

- broad default-on GA for all workflow types
- removal of static safety overrides
- autonomous merge conflict write-back
- unbounded provider-routing complexity as a milestone blocker

---

## 3. Entry Criteria (Hard Prerequisites)

No M7 implementation work may begin until **all** of the following are satisfied:

1. M6 task list and design delta are already approved.
2. M6 has progressed beyond simulation-only evidence and produced **real LLM staging or limited-exposure evidence**.
3. An architect-reviewed **M6 Exposure Retrospective** exists.
4. The latest M6 evidence shows:
   - no unresolved P0 or P1 incident caused by gated parallel exposure
   - rollback drill remains operational
   - `forced_sequential` fallback is measurable and explainable
   - FE-safe denial reasons are categorized and queryable
5. M7 design delta and this task list are both approved before execution begins.

If these conditions are not satisfied, the authoritative state remains:

- production `coding_team_v0` sequential by default
- existing M6 guardrails unchanged
- M7 stays in planning only

---

## 4. Design Principles for M7

### 4.1 Dynamic routing does not replace safety governance

Adaptive routing may recommend a path, but final admission remains subject to runtime policy evaluation and safety override.

### 4.2 Static policy is demoted, not deleted

The existing static eligibility controls are retained as:

- safety override
- denylist / emergency brake
- fallback admission layer during classifier uncertainty or degradation

### 4.3 Low confidence must degrade safely

Any low-confidence, ambiguous, or policy-conflicting routing decision must fall back to:

- `forced_sequential`, or
- static-policy-only evaluation

### 4.4 Queryability is mandatory

Every routing decision must be explainable after the fact using durable machine-readable logs.

### 4.5 Completion semantics remain authoritative

No workflow may be dynamically admitted into a parallel path if completion, QA readiness, or release readiness would still be structurally impossible.

### 4.6 Closure is evidence-based

M7 success is defined by measured routing quality, safety, and operational reversibility—not by architectural ambition alone.

---

## 5. In-Scope Deliverables

M7 includes the following delivery domains:

1. **Brain Router Classification v1**
2. **Adaptive Routing Policy Contract v1**
3. **Runtime Integration for Limited Dynamic Routing**
4. **Routing Observability and Auditability**
5. **Limited Dynamic Exposure Experiment**
6. **M7 Closure Evidence and Go/No-Go Review**

---

## 6. Out of Scope for M7

The following items are explicitly out of scope for milestone closure:

- autonomous AI merge resolution with direct write-back to protected branches
- complete removal of static eligibility policy files
- fully open GA rollout for all workflows and all project types
- provider-agnostic multi-vendor orchestration as a closure-critical dependency
- release from ambiguous merge-ready states
- any design that weakens rollback-to-sequential operability

Optional stretch work may be explored, but it must not block closure.

---

## 7. Workstreams

### WS-27 — M7 Design Delta and Approval

#### WS-27-01: Author M7 Design Delta

Produce `OpenClaw_Nexus_Design_Document_v4.md` as the design authority for M7.

Required content:

- scope boundary for limited dynamic routing v1
- routing-state model and policy precedence
- classification contract and routing-decision schema
- rollback behavior and safety degradation rules
- observability additions and closure metrics
- risk register updates

**Acceptance criteria**

- design delta reviewed by Architect
- all new runtime states and decision fields defined explicitly
- no conflict with M6 guardrails or release boundary rules

---

#### WS-27-02: Approval Gate

Obtain formal approval for:

- M7 design delta
- M7 engineering task list
- M7 success metrics and rollback criteria

**Acceptance criteria**

- approval artifacts stored in milestone governance location
- no implementation begins before approval is recorded

---

### WS-28 — Brain Router Classification and Decision Contract

#### WS-28-01: Build Routing Classification Taxonomy v1

Define a two-layer classification taxonomy for routing decisions.

**Layer A — Work Shape**

- `single_branch_safe`
- `dual_branch_parallel_candidate`
- `architectural_orchestration_required`
- `high_risk_release_sensitive`

**Layer B — Domain Lead**

- `fe_led`
- `be_led`
- `fullstack`
- `infra`
- `architecture`

**Acceptance criteria**

- taxonomy documented with positive and negative examples
- ambiguous task classes explicitly mapped to safe fallback behavior
- taxonomy approved by PM + Architect

---

#### WS-28-02: Implement Brain Router Classification v1

Implement a classification component that converts task envelopes into routing features and routing recommendations.

Minimum outputs per decision:

- `work_shape`
- `domain_lead`
- `confidence`
- `parallel_candidate`
- `model_tier`
- `required_contracts`
- `deny_reason` or `degrade_reason`

Suggested artifacts:

- `brain_router/classifier.*`
- `contracts/routing_decision.schema.json`
- replay evaluation fixtures and scoring scripts

**Acceptance criteria**

- classifier runs on sanitized replay corpus
- classifier supports reproducible offline evaluation
- low-confidence cases are explicitly marked and never silently promoted

---

#### WS-28-03: Routing Decision Contract v1

Define the machine-readable contract used between Brain Router and runtime execution control.

The contract must include:

- classifier version
- input feature snapshot reference
- predicted class outputs
- confidence band
- recommended execution path
- recommended model tier
- safety override result
- final execution decision
- decision source

**Acceptance criteria**

- schema is versioned
- schema validation is enforced in integration tests
- contract is sufficient for incident review and replay comparison

---

### WS-29 — Adaptive Runtime Integration (Limited Scope)

#### WS-29-01: Integrate Dynamic Routing with Runtime Policy Evaluation

Integrate routing decisions into runtime without inverting existing policy precedence.

Required precedence:

1. rollout master controls
2. force-sequential override
3. safety deny / static eligibility override
4. routing recommendation evaluation
5. final admission decision

The system must preserve safe fallback when:

- rollout is disabled
- force-sequential is enabled
- static safety override denies
- classifier confidence is low
- routing contract is invalid or missing

**Acceptance criteria**

- runtime integration tests cover all precedence branches
- no execution path bypasses safety override
- final decision is always one of:
  - `gated_parallel_allowed`
  - `forced_sequential`

---

#### WS-29-02: Model Tier Recommendation Path

Add bounded model-tier recommendation support for routing.

Example tiering:

- `fast_low_cost`
- `balanced_default`
- `deep_reasoning`

This work is limited to recommendation and governed execution selection, not open-ended provider abstraction.

**Acceptance criteria**

- model-tier selection is logged per run
- fallback tier is defined when recommendation is invalid or unavailable
- routing remains safe if model-tier recommendation is disabled

---

#### WS-29-03: Safety Degradation and Rollback Controls

Extend runtime controls so operators can instantly revert M7 behavior at runtime without code rollback.

Required controls:

- `router_mode=static_policy_only`
- `parallel_mode=force_sequential`
- `dynamic_routing_enabled=true|false`
- optional workflow/project cohort gating

**Acceptance criteria**

- runbook updated
- rollback drill executed and timed
- M7-specific rollback evidence produced

---

### WS-30 — Observability, Auditability, and Evidence

#### WS-30-01: Routing Decision Log and Audit Trail

Persist the following per workflow run:

- router version
- routing feature snapshot reference
- classifier outputs
- confidence band
- policy override result
- final execution path
- final model tier used
- decision source
- deny / degrade reason

**Acceptance criteria**

- query API or SQL view can retrieve decision history per run
- normalized decision sources are documented
- records are machine-readable and stable enough for closure review

---

#### WS-30-02: Waterfall Trace and Latency Attribution

Extend observability so end-to-end latency can be decomposed by node/stage.

Minimum stages:

- intake
- routing/classification
- policy evaluation
- execution dispatch
- branch completion
- QA admission
- release pack readiness

**Acceptance criteria**

- waterfall data exists for sequential and gated-parallel runs
- P50/P95 latency attribution is queryable
- M7 review can compare routing overhead against overall latency reduction

---

#### WS-30-03: Routing Evaluation Report

Create a repeatable report that compares:

- static-policy baseline
- dynamic-routing candidate behavior
- fallback behavior on ambiguous tasks

Report dimensions:

- routing precision
- high-risk misroute rate
- low-confidence fallback ratio
- `forced_sequential` ratio
- latency delta
- incident delta

**Acceptance criteria**

- report generated from machine-readable evidence
- report reproducible from governed replay / staging artifacts
- report included in approval package

---

### WS-31 — Limited Dynamic Exposure Program

#### WS-31-01: Define Approved M7 Exposure Cohorts

Define exactly which workflows and project classes may participate in dynamic routing trials.

The cohort definition must specify:

- included workflow classes
- excluded high-risk classes
- allowed project types
- explicit deny conditions
- rollback trigger thresholds

**Acceptance criteria**

- cohort file is machine-readable
- high-risk workflow categories are excluded by default
- cohort controls are runtime-configurable

---

#### WS-31-02: Run Limited Dynamic Exposure Trial

Run limited-scope staging or approved production exposure using M7 dynamic routing.

Trial requirements:

- real LLM calls or production-like governed evidence
- circuit-breaker active
- fallback-to-sequential active
- incident capture active
- routing audit trail enabled

**Acceptance criteria**

- trial evidence bundle completed
- no unresolved severe incident remains open at milestone review
- fallback behavior is exercised and verified

---

### WS-32 — Closure Review and Decision Package

#### WS-32-01: M7 Go/No-Go Review Package

Prepare a final milestone package containing:

- design delta reference
- task completion checklist
- routing evaluation report
- rollback drill evidence
- incident summary
- metric summary versus baseline
- recommendation:
  - remain limited
  - expand exposure
  - rollback to static-policy-only
  - defer wider rollout

**Acceptance criteria**

- package reviewed by PM + Architect
- recommendation is evidence-backed
- closure decision is explicit and archived

---

#### WS-32-02: Milestone Closure or Controlled Continuation

M7 may close only if:

- all mandatory workstreams are complete
- success criteria are met or deviations are explicitly approved
- rollback remains operational
- unresolved risks are acceptable and documented

If closure criteria are not met, milestone outcome must explicitly state one of:

- continue in limited exposure
- revert to static-policy-only
- defer M8 planning until remediation completes

---

## 8. Success Metrics (Go/No-Go)

The following metrics must be defined in implementation and reported in closure materials.

### 8.1 Safety Metrics

- **High-risk misroute rate**: must stay below architect-approved threshold
- **Release-boundary violation count**: must be zero
- **Unexplained routing decisions**: must be zero
- **Rollback drill success**: required

### 8.2 Routing Quality Metrics

- classification accuracy on governed replay set
- precision of `dual_branch_parallel_candidate`
- recall of safe dynamic admissions
- low-confidence fallback ratio
- override rate from static safety layer

### 8.3 Reliability Metrics

- `forced_sequential` ratio under limited exposure
- circuit-breaker trigger rate
- patch mismatch rate versus M6 baseline
- diff-first hit / fallback / mismatch comparison versus M6 baseline

### 8.4 Performance Metrics

- P50 and P95 end-to-end latency versus M6 baseline
- routing overhead percentage
- throughput change under approved cohort load

### 8.5 Operational Metrics

- incident count by severity
- mean time to detect routing degradation
- mean time to force rollback to safe mode

No milestone closure may rely solely on “overall faster” claims without the above breakdown.

---

## 9. Risk Register Update for M7

| Risk ID | Risk | Severity | M7 Mitigation | Status |
|---------|------|----------|---------------|--------|
| R-13 | Classifier admits unsafe workflows into dynamic routing | High | low-confidence fallback, static safety override, replay evaluation, limited cohort only | Open until closure |
| R-14 | Dynamic routing improves speed but weakens completion determinism | High | preserve completion/QA/release boundary checks, block ambiguous states | Open until closure |
| R-15 | Routing decisions are not reproducible during incident review | High | versioned routing contract, feature snapshot reference, durable decision log | Open until closure |
| R-16 | Model-tier recommendation causes hidden quality regression | Medium | bounded tiering, baseline comparison, fallback tier, incident review | Open until closure |
| R-17 | Operators cannot quickly disable M7 behavior during degradation | High | runtime rollback controls, mandatory rollback drill, updated runbook | Open until closure |
| R-18 | Static policy and dynamic routing conflict in undefined ways | Medium | explicit precedence order, integration tests, normalized decision source logging | Open until closure |

---

## 10. Non-Blocking Stretch Items

The following may be explored only if core milestone delivery is already on track:

- AI-assisted merge conflict diagnosis (read-only recommendation mode)
- provider fallback abstraction beyond model-tier recommendation
- advanced adaptive token-budget tuning
- richer cohort experimentation by tenant or workflow family

These items must not become closure blockers.

---

## 11. Definition of Success for M7

M7 is successful only when:

1. Dynamic routing is introduced in a bounded, approved, explainable manner.
2. Static safety override and force-sequential rollback remain authoritative and operational.
3. Every routing decision is logged, queryable, and auditable.
4. Limited exposure evidence shows routing quality is acceptable and risk does not increase.
5. Completion, QA, and release boundaries remain deterministic.
6. Closure materials are sufficient to support one of these decisions:
   - remain limited
   - expand exposure
   - rollback to static-policy-only
   - defer wider rollout

---

## 12. Final Approval Requirement

This task list becomes authoritative for M7 only after Architect review and approval together with the corresponding M7 design delta.

Until then:

- M6 governance remains authoritative
- production `coding_team_v0` remains governed by existing rollout controls
- no M7 implementation should begin

