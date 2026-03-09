# OpenClaw Nexus vNext
## Engineering Task List — Milestone 7 (M7)
## Version: v2.1
## Date: 2026-03-10
## Type: Engineering Task List
## Author: PM / Architecture Review Draft
## Design Authority: OpenClaw_Nexus_Design_Document_v4.md

---

## Changelog from v1

| Section | Change |
|---------|--------|
| Section 3 | Added quantified M6 entry criteria (minimum replay coverage and gated-parallel run counts) |
| Section 4.4 (NEW) | Added classifier unavailability principle |
| Section 6 | Strengthened merge-conflict out-of-scope boundary |
| Section 7 WS-27-01 | Extended required content to include three-layer policy precedence update and classifier degradation semantics |
| Section 7 WS-28-02 | Added directional accuracy constraints and offline evaluation quality gate |
| Section 7 WS-28-04 (NEW) | Internal quality gate: classifier offline evaluation must pass Architect review before runtime integration begins |
| Section 7 WS-29-01 | Added classifier unavailability as explicit fallback path with dedicated integration test requirement |
| Section 7 WS-29-02 | Clarified tier selection authority, tier misroute definition, and M7 boundary for model tier |
| Section 7 WS-30-03 | Added counterfactual replay comparison dimension |
| Section 8 | Added directional thresholds for safety and routing quality metrics |
| Section 9 | Added R-19 (classifier unavailability) and R-20 (insufficient M6 evidence base) |
| Section 10 | Added explicit boundary for merge conflict stretch item |
| Section 12 | Clarified that milestone closure is not production activation authority |
| Section 13 (NEW) | Added post-M8 controlled enablement note for production activation |

---

## 1. Scope Positioning

M7 is **not** a broad "turn everything on" milestone.

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
2. M6 has progressed beyond simulation-only evidence and produced **real LLM staging or limited-exposure evidence** that meets the following minimum coverage thresholds:
   - the governed replay corpus covers at least **3 distinct workflow classes** from production Discord-originated prompts
   - each workflow class has at least **20 successful gated-parallel staging runs** with structured result bundles
   - at least **1 workflow class** has been exposed in limited production (not staging-only) with zero unresolved P0/P1 incidents
3. An architect-reviewed **M6 Exposure Retrospective** exists and includes:
   - quantified summary of replay corpus coverage
   - structured comparison of sequential vs gated-parallel outcomes
   - categorized FE-safe denial reason distribution
   - rollback drill results with timing evidence
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

Adaptive routing may recommend a path, but final admission remains subject to runtime policy evaluation and safety override. The classifier is an advisory layer, not an authority layer.

### 4.2 Static policy is demoted, not deleted

The existing static eligibility controls are retained as:

- safety override
- denylist / emergency brake
- fallback admission layer during classifier uncertainty or degradation
- fallback admission layer during classifier unavailability

### 4.3 Low confidence must degrade safely

Any low-confidence, ambiguous, or policy-conflicting routing decision must fall back to:

- `forced_sequential`, or
- static-policy-only evaluation

### 4.4 Classifier unavailability is treated as low confidence (NEW)

If the classifier is unreachable, timed out, returns invalid responses, or the circuit-breaker is open, the system must treat this as equivalent to low confidence and fall back to static-policy-only evaluation. See design document v4.0 Section 5.4 for authoritative degradation semantics.

### 4.5 Queryability is mandatory

Every routing decision must be explainable after the fact using durable machine-readable logs.

### 4.6 Completion semantics remain authoritative

No workflow may be dynamically admitted into a parallel path if completion, QA readiness, or release readiness would still be structurally impossible.

### 4.7 Closure is evidence-based

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
- any merge-conflict-related work that produces write operations to code repositories (read-only diagnostic exploration is permitted as a non-blocking stretch item but must not appear in M7 closure evidence)

Optional stretch work may be explored, but it must not block closure.

---

## 7. Workstreams

### WS-27 — M7 Design Delta and Approval

#### WS-27-01: Author M7 Design Delta

Produce `OpenClaw_Nexus_Design_Document_v4.md` as the design authority for M7.

Required content:

- scope boundary for limited dynamic routing v1
- routing-state model and policy precedence
- **three-layer policy precedence chain update** (extending the M6 two-layer model to include the dynamic routing advisory layer, with explicit evaluation order and veto semantics)
- classification contract and routing-decision schema
- rollback behavior and safety degradation rules
- **classifier unavailability degradation semantics** (defining the authoritative behavior when the classifier is non-functional, including circuit-breaker, fallback, alerting, and recovery)
- **model tier recommendation design contract** (defining tier set, selection authority, fallback tier, and tier misroute as a tracked quality metric)
- observability additions and closure metrics
- risk register updates

**Acceptance criteria**

- design delta reviewed by Architect
- all new runtime states and decision fields defined explicitly
- three-layer policy precedence is formally specified with evaluation order
- classifier unavailability degradation path is an authoritative design contract, not just an implementation detail
- model tier recommendation has explicit scope boundary for M7
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
- ambiguous task classes explicitly mapped to safe fallback behavior (default to `single_branch_safe` with `parallel_candidate=false`)
- taxonomy approved by PM + Architect

---

#### WS-28-02: Implement Brain Router Classification v1

Implement a classification component that converts task envelopes into routing features and routing recommendations.

Minimum outputs per decision:

- `work_shape`
- `domain_lead`
- `confidence`
- `confidence_band`
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
- offline evaluation report produced with the following directional constraints:
  - **high-risk misroute rate** (cases where `high_risk_release_sensitive` workflows are classified as `dual_branch_parallel_candidate`): must be **below 2%** on the governed replay corpus, or an architect-approved alternative threshold with documented rationale
  - **low-confidence fallback ratio**: should be in the range of **10%-40%** on real replay distribution — below 10% suggests the classifier is overconfident, above 40% suggests insufficient classification power to justify dynamic routing
  - **`dual_branch_parallel_candidate` precision**: must be **above 85%** on governed replay corpus — false parallel admissions are more dangerous than false sequential fallbacks
- these thresholds are directional targets; the Architect may adjust them during review, but the classifier must be evaluated against explicit numeric criteria before proceeding to runtime integration

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

#### WS-28-04: Classifier Offline Evaluation Quality Gate (NEW)

This is an internal quality gate. It must be satisfied before any WS-29 (runtime integration) work begins.

**Gate requirements:**

1. WS-28-02 offline evaluation report is complete
2. the evaluation report has been reviewed by the Architect
3. all directional accuracy thresholds from WS-28-02 are met or the Architect has approved documented deviations
4. ambiguity handling defaults are verified (Section 14.4 of design document)
5. the Architect provides a written go/no-go for runtime integration

**Acceptance criteria**

- gate review is documented and archived
- if the gate is not passed, WS-29 remains blocked and the reason is explicitly recorded
- no runtime integration code may be merged before this gate is cleared

---

### WS-29 — Adaptive Runtime Integration (Limited Scope)

**Hard prerequisite: WS-28-04 quality gate must be passed before any WS-29 work begins.**

#### WS-29-01: Integrate Dynamic Routing with Runtime Policy Evaluation

Integrate routing decisions into runtime without inverting existing policy precedence.

Required precedence (per design document v4.0 Section 5.2.1):

1. rollout master controls (Layer 1)
2. force-sequential override
3. static eligibility evaluation (Layer 2)
4. dynamic routing disabled check
5. Brain Router classification evaluation (Layer 3)
6. confidence threshold check
7. completion boundary check
8. final admission decision

The system must preserve safe fallback when:

- rollout is disabled
- force-sequential is enabled
- static safety override denies
- classifier confidence is low
- routing contract is invalid or missing
- **classifier is unavailable** (connection failure, timeout, circuit-breaker open, invalid response)

**Acceptance criteria**

- runtime integration tests cover all precedence branches
- no execution path bypasses safety override
- **classifier unavailability is treated as `classifier_confidence_band = unavailable` and falls back to static-policy-only evaluation**
- **classifier unavailability fallback path has dedicated integration tests** (covering connection failure, timeout, invalid response, and circuit-breaker open scenarios)
- **an operational alert is raised on classifier unavailability**
- final decision is always one of:
  - `gated_parallel_allowed`
  - `forced_sequential`

---

#### WS-29-02: Model Tier Recommendation Path

Add bounded model-tier recommendation support for routing.

Defined tiers:

- `fast_low_cost`
- `balanced_default`
- `deep_reasoning`

**Tier selection authority (per design document v4.0 Section 5.5.3):**

1. if a static policy override specifies a tier for a workflow class, that tier is authoritative
2. if no static override exists and the classifier provides a recommendation with sufficient confidence, the recommendation is used
3. if the classifier is unavailable, low-confidence, or does not provide a tier recommendation, the tier defaults to `balanced_default`

**Tier misroute definition:**

A tier misroute occurs when a workflow that required a higher-capability tier is executed on a lower tier, resulting in measurably lower output quality or completion failure. Tier misroute is tracked as part of the routing evaluation report and included in the high-risk misroute metric.

**M7 boundary:**

Model-tier recommendation in M7 is limited to recommendation, logging, and governed execution selection. It does not introduce open-ended provider abstraction or multi-vendor orchestration. If tier misroute rate exceeds the architect-approved threshold, the system must support runtime fallback to `balanced_default` for all workflows without code changes.

**Acceptance criteria**

- model-tier selection is logged per run
- fallback tier (`balanced_default`) is defined and used when recommendation is invalid, unavailable, or low-confidence
- routing remains safe if model-tier recommendation is disabled
- tier misroute is defined, measurable, and included in routing evaluation report
- runtime control exists to disable tier recommendation and force `balanced_default` globally

---

#### WS-29-03: Safety Degradation and Rollback Controls

Extend runtime controls so operators can instantly revert M7 behavior at runtime without code rollback.

Required controls:

- `router_mode=static_policy_only`
- `parallel_mode=force_sequential`
- `dynamic_routing_enabled=true|false`
- optional workflow/project cohort gating
- classifier health circuit-breaker with configurable thresholds

**Acceptance criteria**

- runbook updated
- rollback drill executed and timed
- M7-specific rollback evidence produced
- circuit-breaker behavior verified under simulated classifier failure

---

### WS-30 — Observability, Auditability, and Evidence

#### WS-30-01: Routing Decision Log and Audit Trail

Persist the following per workflow run:

- router version
- routing feature snapshot reference
- classifier outputs (all fields from design document Section 14.3)
- confidence band
- policy override result
- final execution path
- final model tier used
- decision source (using normalized values from design document Section 5.3)
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
- routing overhead percentage is reported as a standalone metric
- M7 review can compare routing overhead against overall latency reduction

---

#### WS-30-03: Routing Evaluation Report

Create a repeatable report that compares:

- static-policy baseline
- dynamic-routing candidate behavior
- fallback behavior on ambiguous tasks

Report dimensions:

- routing precision
- high-risk misroute rate (including tier misroute)
- low-confidence fallback ratio
- `forced_sequential` ratio
- latency delta
- incident delta

**Counterfactual replay comparison (NEW):**

The report must include a counterfactual comparison dimension: for the same governed replay cases, a structured comparison of the result under static-policy path versus dynamic-routing path. This comparison is required in staging environment only and is not required in production. The purpose is to provide evidence for whether dynamic routing produces measurably better outcomes than static policy for the approved cohorts.

**Acceptance criteria**

- report generated from machine-readable evidence
- report reproducible from governed replay / staging artifacts
- counterfactual comparison included for at least the approved exposure cohorts
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
- classifier health monitoring active

**Acceptance criteria**

- trial evidence bundle completed
- no unresolved severe incident remains open at milestone review
- fallback behavior is exercised and verified
- classifier unavailability fallback is exercised at least once during trial (may be simulated)

---

### WS-32 — Closure Review and Decision Package

#### WS-32-01: M7 Go/No-Go Review Package

Prepare a final milestone package containing:

- design delta reference
- task completion checklist
- routing evaluation report (including counterfactual comparison)
- rollback drill evidence
- incident summary
- metric summary versus baseline
- classifier offline evaluation results
- classifier unavailability drill results
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
- WS-28-04 quality gate was passed (or deviation was approved)
- success criteria are met or deviations are explicitly approved
- rollback remains operational
- unresolved risks are acceptable and documented

If closure criteria are not met, milestone outcome must explicitly state one of:

- continue in limited exposure
- revert to static-policy-only
- defer M8 planning until remediation completes

---

## 8. Success Metrics (Go/No-Go)

The following metrics must be reported in closure materials. Directional thresholds are provided as architect-reviewable targets; final thresholds are confirmed during WS-27-02 approval.

### 8.1 Safety Metrics

- **High-risk misroute rate**: must stay below architect-approved threshold (directional target: **< 2%**)
- **Tier misroute rate**: must stay below architect-approved threshold (directional target: **< 5%**)
- **Release-boundary violation count**: must be **zero**
- **Unexplained routing decisions**: must be **zero** (every decision must have a normalized decision source)
- **Rollback drill success**: required, must complete within architect-approved time window
- **Classifier unavailability drill success**: required

### 8.2 Routing Quality Metrics

- classification accuracy on governed replay set (directional target: **> 85%** weighted accuracy)
- precision of `dual_branch_parallel_candidate` (directional target: **> 85%**)
- recall of safe dynamic admissions
- low-confidence fallback ratio (directional target: **10%-40%** on real replay distribution)
- override rate from static safety layer

### 8.3 Reliability Metrics

- `forced_sequential` ratio under limited exposure
- circuit-breaker trigger rate
- classifier unavailability rate and mean recovery time
- patch mismatch rate versus M6 baseline
- diff-first hit / fallback / mismatch comparison versus M6 baseline

### 8.4 Performance Metrics

- P50 and P95 end-to-end latency versus M6 baseline
- routing overhead percentage (directional target: **< 5%** of total workflow latency)
- throughput change under approved cohort load

### 8.5 Operational Metrics

- incident count by severity
- mean time to detect routing degradation
- mean time to force rollback to safe mode

No milestone closure may rely solely on "overall faster" claims without the above breakdown.

---

## 9. Risk Register Update for M7

| Risk ID | Risk | Severity | M7 Mitigation | Status |
|---------|------|----------|---------------|--------|
| R-13 | Classifier admits unsafe workflows into dynamic routing | High | low-confidence fallback, static safety override, replay evaluation, limited cohort only, internal quality gate (WS-28-04) before runtime integration | Open until closure |
| R-14 | Dynamic routing improves speed but weakens completion determinism | High | preserve completion/QA/release boundary checks, block ambiguous states | Open until closure |
| R-15 | Routing decisions are not reproducible during incident review | High | versioned routing contract, feature snapshot reference, durable decision log, normalized decision sources | Open until closure |
| R-16 | Model-tier recommendation causes hidden quality regression | High | bounded tiering, tier misroute tracked in routing evaluation, fallback to balanced_default, quality comparison on replay corpus, runtime switch to disable tier recommendation | Open until closure |
| R-17 | Operators cannot quickly disable M7 behavior during degradation | High | runtime rollback controls, mandatory rollback drill, updated runbook, classifier circuit-breaker | Open until closure |
| R-18 | Static policy and dynamic routing conflict in undefined ways | Medium | explicit three-layer precedence order, integration tests for all precedence branches, normalized decision source logging | Open until closure |
| R-19 (NEW) | Classifier unavailability causes uncontrolled routing behavior | High | explicit degradation semantics in design document Section 5.4, circuit-breaker, dedicated integration tests, operational alert, unavailability drill in trial | Open until closure |
| R-20 (NEW) | M7 launched on insufficient M6 evidence base | High | quantified M6 entry criteria with minimum replay coverage (3 workflow classes, 20 runs each) and at least 1 class in limited production exposure | Open until closure |

---

## 10. Non-Blocking Stretch Items

The following may be explored only if core milestone delivery is already on track:

- AI-assisted merge conflict diagnosis (**strictly read-only recommendation mode only; no write operations to any code repository; must not appear in M7 closure evidence**)
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
4. Classifier unavailability degrades safely and is verified by integration tests and trial drill.
5. Limited exposure evidence shows routing quality is acceptable and risk does not increase.
6. Model-tier recommendation is logged, measured against tier misroute metric, and has safe fallback.
7. Completion, QA, and release boundaries remain deterministic.
8. Counterfactual replay comparison exists for approved cohorts (staging).
9. Internal quality gate (WS-28-04) was passed before runtime integration.
10. Closure materials are sufficient to support one of these decisions:
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

After M7 milestone closure, any move from `static_policy_only` toward production dynamic routing must be governed by a separate post-M8 enablement review. Closure of this task list is not equivalent to production activation approval.

---

## 13. Post-M8 Controlled Enablement Note (NEW)

This task list remains the implementation authority for M7, but it is no longer sufficient by itself to authorize production enablement.

As of 2026-03-10:

- M7 implementation is complete
- M8 staging/live evidence is complete
- accelerated validation evidence exists
- production still remains at:
  - `master_enabled=true`
  - `dynamic_routing_enabled=false`
  - `router_mode=static_policy_only`

Therefore, the next step after milestone closure is not additional M7 implementation work. The next step is controlled production enablement under separate governance.

The authoritative enablement direction is:

1. start with `Phase A: advisory-only`
2. restrict to approved cohort only
3. require PM + Architect sign-off before any enforced mode
4. preserve rollback through runtime config only

Reference:

- `docs/governance/post_m8_m7_controlled_enablement_plan_2026-03-10.md`
