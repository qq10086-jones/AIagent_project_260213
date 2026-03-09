# OpenClaw Nexus vNext
## Design Document v4.0
## Date: 2026-03-09
## Type: Design Addendum for M7
## Supplements: OpenClaw_Nexus_Design_Document_v3.2.md

---

## Changelog from v3.2

| Section | Change |
|---------|--------|
| Section 1 | Updated scope clarification to cover M7 limited dynamic routing boundary |
| Section 5 | Extended runtime state model to include dynamic routing admission layer |
| Section 5.2.1 | Extended policy precedence from two-layer to three-layer with dynamic routing |
| Section 5.4 (NEW) | Classifier unavailability degradation semantics |
| Section 5.5 (NEW) | Model tier recommendation design contract |
| Section 7 | Extended observability requirements for routing decision audit and counterfactual replay |
| Section 8 | Extended state machine with routing decision dimension |
| Section 9 | Added M7 guardrails |
| Section 10 | Updated definition of success for M7 |
| Section 11 | Updated risk register for M7 |
| Section 12 | Updated future roadmap |
| Section 13 | Approval requirement updated for M7 |
| Section 14 (NEW) | Brain Router classification design contract |
| Section 15 (NEW) | Adaptive routing safety degradation model |

Review source: `OpenClaw_Nexus_Engineering_Task_List_M7_v2.md`

---

## 1. Scope Clarification

This document is an M7 design addendum. It does not replace the full v3.2 baseline text.

This addendum keeps the M6 controlled-exposure baseline intact and defines the M7 design delta only.

M7 is not a broad feature-expansion milestone. It is a limited dynamic routing milestone that introduces explainable, auditable, classifier-driven routing for approved workflow classes without removing the existing safety override model.

M7 does not change these baseline truths:

- M6 remains the production baseline before any explicit M7 routing gate is enabled
- production `coding_team_v0` remains sequential by default unless M6 gated parallel exposure is already approved
- static eligibility policy is demoted in priority but not deleted
- deny-by-default remains the governing principle for all routing decisions
- rollback to sequential execution must remain operational through runtime controls

---

## 2. M7 Objective

The purpose of M7 is to prove that the system can make dynamic routing decisions — including execution-path selection and model-tier recommendation — more effectively than the current static exposure policy, without increasing release risk.

M7 therefore focuses on:

- Brain Router classification and routing-decision contracts
- dynamic routing only for approved workflow cohorts
- runtime deny-by-default safety behavior preserved
- routing precision and latency observability
- rollback to sequential behavior immediate and operational
- approval-ready evidence for either wider rollout or continued restriction

M7 does not authorize:

- broad default-on GA for all workflow types
- removal of static safety overrides
- autonomous merge conflict write-back
- unbounded provider-routing complexity as a milestone blocker

---

## 3. Design Principles for M7

### 3.1 Sequential is still the safe baseline

This principle is inherited from M6 and unchanged.

### 3.2 Dynamic routing does not replace safety governance

Adaptive routing may recommend a path, but final admission remains subject to runtime policy evaluation and safety override. The classifier is an advisory layer, not an authority layer.

### 3.3 Static policy is demoted, not deleted

The existing static eligibility controls (`production_parallel_rollout.json` and `parallel_exposure_policy.json`) are retained as:

- safety override
- denylist / emergency brake
- fallback admission layer during classifier uncertainty or degradation
- fallback admission layer during classifier unavailability

### 3.4 Low confidence must degrade safely

Any low-confidence, ambiguous, or policy-conflicting routing decision must fall back to `forced_sequential` or static-policy-only evaluation. Silent promotion of uncertain decisions is a design violation.

### 3.5 Classifier unavailability is treated as low confidence

If the Brain Router classifier is unreachable, times out, returns an invalid response, or is otherwise non-functional, the system must treat this as equivalent to a low-confidence result. The degradation path is defined in Section 5.4.

### 3.6 Dispatch and completion must still match

Inherited from M6. No workflow may be dynamically admitted into a parallel path if completion, QA readiness, or release readiness would still be structurally impossible.

### 3.7 Queryability is mandatory

Every routing decision — including classifier-driven, static-fallback, and degradation-driven decisions — must be explainable after the fact using durable machine-readable logs.

### 3.8 Closure is evidence-based

M7 success is defined by measured routing quality, safety, and operational reversibility — not by architectural ambition alone.

---

## 4. Coding Team Update for M7

### 4.1 Production Parallel Exposure Boundary (Unchanged from M6)

`coding_team_v0` retains the two-state model from M6 (`readiness state` and `exposure state`). M7 does not change these definitions.

### 4.2 FE-safe Completion Boundary (Unchanged from M6)

Unchanged. FE-safe parallel exposure validity rules remain as defined in v3.2.

### 4.3 QA Admission Boundary (Unchanged from M6)

Unchanged. Partial branch success must not unlock QA.

### 4.4 Release Boundary (Unchanged from M6)

Unchanged. Release from `partial_failure`, quarantine, or ambiguous states remains prohibited.

---

## 5. Production Routing and Exposure Model

### 5.1 Runtime State Model (Extended for M7)

The M6 runtime state machine is extended with a routing decision dimension:

**Exposure decision path (M6 baseline, preserved):**

`sequential_locked`
-> `policy_evaluated`
-> `gated_parallel_allowed` or `forced_sequential`

**Routing decision path (M7 extension):**

`routing_pending`
-> `classifier_evaluated` or `classifier_degraded`
-> `dynamic_recommendation_available` or `static_fallback_only`
-> (merges into exposure decision path above)

The routing decision path feeds into the exposure decision path. Routing recommendation alone never overrides the exposure decision.

### 5.2 Required Runtime Controls (Extended for M7)

All M6 controls remain required. M7 adds:

- `dynamic_routing_enabled` (global enable/disable)
- `router_mode` (`dynamic` or `static_policy_only`)
- optional workflow/project cohort gating for dynamic routing
- classifier health circuit-breaker

### 5.2.1 Policy Precedence (Extended to Three Layers)

M7 extends the M6 two-layer policy evaluation to a three-layer model. The M6 layers are preserved in their existing positions. The new dynamic routing layer is inserted between static eligibility evaluation and final admission decision, and is always subordinate to the first two layers.

The required evaluation order is:

1. evaluate `production_parallel_rollout.json` (Layer 1 — rollout master)
2. if rollout is disabled or force-sequential is enabled, final decision is `forced_sequential` — skip all subsequent layers
3. evaluate `parallel_exposure_policy.json` (Layer 2 — static eligibility)
4. if static eligibility denies, final decision is `forced_sequential` — skip routing layer
5. if dynamic routing is disabled (`dynamic_routing_enabled=false` or `router_mode=static_policy_only`), use static eligibility result as final decision — skip routing layer
6. evaluate Brain Router classification result (Layer 3 — dynamic routing advisory)
7. if classifier is unavailable or degraded, treat as low-confidence and fall back to static eligibility result (see Section 5.4)
8. if classifier confidence is below approved threshold, fall back to static eligibility result
9. if classifier recommends and confidence is sufficient, apply recommendation subject to completion boundary check
10. final decision is one of: `gated_parallel_allowed` or `forced_sequential`

No implementation may invert this order. Layers 1 and 2 always have veto authority over Layer 3.

### 5.3 Queryability (Extended for M7)

For every workflow run, the system must persist all M6-required fields plus:

- `router_mode` used at run start
- `dynamic_routing_enabled` state
- `classifier_version`
- `classifier_confidence`
- `classifier_confidence_band` (e.g., `high`, `medium`, `low`, `unavailable`)
- `classifier_work_shape`
- `classifier_domain_lead`
- `classifier_parallel_candidate`
- `classifier_model_tier`
- `classifier_deny_reason` or `classifier_degrade_reason`
- `routing_decision_source` (one of the normalized values below)

Additional normalized values for M7:

- `routing_decision_source`
  - `rollout_master_disabled`
  - `force_sequential_override`
  - `static_eligibility_denied`
  - `dynamic_routing_disabled`
  - `classifier_unavailable_fallback`
  - `classifier_low_confidence_fallback`
  - `classifier_recommended_parallel`
  - `classifier_recommended_sequential`

### 5.4 Classifier Unavailability Degradation Semantics (NEW)

This section defines the authoritative degradation behavior when the Brain Router classifier is non-functional.

**Definition of classifier unavailability:**

The classifier is considered unavailable if any of the following is true:

- classifier service is unreachable (connection refused, DNS failure)
- classifier service times out beyond configured threshold
- classifier returns an HTTP error status
- classifier returns a response that fails routing decision contract schema validation
- classifier health circuit-breaker is in open state

**Required degradation behavior:**

When the classifier is unavailable:

1. the system must treat this as `classifier_confidence_band = unavailable`
2. the system must fall back to static-policy-only evaluation (Layers 1 and 2)
3. the routing decision source must be logged as `classifier_unavailable_fallback`
4. no workflow may be dynamically promoted to parallel execution during classifier unavailability
5. an operational alert must be raised for classifier unavailability

**Recovery behavior:**

When the classifier returns to healthy status:

1. the circuit-breaker must recover through standard half-open / probe behavior
2. the system must resume dynamic routing evaluation for new workflow runs only
3. in-flight workflows that were admitted under static fallback must not be retroactively re-evaluated

This degradation path must have dedicated integration test coverage.

### 5.5 Model Tier Recommendation Design Contract (NEW)

#### 5.5.1 Purpose

The Brain Router classifier may include a model-tier recommendation as part of its routing output. This recommendation advises which LLM capability tier should be used for execution.

#### 5.5.2 Defined Tiers

M7 defines a bounded set of model tiers:

- `fast_low_cost` — for well-understood, low-complexity workflow classes
- `balanced_default` — the standard execution tier and the safe fallback
- `deep_reasoning` — for complex architectural, multi-dependency, or high-risk workflow classes

#### 5.5.3 Tier Selection Authority

Model tier selection follows this precedence:

1. if a static policy override specifies a tier for a workflow class, that tier is authoritative
2. if no static override exists and the classifier provides a recommendation with sufficient confidence, the recommendation is used
3. if the classifier is unavailable, low-confidence, or does not provide a tier recommendation, the tier defaults to `balanced_default`

#### 5.5.4 Tier Misroute as Quality Risk

A tier misroute occurs when a workflow that required a higher-capability tier is executed on a lower tier, resulting in measurably lower output quality or completion failure.

Tier misroute is treated as a routing quality issue and is tracked in the routing evaluation report. Specifically:

- tier misroute rate must be measured against the governed replay corpus
- tier misroute must be included in the high-risk misroute metric
- a sustained tier misroute rate above the architect-approved threshold is a valid reason to disable model-tier recommendation and fall back to `balanced_default` for all workflows

#### 5.5.5 M7 Boundary

In M7, model-tier recommendation is limited to:

- classifier recommends a tier
- runtime logs the recommendation and the actual tier used
- execution uses the recommended tier if confidence and policy permit
- quality comparison between tiers is measured on replay corpus in staging

Model-tier recommendation does not introduce open-ended provider abstraction or multi-vendor orchestration in M7.

---

## 6. Replay Corpus Governance (Unchanged from M6)

All M6 replay governance rules remain in effect. M7 replay evaluation uses the same governed, sanitized corpus.

---

## 7. Observability Additions for M7

In addition to all M6 observability requirements, M7 requires:

- routing decision audit log per run (classifier version, confidence, recommendation, override result, final decision, decision source)
- classifier health and availability metrics
- classifier confidence distribution histogram
- waterfall trace with latency decomposition by stage (intake, routing/classification, policy evaluation, execution dispatch, branch completion, QA admission, release pack readiness)
- P50/P95 latency attribution queryable for sequential and gated-parallel runs
- routing overhead percentage as a standalone metric
- routing evaluation report comparing static-policy baseline vs dynamic-routing candidate behavior, including:
  - routing precision
  - high-risk misroute rate (including tier misroute)
  - low-confidence fallback ratio
  - `forced_sequential` ratio
  - latency delta
  - incident delta
- counterfactual replay comparison: for the same replay case, structured comparison of the result under static-policy path versus dynamic-routing path (staging environment only, not required in production)

These records must be machine-readable and suitable for milestone approval review.

---

## 8. State Machine Update

The M6 workflow state machine remains valid. M7 adds the following clarifications:

- routing decision is a parallel state dimension, not a replacement for workflow state
- `forced_sequential` may result from static policy denial, classifier low-confidence fallback, classifier unavailability fallback, or classifier recommendation — each source is logged distinctly
- `classifier_degraded` is an operational state for the routing subsystem, not a workflow failure state
- rollback to `static_policy_only` mode affects subsequent routing evaluation without redefining the workflow success schema
- `partial_failure` remains never release-eligible regardless of routing path

---

## 9. Guardrails for M7

M7 inherits all M6 guardrails and adds:

1. No M7 implementation work begins before the M7 design delta and task list are both approved.
2. No M7 runtime integration work begins before the Brain Router classifier has passed offline evaluation review by the Architect (internal quality gate between classification work and runtime integration work).
3. Dynamic routing is advisory only; it must never override static safety denials or rollout master controls.
4. Classifier unavailability must degrade to static-policy-only behavior, never to uncontrolled promotion.
5. Model-tier recommendation must have a safe fallback tier (`balanced_default`) and must not introduce open-ended provider abstraction.
6. Any merge-conflict-related exploratory work in M7 must be strictly read-only diagnostic mode with no write operations to any code repository, and must not be included in M7 closure evidence.
7. Emergency rollback must not require code rollback. Operators must be able to disable all M7 behavior through runtime controls alone.

---

## 10. Definition of Success for M7

M7 is successful only when:

1. Dynamic routing is introduced in a bounded, approved, explainable manner for approved workflow cohorts only.
2. Static safety override and force-sequential rollback remain authoritative and operational.
3. Every routing decision — including classifier-driven, static-fallback, and degradation-driven — is logged, queryable, and auditable.
4. Classifier unavailability degrades safely and is covered by integration tests.
5. Limited exposure evidence shows routing quality is acceptable and risk does not increase versus M6 baseline.
6. Model-tier recommendation is logged, measured, and has a safe fallback.
7. Completion, QA, and release boundaries remain deterministic regardless of routing path.
8. Counterfactual replay comparison exists for approved replay cases (staging).
9. Rollback controls are runtime-operable and validated by drill evidence.
10. Closure materials are sufficient to support one of these decisions:
    - remain limited
    - expand exposure
    - rollback to static-policy-only
    - defer wider rollout

---

## 11. Risk Register Update for M7

All M6 risks remain open unless explicitly closed by M6 retrospective.

| Risk ID | Risk | Severity | M7 Mitigation | Status |
|---------|------|----------|---------------|--------|
| R-13 | Classifier admits unsafe workflows into dynamic routing | High | low-confidence fallback, static safety override, replay evaluation, limited cohort only, internal quality gate before runtime integration | Open until M7 closure |
| R-14 | Dynamic routing improves speed but weakens completion determinism | High | preserve completion/QA/release boundary checks, block ambiguous states | Open until M7 closure |
| R-15 | Routing decisions are not reproducible during incident review | High | versioned routing contract, feature snapshot reference, durable decision log, normalized decision sources | Open until M7 closure |
| R-16 | Model-tier recommendation causes hidden quality regression | High (upgraded from Medium) | bounded tiering, tier misroute tracked in routing evaluation, fallback to balanced_default, quality comparison on replay corpus, threshold for disabling tier recommendation | Open until M7 closure |
| R-17 | Operators cannot quickly disable M7 behavior during degradation | High | runtime rollback controls, mandatory rollback drill, updated runbook, classifier circuit-breaker | Open until M7 closure |
| R-18 | Static policy and dynamic routing conflict in undefined ways | Medium | explicit three-layer precedence order, integration tests for all precedence branches, normalized decision source logging | Open until M7 closure |
| R-19 (NEW) | Classifier unavailability causes uncontrolled routing behavior | High | explicit degradation semantics (Section 5.4), circuit-breaker, dedicated integration tests, operational alert | Open until M7 closure |
| R-20 (NEW) | M7 launched on insufficient M6 evidence base | High | quantified M6 entry criteria (minimum replay coverage and gated-parallel run counts), M6 retrospective required | Open until M7 closure |

---

## 12. Future Roadmap Update

| Item | Target | Status | Reference |
|------|--------|--------|-----------|
| Structured diff / patch execution | M5 | Complete and governed | M5 WS-19 |
| Context budget tracking | M5 | Complete and governed | M5 WS-20 |
| BE + FE parallel execution readiness | M5 | Complete as readiness only | M5 WS-21 |
| Controlled production parallel exposure | M6 | In progress or complete (per M6 retrospective) | M6 WS-23 to WS-26 |
| Brain Router LLM classification v1 | M7 | Active | M7 WS-28 |
| Adaptive model routing (limited) | M7 | Active | M7 WS-29 |
| Full adaptive model routing | Future | Deferred | Future milestone |
| AI merge conflict resolution (write-back) | Future | Deferred | Future milestone |
| Multi-vendor provider orchestration | Future | Deferred | Future milestone |

---

## 13. Approval Requirement

This design addendum becomes authoritative for M7 only after Architect review and approval, together with the corresponding M7 engineering task list.

Until then:

- M6 governance remains authoritative
- production `coding_team_v0` remains governed by existing M6 rollout controls
- no M7 implementation should begin

---

## 14. Brain Router Classification Design Contract (NEW)

### 14.1 Purpose

The Brain Router is a classifier component that converts task envelopes into routing features and routing recommendations. It is advisory only and does not have authority to override safety controls.

### 14.2 Classification Taxonomy v1

**Layer A — Work Shape:**

- `single_branch_safe` — task can be completed by a single implementation branch
- `dual_branch_parallel_candidate` — task is a candidate for BE+FE parallel execution
- `architectural_orchestration_required` — task requires coordinated multi-component changes
- `high_risk_release_sensitive` — task touches release-critical paths and must be treated conservatively

**Layer B — Domain Lead:**

- `fe_led`
- `be_led`
- `fullstack`
- `infra`
- `architecture`

### 14.3 Required Classifier Outputs

For every classification, the classifier must produce:

- `work_shape` — one of the Layer A values
- `domain_lead` — one of the Layer B values
- `confidence` — numeric confidence score
- `confidence_band` — `high`, `medium`, `low`, or `unavailable`
- `parallel_candidate` — boolean
- `model_tier` — one of the defined tiers or `null`
- `required_contracts` — list of contracts the workflow must satisfy
- `deny_reason` or `degrade_reason` — if the classifier recommends denial or degradation

### 14.4 Ambiguity Handling

Ambiguous task classes — those where the classifier cannot assign a work shape with confidence above the approved threshold — must be explicitly mapped to safe fallback behavior:

- `work_shape` defaults to `single_branch_safe`
- `parallel_candidate` defaults to `false`
- `model_tier` defaults to `balanced_default`
- `confidence_band` is set to `low`

This prevents silent promotion of uncertain classifications.

---

## 15. Adaptive Routing Safety Degradation Model (NEW)

### 15.1 Degradation Hierarchy

The system recognizes three degradation levels for the routing subsystem:

**Level 0 — Fully Operational:**

- classifier is healthy and responsive
- dynamic routing evaluation proceeds normally
- all three policy layers are active

**Level 1 — Classifier Degraded:**

- classifier is reachable but returning low-confidence results at an abnormal rate
- dynamic routing falls back to static-policy-only for all low-confidence cases
- operational warning is raised
- system remains in dynamic mode for high-confidence cases

**Level 2 — Classifier Unavailable:**

- classifier is unreachable, timed out, or circuit-breaker is open
- all dynamic routing evaluation is bypassed
- system operates in static-policy-only mode
- operational alert is raised
- recovery follows circuit-breaker half-open / probe semantics

### 15.2 Degradation Logging

Each degradation event must be logged with:

- degradation level
- trigger reason
- timestamp
- affected workflow run IDs (if applicable)
- recovery timestamp (when resolved)

### 15.3 Degradation Testing

Integration tests must cover:

- transition from Level 0 to Level 1
- transition from Level 0 to Level 2
- transition from Level 1 to Level 2
- recovery from Level 2 to Level 0
- recovery from Level 1 to Level 0
- correct routing behavior at each level
- correct logging at each level
