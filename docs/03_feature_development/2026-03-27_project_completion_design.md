# Project Completion Design

- Date: 2026-03-27
- Status: DRAFT FOR EXECUTION
- Classification: MAINLINE PRODUCT COMPLETION
- Scope: close the gap between "workflow succeeded" and "user actually received a usable product"

---

## 1. Background

The project has made clear progress on workflow orchestration, Discord intake, coding-team routing, artifact handoff, and release-pack automation.

Recent verified signal:

- Discord-style `/coder` intake can route into `coding_team_v0`
- generic app requests can complete end-to-end after runtime lane stabilization
- workflow artifacts, handoffs, QA, release, and preview outputs can all be emitted under one run
- orchestrator automated tests remain broadly healthy

However, the latest real run also shows a structural quality gap:

- the workflow can return `succeeded`
- QA can return `GO`
- but the generated product may still be placeholder-grade rather than user-usable

This is now the main product problem.

---

## 2. Current Progress Summary

### 2.1 Platform Progress

- Discord intake, routing, and workflow dispatch are functioning
- shared `coding_team_v0` workflow supports `generic_app`
- artifact scaffolding, handoff contracts, and release-pack generation are implemented
- preview deployment can publish a runnable output target
- runtime lane selection is now explicit enough to recover a stable coding path

### 2.2 Validation Progress

- a real Discord-to-coder ecommerce demo request completed `7/7` workflow steps
- `GO/No-Go` can be emitted as a machine-readable artifact
- orchestrator tests show high pass rate with a small number of remaining regressions

### 2.3 Engineering Maturity Signals

- contract-driven workflow design is already present
- observability and validation are stronger than in early-stage prototypes
- execution lane, provider behavior, and workflow metadata are increasingly auditable

---

## 3. Problem Statement

The system is currently stronger at producing a complete artifact set than at producing a complete product.

Observed gaps:

1. Product outputs remain too generic
   - PM artifacts often describe CRUD-style placeholders instead of domain-specific requirements
2. QA is too artifact-driven
   - a run can receive `GO` even when end-user functionality is still shallow
3. Preview fidelity is too weak
   - preview can point to a legacy or shared sandbox target rather than the intended product slice
4. Generated frontend/backend implementations are often scaffold-level
   - they satisfy schema and file-shape requirements without satisfying meaningful user journeys
5. Release semantics are too permissive
   - "workflow success" is still interpreted too close to "product success"

This creates a false-positive success mode:

- routing works
- workflow closes
- artifacts look complete
- actual user value is incomplete

---

## 4. Target State

The desired end state is:

1. A `/coder` request produces domain-specific planning artifacts
2. Implementation outputs represent real user-visible functionality, not scaffolds
3. Preview points to the correct runtime and product slice
4. QA verifies user journeys, not only artifact presence
5. `GO` means the product is demo-usable, not merely contract-complete

Definition of "demo-usable":

- primary user journey is visually present
- primary interaction path is executable
- key domain nouns appear in UI/API artifacts
- preview target matches the requested app type
- QA report includes journey-based evidence

---

## 5. Design Principles

### 5.1 Product Truth Over Artifact Completeness

Artifact presence is necessary but not sufficient.

Release gates must prefer:

- product fidelity
- journey completeness
- preview correctness
- domain-specific acceptance evidence

over:

- file count
- schema-only success

### 5.2 Domain-Specific Planning

The PM and architect stages must stop collapsing broad product requests into generic entity management.

Each recognized project type should carry domain templates for:

- user journeys
- domain vocabulary
- expected UI sections
- expected API surface
- demo-grade acceptance criteria

### 5.3 Truthful QA

QA should be allowed to say:

- `artifact_complete_but_product_shallow`
- `preview_mismatch`
- `journey_missing`
- `ui_not_domain_aligned`

without forcing a full workflow infrastructure failure.

### 5.4 Explicit Preview Semantics

Preview must declare:

- the exact product root used
- why that target root was selected
- whether the preview is domain-aligned or legacy-fallback

### 5.5 Stable Runtime Defaults

Infrastructure/runtime drift must not silently alter coding quality or user-visible outcomes.

---

## 6. Proposed Functional Enhancements

## FE-01 Product Fidelity Layer

Introduce a product-fidelity evaluator that runs after implementation and before final `GO`.

Responsibilities:

- inspect PM, FE, BE, preview, and QA artifacts together
- score domain alignment
- detect scaffold-only outputs
- detect mismatch between requested product and generated product

Minimum output:

- `product_fidelity_report.json`

Minimum result classes:

- `demo_usable`
- `artifact_complete_but_shallow`
- `preview_mismatch`
- `domain_misaligned`

## FE-02 Domain Acceptance Packs

Create domain-specific acceptance packs for common request classes:

- ecommerce demo
- CRM
- document/release management
- marketing site
- utility tool

Each pack should define:

- required user journeys
- expected UI components
- expected domain nouns
- expected backend behavior
- preview expectations

## FE-03 Preview Guardrail

Add a preview gate that fails or downgrades a run when:

- preview root points to a legacy sandbox unrelated to the request
- preview is static but request requires dynamic interactions
- preview target lacks required domain artifacts

## FE-04 QA Depth Upgrade

Upgrade QA from artifact-presence checks to journey checks.

Examples:

- ecommerce: browse product grid, open detail, add to cart, begin checkout
- CRM: list customer, open detail, create/edit record
- document hub: create document, assign identifier, inspect release history

## FE-05 Release Semantics Tightening

Change `GO/No-Go` logic so `GO` requires:

- artifact pack valid
- workflow succeeded
- strict canary passed
- product fidelity at or above threshold
- no preview mismatch
- no critical journey missing

---

## 7. Non-Functional Enhancements

## NFE-01 Runtime Configuration Hygiene

- eliminate ambiguous environment overrides for execution lane defaults
- ensure startup preflight rejects conflicting lane/model settings
- expose actual runtime lane/model in operator-facing summaries

## NFE-02 Typed Failure Quality

- fix provider/auth/model error normalization
- avoid collapsing actionable failures into `E_INTERNAL`

## NFE-03 Test Reliability

- keep orchestrator and worker-coder suites green
- add regression tests for preview payload defaults and fidelity gating

---

## 8. Milestones

### Milestone A: Truthful Success

Goal:

- a run cannot receive `GO` if the generated product is only scaffold-level

Includes:

- product fidelity report
- release gate integration
- QA wording cleanup

### Milestone B: Domain Usability

Goal:

- common request classes produce domain-shaped PM/QA artifacts and more realistic implementations

Includes:

- domain acceptance packs
- prompt and planner updates
- preview guardrails

### Milestone C: Stable Completion Quality

Goal:

- success rate remains high while false-positive success rate drops materially

Includes:

- regression coverage
- dashboard/report additions
- real Discord canary comparison before/after

---

## 9. Acceptance Criteria

This design is considered implemented when:

1. `GO` is blocked for scaffold-only product outputs
2. preview mismatch is machine-detected and surfaced
3. at least three project types have domain-specific acceptance packs
4. QA reports include journey-based evidence
5. release semantics distinguish workflow success from product success
6. runtime defaults are stable and auditable
7. orchestrator and worker-coder core test suites return green

---

## 10. Out of Scope

- full redesign of all agent prompts
- multi-tenant productization
- full visual-design system rebuild
- broad governance/program changes outside product completion quality

---

## 11. Key Deficiencies To Address First

The current highest-priority deficiencies are:

1. false-positive `GO` on shallow product output
2. domain-generic PM/QA artifacts
3. preview target mismatch risk
4. remaining automated test regressions
5. provider/runtime error misclassification

These should be treated as blocking quality issues, not polish.
