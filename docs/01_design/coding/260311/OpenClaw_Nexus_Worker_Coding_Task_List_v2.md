# OpenClaw Nexus Worker-Coding Task List v2

- Date: 2026-03-11
- Status: DRAFT FOR REVIEW
- Scope: next-stage worker-coding capability uplift after M9 closeout and next-stage hardening completion
- Supersedes: Worker-Coding Task List v1 (2026-03-11)

---

## 1. Decision

Current worker-coding uplift is classified as:

`MAINLINE TYPE A`

Interpretation:

- this is not a side branch
- this is not a new subsystem program
- this is the next productization layer for the existing North Star coding workflow

---

## 2. Changelog from v1

| Task | Change |
|------|--------|
| WC-NEXT-01 | Refined task-class contract around context envelope and failure attribution only |
| WC-NEXT-02 (NEW) | Split execution state isolation and rollback hardening into a dedicated workstream |
| WC-NEXT-03 | Beta template registry remains productization-focused, no longer overloaded with execution isolation |
| WC-NEXT-04 | Multi-class cohort validation now depends on both contract and execution readiness |
| WC-NEXT-05 | Added structured diff/patch output requirement; added verification tier labeling |
| WC-NEXT-06 | Split metrics into currently observable vs later instrumentation-required |
| WC-NEXT-07 (NEW) | Context Failure Analysis and RAG Decision Gate |

---

## 3. Task List

### P0

#### WC-NEXT-01: Task-Class Contract Definition

**Status**

`IN PROGRESS`

**Task Name**

`WC-NEXT-01 Task-Class Contract Definition`

**Pipeline Node**

`OpenClaw Orchestration -> Coding Team Workflow`

**Task Type**

`Type A`

**Upstream Dependency**

- M9 guardrails are landed
- release gate is passing
- worker structural governance is complete

**Goal**

Define the controlled coding task classes and make them explicit in the worker-coding contract. Establish context boundaries and failure attribution semantics per task class.

**Deliverables**

- task-class taxonomy with context complexity rating per class
- payload extension proposal for `task_class` including `context_envelope` fields
- reporting field proposal for class-aware evidence with failure attribution categories
- short contract note for orchestrator and worker ownership boundaries
- context envelope defaults per task class (max files, max tokens, dependency depth)

**Non-Scope Declaration**

- no model-policy redesign
- no new provider work
- no user-facing UI redesign
- no RAG implementation (context envelopes are manually populated)
- no task chaining or multi-step orchestration
- no Git/worktree execution redesign inside this task

**Acceptance Criteria**

- task classes are explicit and documented with context complexity ratings
- task-class metadata includes context_envelope and context_source fields
- context envelope limits are enforceable (exceed = graceful refusal)
- task-class metadata does not break current workflow compatibility
- PM/QA can distinguish coding evidence by task class and by failure attribution category

**LLM Role**

`none`

**Current Progress**

- initial compatible contract note landed under `docs/03_feature_development/2026-03-11_worker_coding_task_contract_note.md`
- `coding.delegate` now accepts optional `task_class`, `beta_template_id`, and `context_envelope`
- success/failure diagnostics now persist normalized `task_contract`
- failure summaries and coding failure memory now persist `failure_attribution`
- compatibility preserved: existing flows continue unchanged when new fields are omitted
- authoritative schema assets landed:
  - `orchestrator/contracts/worker_coding_task_contract.schema.json`
  - `orchestrator/contracts/worker_coding_beta_template_registry.schema.json`
- initial beta template registry landed under `configs/registry/worker_coding_beta_templates.json`
- validation command `npm.cmd --prefix orchestrator run validate:worker_coding_contract` added and passing
- template default injection now lands in `orchestrator/src/domain/workflow_step_builder.js`
- builder-level coverage added in `orchestrator/test/worker_coding_templates.test.js`

---

#### WC-NEXT-02: Execution State Isolation and Rollback Hardening

**Status**

`IN PROGRESS`

**Task Name**

`WC-NEXT-02 Execution State Isolation and Rollback Hardening`

**Pipeline Node**

`OpenClaw Orchestration -> Coding Team Workflow`

**Task Type**

`Type A`

**Upstream Dependency**

- task-class contract exists (WC-NEXT-01)
- current worker-coder guardrails and release gate remain green

**Goal**

Implement execution isolation as a dedicated architecture workstream so coding tasks can fail safely without polluting the working branch.

**Deliverables**

- execution isolation design note:
  - Git/worktree/branch strategy decision
  - rollback contract
  - partial-write prohibition rule
- implementation plan and integration impact note
- validation approach proving failure and timeout do not pollute the target branch

**Non-Scope Declaration**

- no provider expansion
- no merge automation beyond the approved governed path
- no contract redesign for task-class metadata except what isolation strictly requires

**Acceptance Criteria**

- execution isolation is defined as an independent workstream with explicit architecture choice
- rollback behavior is testable
- timeout/failure trunk-pollution risk is materially reduced
- this work does not block WC-NEXT-03 template progress unless a hard dependency is proven

**LLM Role**

`none`

---

#### WC-NEXT-03: Beta Template Registry

**Status**

`IN PROGRESS`

**Task Name**

`WC-NEXT-03 Beta Template Registry`

**Pipeline Node**

`Human Input -> OpenClaw Orchestration -> Coding Team Workflow`

**Task Type**

`Type A`

**Upstream Dependency**

- task-class definition exists (WC-NEXT-01)

**Goal**

Create a governed registry of approved internal beta coding templates so first-use success does not depend on tribal knowledge. Templates must include tiered verification definitions and context envelope defaults.

**Deliverables**

- template registry format
- initial templates for:
  - FE create
  - FE modify
  - BE create
  - bug fix
- documentation for required fields:
  - expected artifacts
  - target path hints
  - context envelope defaults (max files, max tokens, dependency depth)
  - auto_verification_scripts (ordered by tier):
    - `lint` (mandatory for all classes - failure triggers internal retry)
    - `type_check` (mandatory for TypeScript/typed classes)
    - `unit_test` (where test commands are specified)
    - `build` (where build step exists)
  - human_acceptance_criteria:
    - UI review criteria (FE classes)
    - API contract review criteria (BE classes)
    - regression scope definition (Bug Fix class)
  - verification level declaration (what tier this template achieves)
  - summary expectations

**Non-Scope Declaration**

- no broad prompt-library system
- no open-ended freeform agent marketplace
- no interactive refinement loops

**Acceptance Criteria**

- internal testers can start a bounded coding task from a documented template
- template fields align with current workflow contracts including context envelope
- each template maps to one task class
- each template declares its verification level explicitly
- a template with only lint-level verification is labeled as "lint-verified only" (not "verified")

**LLM Role**

`none`

**Current Progress**

- template-declared verification tiers are now translated into structured `verification_plan` payloads at orchestrator build time
- worker-coder now executes ordered verification plans while keeping legacy `verification_command` compatibility
- prompt contract artifacts now record both `verification_command` and `verification_plan`
- first result-quality hardening pass is landed:
  - achieved tiers are persisted in verification diagnostics
  - unresolved tiers remain distinguishable from executed tiers
  - cohort reports now reflect actual achieved verification evidence instead of a generic pass/fail label
- repo-aware verification source is now wired for `sandbox/crm_site` via local `package.json` scripts
- latest controlled cohort after real verification-source wiring produced a stricter signal:
  - `0 pass / 4 fail / 0 partial`
  - `fe_create`, `fe_modify`, and `bug_fix` currently fail under `verification_failure`
  - `be_create` currently fails under `coding_logic_failure`
- current quality conclusion:
  - earlier `partial` results were optimistic because verification depth was not truly enforced in live runtime
  - after true enforcement, current beta cohort is not yet ready for pass claims

---

### P1

#### WC-NEXT-04: Multi-Class Cohort Validation

**Status**

`TODO`

**Task Name**

`WC-NEXT-04 Multi-Class Cohort Validation`

**Pipeline Node**

`Coding Team Workflow -> Artifacts`

**Task Type**

`Type A`

**Upstream Dependency**

- task-class contract exists (WC-NEXT-01)
- beta templates exist (WC-NEXT-03)
- if execution isolation is in current phase scope, its minimum validation is available (WC-NEXT-02)

**Goal**

Replace single-scenario confidence with a controlled validation cohort covering multiple coding task classes. Cohort data must support failure attribution and context failure measurement.

**Deliverables**

- one cohort validation plan with sample tasks per class
- one machine-readable summary artifact grouped by task class
- curated sample tasks covering at least:
  - FE create
  - FE modify
  - BE create
  - bug fix
- failure attribution tagging for every cohort run:
  - `coding_logic_failure`: worker wrote incorrect code
  - `context_failure`: worker had wrong or insufficient context
  - `verification_failure`: code was correct but verification was misconfigured
  - `infrastructure_failure`: timeout, resource limit, service unavailability
- context failure tracking:
  - count of runs where context envelope was insufficient
  - count of runs where context was technically within envelope but semantically wrong
  - operator notes on what additional context would have been needed

**Non-Scope Declaration**

- no broad benchmark framework
- no public leaderboard
- no automated context retrieval (manual context only)

**Acceptance Criteria**

- validation can show success/failure by task class
- failure reasons are categorized by attribution type
- context-related failure rates are quantified per task class
- common failure reasons are visible and actionable
- project does not rely on one demo scenario for readiness claims
- if execution isolation is enabled for the cohort, no trunk pollution is observed

**LLM Role**

`none`

**Current Preparation**

- initial cohort task matrix landed under `docs/03_feature_development/2026-03-11_worker_coding_cohort_task_matrix.md`
- first cohort set now covers:
  - `C-FE-01` `fe_create`
  - `C-FE-02` `fe_modify`
  - `C-BE-01` `be_create`
  - `C-BUG-01` `bug_fix`
- cohort result format note landed under `docs/03_feature_development/2026-03-11_worker_coding_cohort_result_format.md`
- authority schema landed under `orchestrator/contracts/worker_coding_cohort_result.schema.json`
- validation command `npm.cmd --prefix orchestrator run validate:worker_coding_cohort_result` added and passing
- cohort execution plan landed under `configs/registry/worker_coding_cohort_plan_v1.json`
- authority schema landed under `orchestrator/contracts/worker_coding_cohort_plan.schema.json`
- validation command `npm.cmd --prefix orchestrator run validate:worker_coding_cohort_plan` added and passing
- executable cohort runner landed under `orchestrator/scripts/run_worker_coding_cohort.js`
- validation command `npm.cmd --prefix orchestrator run validate:worker_coding_cohort_execute` added and passing
- first controlled cohort cycle completed on 2026-03-11:
  - total runs: `4`
  - pass: `0`
  - fail: `0`
  - partial: `4`
- first cohort artifact written to `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-17-51-917Z/worker_coding_cohort_result.json`
- current gap is not workflow failure; it is verification-tier gap:
  - achieved tier observed: `syntax_check`
  - target tiers remain `lint + build`, `lint + type_check + build`, `lint + unit_test`
- latest enforced live cohort signal is now stricter and should be treated as authoritative:
  - artifact: `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-57-01-157Z/worker_coding_cohort_result.json`
  - total runs: `4`
  - pass: `0`
  - fail: `4`
  - partial: `0`
  - `fe_create`, `fe_modify`, `bug_fix` => `verification_failure`
  - `be_create` => `coding_logic_failure`
- residual issues for follow-up:
  - inspect why FE cohort cases fail before any achieved verification tier is recorded
  - inspect why `be_create` still fails as coding logic rather than verification-only
  - keep using the latest fail artifact, not earlier partial artifacts, as readiness evidence

---

#### WC-NEXT-05: User-Facing Result Quality Hardening

**Status**

`IN PROGRESS`

**Task Name**

`WC-NEXT-05 User-Facing Result Quality Hardening`

**Pipeline Node**

`Coding Team Workflow -> Artifacts -> Human Review`

**Task Type**

`Type A`

**Upstream Dependency**

- cohort validation reveals current result-quality gaps (WC-NEXT-04)

**Goal**

Improve the readability and actionability of coding outcomes for internal beta users and reviewers. Standardize diff output and verification tier labeling.

**Deliverables**

- result summary quality rubric
- failure-summary quality improvements with failure attribution labels
- structured diff/patch output standard:
  - files added / modified / deleted
  - section-level change description for modified files
  - QA-readable without IDE
- verification tier labeling in all output summaries:
  - which auto_verification_scripts passed/failed
  - human_acceptance_criteria checklist status
  - explicit label: "lint-verified only" vs "unit-test verified" vs "build verified"
- artifact completeness checks where current outputs are noisy or ambiguous
- context acquisition failure diagnostic in failure summaries (what context was missing)

**Non-Scope Declaration**

- no new chat product surface
- no cosmetic-only formatting work without evidence benefit

**Acceptance Criteria**

- successful runs produce short, reviewable summaries with structured diff
- failed runs state what blocked progress, what was attempted, and which failure attribution category applies
- verification tier is explicit in every output (no ambiguous "verified" claims)
- QA can inspect outcomes without raw log archaeology
- context-related failures are distinguishable from coding logic failures in output

**LLM Role**

`none`

**Current Progress**

- `verification_plan` execution is landed in orchestrator and worker runtime
- `sandbox/crm_site` now has repo-aware verification sources via local `package.json` scripts
- container-safe task-class authority is now loaded from `configs/registry/worker_coding_task_classes.json`
- current truthful live signal is negative but useful:
  - real verification enforcement no longer reports optimistic `partial` readiness
  - latest cohort result is `4 fail / 0 partial / 0 pass`
- residual issues recorded for next session:
  - FE path currently surfaces as `verification_failure`
  - BE path currently surfaces as `coding_logic_failure`

---

### P2

#### WC-NEXT-06: Controlled Beta Operations Metrics

**Status**

`TODO`

**Task Name**

`WC-NEXT-06 Controlled Beta Operations Metrics`

**Pipeline Node**

`Human Input -> Coding Team Workflow -> Artifacts`

**Task Type**

`Type B`

**Upstream Dependency**

- multi-class cohort validation exists (WC-NEXT-04)

**Goal**

Track whether worker-coding is becoming more usable for internal beta, not just more technically correct. Metrics must be operationally actionable.

**Deliverables**

- metric definitions split into phases:
  - **currently observable**
    - first-pass verification rate by task class
    - context failure rate by task class
    - verification pass rate by tier
    - dominant failure categories by attribution type
  - **phase-2 instrumentation required**
    - human modification rate: binary + severity tier
    - regression rate for Bug Fix class
- one short reporting note template for weekly review

**Non-Scope Declaration**

- no dashboard platform project
- no analytics subsystem expansion

**Acceptance Criteria**

- weekly beta quality can be discussed with evidence using quantified currently observable metrics
- product and engineering can prioritize using the same metric definitions
- any phase-2 metrics are clearly marked as manual-observation or future instrumentation items
- context failure rate is visible and trackable over time (feeds RAG decision gate)

**LLM Role**

`none`

---

#### WC-NEXT-07: Context Failure Analysis and RAG Decision Gate (NEW)

**Status**

`TODO`

**Task Name**

`WC-NEXT-07 Context Failure Analysis and RAG Decision Gate`

**Pipeline Node**

`Coding Team Workflow -> Artifacts -> Planning`

**Task Type**

`Type B`

**Upstream Dependency**

- at least one full cohort validation cycle completed (WC-NEXT-04)
- operations metrics are being collected (WC-NEXT-06)

**Goal**

Analyze cohort validation data to determine whether context-related failures justify investing in automated context retrieval (dual-layer RAG). Produce an evidence-based go/no-go recommendation.

**Deliverables**

- context failure analysis report:
  - context failure rate by task class
  - breakdown: insufficient context vs. wrong context vs. context exceeds envelope
  - operator-reported "what was missing" aggregation
  - estimated effort to resolve top context failures manually vs. via automation
- RAG decision recommendation:
  - go: context failures exceed 30% of total failures across TC-02/TC-03/TC-04 combined, or operator feedback consistently identifies context as dominant pain point
  - no-go: context failures are below threshold, manual provision is sufficient for current beta scale
  - defer: insufficient data, extend cohort validation
- if go: initial requirements document for dual-layer RAG system design (scope, interfaces, governance classification)

**Non-Scope Declaration**

- no RAG implementation
- no retrieval infrastructure build
- no index construction

**Acceptance Criteria**

- decision is data-driven, not assumption-driven
- recommendation is actionable: proceed to RAG design, continue manual, or extend validation
- if "go", the requirements document satisfies governance checklist guardrails
- PM, Architect, and QA can review the recommendation using shared evidence

**LLM Role**

`none`

---

## 4. Recommended Order

1. `WC-NEXT-01` Task-Class Contract Definition
2. `WC-NEXT-02` Execution State Isolation and Rollback Hardening
3. `WC-NEXT-03` Beta Template Registry
4. `WC-NEXT-04` Multi-Class Cohort Validation
5. `WC-NEXT-05` User-Facing Result Quality Hardening
6. `WC-NEXT-06` Controlled Beta Operations Metrics
7. `WC-NEXT-07` Context Failure Analysis and RAG Decision Gate

Reasoning:

- define the product boundary first
- separate execution architecture hardening from contract-definition scope
- reduce first-use ambiguity next, with explicit verification tiers
- validate across classes after contract and minimum execution safety are clear
- improve user-facing quality using evidence rather than guesswork
- add lightweight operational metrics after the workflow shape is stable
- make the RAG investment decision using real data, not architectural intuition

---

## 5. Exit Standard

This sub-program should be considered ready for closeout review when:

- worker-coding supports a governed multi-class beta cohort
- first-use task entry is documented and repeatable
- evidence is no longer anchored to one narrow scenario
- PM, QA, and Architect can review capability quality using shared artifacts
- if execution isolation is included in this phase, it is enforced for approved task executions
- failure attribution is clean enough to support infrastructure investment decisions
- context failure rates are measured and a RAG decision recommendation exists
