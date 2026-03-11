# OpenClaw Nexus Worker-Coding Design v4.2

## Date
2026-03-11

## Status
Draft for review - supersedes v4.1 Worker-Coding Design

## Type
Domain design addendum for worker-coding capability uplift

## Changelog from v4.1

| Section | Change |
|---------|--------|
| Section 7 Layer A | Added Context Boundaries and Context Envelope definition |
| Section 7 Layer B | Added State Isolation & Rollback as mandatory execution requirement |
| Section 7 (NEW Layer E) | Added Context Infrastructure Readiness layer with RAG interface reservation |
| Section 6 | Added context complexity classification per task class |
| Section 8 | Added Principle 6: Isolate Failure Attribution |
| Section 9.1 | Extended task-class metadata with context_envelope fields |
| Section 9.2 | Extended beta template with auto_verification_scripts and human_acceptance_criteria |
| Section 9.4 (NEW) | Context Infrastructure Interface Reservation |
| Section 12 | Updated explicit non-goals to include RAG implementation in current phase |
| Section 13 (NEW) | RAG Architectural Direction and Phasing Guidance |

---

## Why This Document

The project has completed the current worker-coder hardening baseline:

- guardrails landed
- release gate passed
- runtime path consistency was hardened
- worker lifecycle consistency was fixed
- structural decomposition completed

The next problem is no longer architecture survival.

The next problem is productized usefulness:

`can internal beta users reliably use coding through Nexus across multiple bounded task classes`

This document defines that next-stage sub-program, incorporating architectural and PM feedback on context management, state isolation, verification granularity, and RAG infrastructure phasing.

This is not a project-wide `v4.2` system design replacement. It is a worker-coding domain design addendum under the current project-level v4 baseline.

---

## 1. Problem Statement

Current worker-coding is technically available, but product success still depends on operator knowledge and controlled prompting.

This creates four gaps:

1. task-class coverage is not yet formally defined
2. first-use beta success is not yet treated as a governed capability target
3. validation evidence exists at workflow level, but not yet as a multi-class coding quality cohort
4. context acquisition for repo-aware task classes (FE Modify, Bug Fix) has no governed boundary or failure mode

If unaddressed, the system risks three bad outcomes:

- appearing usable in isolated demos but failing real beta onboarding
- overfitting to one scenario such as simple webpage generation
- context-dependent task classes failing silently due to unbounded or incorrect context injection

---

## 2. Design Goal

Promote worker-coding from:

`hardened execution component`

to:

`controlled beta-ready coding capability`

The capability must support a bounded but meaningful set of task classes without collapsing into one narrow demo flow. Context-dependent task classes must have explicit context boundaries even before a full retrieval infrastructure exists.

---

## 3. In Scope

- define controlled coding task classes
- standardize beta entry contracts for each class, including context boundaries
- define the execution-state-isolation target and rollout boundary as a dedicated adjacent workstream
- improve verification defaults and evidence quality with explicit verification tiers
- improve task-result readability for internal users
- create repeatable cohort validation for worker-coding
- tighten execution contracts only where needed for productized use
- define context infrastructure interface reservation for future RAG integration

---

## 4. Out of Scope

- full autonomous product-building
- open-ended internet-dependent coding
- broad provider marketplace support
- unbounded repo bootstrap / dependency installation
- replacing orchestrator governance with direct agent freedom
- dual-layer RAG implementation (reserved for post-cohort-validation phase)
- interactive refinement / multi-turn correction loops (reserved for post-validation phase)

---

## 5. North Star Mapping

This sub-program directly supports:

`Human Input -> Discord Gateway -> OpenClaw Orchestration -> Coding Team Workflow -> Artifacts`

Specifically, it improves the last three segments:

- task enters coding workflow with clearer task-class framing and context boundaries
- worker-coder executes with higher first-pass reliability under governed execution isolation
- artifacts and summaries become more usable for internal beta review

---

## 6. Controlled Task Classes

Worker-coding should be explicitly designed around these initial task classes. Each class now includes a context complexity rating that determines the required context management behavior.

### TC-01 FE Create

Context Complexity: `LOW`

Example:
- create a simple landing page
- create a small dashboard page in an existing app

Purpose:
- validate greenfield UI output

Context Requirement:
- minimal - project scaffolding hints and target path are sufficient
- no deep repo traversal required

### TC-02 FE Modify

Context Complexity: `MEDIUM-HIGH`

Example:
- update an existing page
- add a component
- refine layout, copy, or interaction

Purpose:
- validate real repo edit workflows

Context Requirement:
- target file(s) and immediate dependency graph must be identified before execution
- component hierarchy and import chain must be bounded
- failure to acquire sufficient context must block execution, not produce blind edits

### TC-03 BE Create

Context Complexity: `MEDIUM`

Example:
- add a small API route
- add a service function or controller behavior

Purpose:
- prevent frontend-only roadmap narrowing

Context Requirement:
- existing API patterns, data models, and routing conventions must be discoverable
- target module boundaries must be pre-identified

### TC-04 Bug Fix

Context Complexity: `HIGH`

Example:
- repair a scoped defect
- correct validation or state handling

Purpose:
- validate precision + verification quality

Context Requirement:
- defect localization requires understanding of call chain and state flow
- highest risk of incorrect context leading to "fix in wrong place" failures
- context acquisition failure must produce explicit diagnostic, not a guess

### TC-05 Artifact Completion

Context Complexity: `LOW-MEDIUM`

Example:
- fill expected workflow artifact gaps
- repair a broken handoff artifact bundle

Purpose:
- protect orchestrated workflow continuity

Context Requirement:
- artifact schema and expected structure must be provided in template
- limited repo context needed

---

## 7. Capability Layers

### Layer A: Beta Entry Contract

Each task class should have a stable entry template:

- task framing
- expected artifacts
- allowed target paths
- verification command guidance (see tiered verification below)
- failure summary expectations
- **context boundaries**: explicit definition of what context the worker may access
- **context envelope**: maximum files, maximum token budget, required dependency depth per task class

This is the product boundary between the user task and the coding engine. The context envelope is a hard contract - exceeding it must trigger a graceful refusal, not silent degradation.

Context Envelope Defaults (adjustable per template):

| Task Class | Max Files | Max Token Budget | Dependency Depth |
|-----------|-----------|-----------------|-----------------|
| FE Create | 5 | 8K | 0 (greenfield) |
| FE Modify | 15 | 24K | 2 (imports of imports) |
| BE Create | 10 | 16K | 1 (direct imports) |
| Bug Fix | 20 | 32K | 3 (call chain) |
| Artifact Completion | 5 | 8K | 0 |

### Layer B: Execution Reliability

Worker-coder must keep:

- scoped write control
- deterministic artifact scaffold
- static checks
- verification command execution
- retry and failure memory
- single-finalization lifecycle behavior
- **state isolation via Git-level branching**: every task execution must operate on an isolated temporary branch; verification failure or timeout must guarantee the trunk is not polluted
- **rollback contract**: failed executions must leave the repository in a clean state; partial writes are governance violations

State Isolation Requirements:

1. task start: checkout a temporary branch from the target ref
2. task execution: all writes happen on the temporary branch only
3. verification pass: merge to target branch through governed merge path
4. verification fail or timeout: abandon temporary branch, generate failure summary, trunk remains untouched
5. no partial merge is permitted under any circumstance

### Layer C: Beta Reviewability

Outputs must be understandable to PM/QA/internal users:

- short summary of what changed
- files changed with structured diff/patch output (not raw log)
- verification outcome with tier classification
- explicit failure cause when unsuccessful, including context acquisition failures
- structured artifacts for inspection

Diff Output Standard:
- every successful execution must produce a machine-readable and human-scannable diff summary
- diff must show: files added, files modified, files deleted
- for modified files: section-level change description (not line-by-line noise)
- QA must be able to review what changed without opening an IDE

### Layer D: Cohort Validation

The system needs a beta matrix proving capability by task class, not just one successful workflow sample.

### Layer E: Context Infrastructure Readiness (Architectural Reservation Only)

This layer defines the interface contract that a future context retrieval system (including but not limited to dual-layer RAG) must satisfy when integrated. **This layer does not authorize implementation of RAG in the current phase.**

The interface reservation exists so that:
- current Layer A context envelopes can be manually populated today
- future automated context retrieval can satisfy the same contract without changing downstream layers
- cohort validation data can measure context-related failure rates to justify retrieval investment

Required Interface (for future implementation):

```
ContextRequest {
  task_class: string
  target_paths: string[]
  max_files: int
  max_tokens: int
  dependency_depth: int
}

ContextResponse {
  status: "complete" | "partial" | "failed"
  files: FileContext[]
  token_usage: int
  missing_context: string[]  // what could not be retrieved
  confidence: float
}
```

Current Phase Behavior:
- context is provided manually via beta template or operator input
- context envelope limits are enforced
- context acquisition failures are logged as structured events for cohort analysis

---

## 8. Design Principles

### Principle 1: Avoid Scenario Narrowing

No single scenario may define the roadmap.

`build a webpage` is a sample, not the whole product.

### Principle 2: Productize the Contract, Not Just the Engine

A stronger worker is insufficient if users still need tribal knowledge to use it correctly.

### Principle 3: Preserve Governance

Coding capability uplift must not bypass target paths, verification, artifact contracts, or release evidence.

### Principle 4: Optimize for First-Use Success

Internal beta users should be able to submit a bounded coding task and receive a reviewable outcome without manual system debugging.

### Principle 5: Measure by Task Class

Quality must be assessed separately for FE create, FE modify, BE create, bug fix, and artifact completion.

### Principle 6: Isolate Failure Attribution (NEW)

When a task fails, the system must be able to distinguish between:
- coding logic failure (worker wrote wrong code)
- context failure (worker had wrong or insufficient context)
- verification failure (code was correct but verification was misconfigured)
- infrastructure failure (timeout, resource limit, service unavailability)

This principle exists to prevent RAG introduction from polluting failure analysis, and to ensure that current-phase cohort data cleanly separates these failure modes.

---

## 9. Proposed Interfaces

### 9.1 Task-Class Metadata (Extended)

Each coding task should include:

```json
{
  "task_class": "fe_create|fe_modify|be_create|bug_fix|artifact_completion",
  "beta_template_id": "string|null",
  "context_envelope": {
    "max_files": "int",
    "max_tokens": "int",
    "dependency_depth": "int",
    "context_source": "manual|template|automated"
  }
}
```

Purpose:
- improve routing/reporting clarity
- make validation cohort results comparable
- avoid free-text-only classification downstream
- track context acquisition method for future RAG comparison

### 9.2 Beta Template Registry (Extended)

Each template should define:

- task class
- expected artifacts
- target path hints
- output summary expectations
- context envelope defaults for this task class
- **auto_verification_scripts**: ordered list of automated checks that must pass
  - `lint`: static analysis / syntax check (must pass - failure triggers internal retry)
  - `type_check`: type-level correctness where applicable
  - `unit_test`: specified test commands to execute
  - `build`: build/compile check
- **human_acceptance_criteria**: what PM/QA should evaluate after automated checks pass
  - UI review criteria (for FE classes)
  - API contract review criteria (for BE classes)
  - Regression scope (for Bug Fix class)

Verification Level Declaration:
- each template must declare which auto_verification levels are required vs. optional
- a template that only has lint-level auto verification cannot claim "verification pass" - it must be labeled as "lint-verified only"

### 9.3 Cohort Validation Summary (Extended)

Validation should aggregate by task class:

- runs attempted
- success count
- verification pass count (broken down by verification tier)
- common failure reasons (categorized by Principle 6 attribution)
- **first-pass verification rate**: passed all auto checks on first attempt without retry
- **human modification rate**: binary (did human need to modify?) plus severity tier (trivial / significant / rewrite)
- **regression rate**: for Bug Fix class, frequency of new defects introduced
- context acquisition failure count (for future RAG justification)

### 9.4 Context Infrastructure Interface Reservation (NEW)

This section reserves the integration point for future context retrieval infrastructure.

Current-phase contract:
- context is provided as part of the beta template or operator input
- context envelope limits are enforced by Layer A
- context-related failures are tagged distinctly in cohort data

Future-phase contract (not authorized for implementation):
- a context retrieval service satisfying the ContextRequest/ContextResponse interface (defined in Layer E) may be integrated
- integration must not change Layer B execution behavior or Layer C output format
- integration must pass through its own validation cohort before being enabled for production task classes

Architectural Direction for Future RAG:
- Layer 1 (Structural): repo-level indexing - file tree, module dependency graph, interface signatures
- Layer 2 (Semantic): code-level retrieval - function bodies, type definitions, usage patterns
- Both layers feed into a unified ContextResponse that respects the task-class context envelope

This direction is documented for planning coherence only. Implementation requires separate design approval after cohort validation data demonstrates context-related failure rates that justify the investment.

---

## 10. Delivery Strategy

The next-stage sequence should be:

1. define the task-class contract including context envelopes and failure attribution semantics
2. run a dedicated execution-state-isolation workstream without coupling it to contract-definition closure
3. define beta templates with tiered verification
4. create cohort validation for multiple task classes with failure attribution
5. improve result readability, diff output, and failure reporting where needed
6. analyze cohort data for context-related failure rates
7. if context failure rates justify it, proceed to RAG system design as a separate governed program
8. expand only after evidence is stable

---

## 11. Success Criteria

This sub-program is successful when:

- worker-coding is governed as a product capability, not just a worker module
- at least four controlled task classes are validated
- beta task entry becomes easier and more repeatable
- success/failure boundaries are observable by class and by failure attribution category
- if the execution-state-isolation workstream is included in this phase, it is enforced and verified for approved task executions
- verification outcomes are tiered and labeled accurately
- context-related failure rates are measurable from cohort data
- no new subsystem sprawl is introduced

---

## 12. Explicit Non-Goals

This document does not authorize:

- general autonomous engineering expansion
- replacing the existing workflow with direct LLM execution
- massive provider/runtime diversification
- broad open-ended code generation without verification
- dual-layer RAG implementation (interface is reserved; implementation requires separate approval)
- interactive refinement / multi-turn user correction loops (deferred to post-validation phase)
- task chaining or multi-step orchestrated execution (interface should be noted in non-scope; implementation deferred)

---

## 13. RAG Architectural Direction and Phasing Guidance (NEW)

This section documents the architectural thinking for a future dual-layer RAG system. It is included for planning coherence and to ensure current-phase design decisions do not foreclose the integration path. **Nothing in this section authorizes implementation work.**

### 13.1 Why Dual-Layer

Single-layer retrieval (e.g., naive vector search over code chunks) fails in two ways:
- it misses structural relationships (module A depends on module B depends on module C)
- it retrieves syntactically similar but semantically irrelevant code

A dual-layer approach addresses this:

**Layer 1 - Structural Index (Coarse)**
- indexes: file tree, module dependency graph, export/import relationships, interface signatures, API route maps
- query: "given target path X, what are the structurally related files within dependency depth N?"
- output: a bounded set of file paths and their structural roles
- update frequency: on commit or periodic rebuild

**Layer 2 - Semantic Index (Fine)**
- indexes: function bodies, type definitions, docstrings, usage patterns, test coverage references
- query: "given the structural file set from Layer 1, retrieve the specific code segments relevant to task description Y"
- output: code snippets with source location and relevance score
- update frequency: on commit or periodic rebuild

### 13.2 Integration Contract

When implemented, RAG must:
- satisfy the ContextRequest/ContextResponse interface defined in Layer E
- respect the context envelope limits defined per task class
- produce a confidence score that Layer A can use to decide whether context is sufficient
- log retrieval metrics (latency, recall proxy, token usage) for cohort analysis
- degrade gracefully: if RAG is unavailable, task execution falls back to manual context (same as current phase)

### 13.3 Phasing Decision Criteria

RAG system design should proceed when:
- at least one full cohort validation cycle (WC-NEXT-04) has completed
- context-related failure rate across TC-02, TC-03, and TC-04 exceeds 30% of total failures
- or operator/beta-user feedback consistently identifies "wrong context" as the dominant pain point
- and the coding contract (WC-NEXT-01) is stable enough that RAG integration will not require contract redesign

If these criteria are not met, RAG remains deferred and manual context provision continues.

### 13.4 Governance

RAG implementation, when authorized, must:
- be classified as a separate governed program (likely Type A, same pipeline node)
- have its own cohort validation before being enabled for production task classes
- not bypass any existing Layer B execution reliability requirements
- not introduce new provider dependencies without explicit approval
