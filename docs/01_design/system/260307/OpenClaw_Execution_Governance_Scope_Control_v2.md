# Execution Governance & Scope Control v2
## OpenClaw Nexus Project Governance Specification
## Date: 2026-03-07
## Supersedes: docs/01_design/system/260306/OpenClaw_Execution_Governance_Scope_Control.md

---

## Changelog from v1

| Section | Change |
|---------|--------|
| Section 3 | Added Principle 6: Structural Entropy Governance |
| Section 6 | Updated DoD to include internal layer compliance |
| Section 8 | Added complexity budget enforcement |
| Section 11 | Added vnext completion definition and route deprecation protocol |
| Section 12 | Added module complexity budget table |
| Section 13 | Added Architect output quality standard |

---

## 1. Purpose

This document defines the execution governance system for the OpenClaw Nexus project.

Its purpose is to prevent:
- architectural drift
- uncontrolled scope expansion
- premature development of downstream components before upstream dependencies are completed
- structural entropy accumulation inside existing modules

This governance layer ensures:
- stable architectural evolution
- controlled development sequencing
- high signal-to-noise engineering progress
- alignment with the North Star execution pipeline
- maintainable internal module structure

---

## 2. North Star Execution Path

All development activity must ultimately support the primary operational pipeline.

```
Human Input
↓
Discord Gateway (adapter layer)
↓
Brain Router + Policy Layer
↓
TaskEnvelope Normalization
↓
OpenClaw Orchestration (4-layer internal structure)
↓
Coding Team Workflow (Architect-hardened)
↓
Artifacts (docs / interfaces / code / reports)
```

Any development work must prove that it shortens, stabilizes, or enables this pipeline.

If the work cannot clearly map to this pipeline, it must be categorized as Exploratory Work (Backlog).

---

## 3. Governance Principles

### Principle 1 — Upstream Completion Rule

No downstream component may begin implementation until its upstream dependency is fully completed and validated.

### Principle 2 — North Star Alignment

All tasks must demonstrate direct alignment with the North Star pipeline.

Allowed justification examples:
- Enables routing accuracy
- Enables workflow orchestration
- Enables agent role handoff
- Enables artifact delivery

Rejected justification examples:
- "Useful in the future"
- "Nice to have"
- "Makes the system more flexible"
- "General framework improvement"

### Principle 3 — Controlled Expansion

New subsystems may only be introduced after the current pipeline stage reaches Definition of Done.

Prohibited expansions (until Coding Team workflow is fully hardened):
- New agent teams (quant, ecommerce, research)
- New orchestration layers
- New UI systems

### Principle 4 — Contract-Based Work

All modules must operate on explicit contracts.

Contracts must define: input schema, output schema, expected artifacts, validation rules, failure conditions.

No module may rely on implicit assumptions, undocumented message formats, or free-text interfaces.

### Principle 5 — Minimal Execution Surface

At each stage, implement only the minimal functionality required to support the North Star pipeline.

Feature expansion occurs only after the minimal pipeline functions reliably.

### Principle 6 — Structural Entropy Governance (NEW)

Internal module structure is subject to the same governance discipline as feature scope.

Rules:
1. No source file may exceed its complexity budget (see Section 12) without an architectural review.
2. New code added to a module that is already at budget must first be placed in a correctly-layered sub-module.
3. Cross-layer imports are architectural violations — Layer 1 must not import from Layer 4 directly; Layer 2 must not call raw database.
4. Every quarter (or at every milestone boundary), the module complexity budget table must be reviewed and violations tracked.

Structural debt is not automatically deferred — it must be explicitly tracked as a Type A task if it blocks testing, deployment, or new feature work.

---

## 4. Task Classification System

### Type A — Critical Path Tasks

Tasks required to enable the North Star pipeline.

Examples:
- Brain Router schema
- TaskEnvelope definition
- Workflow Planner
- Coding Team contracts
- Artifact packaging
- Orchestrator internal decomposition (new)
- Architect Engineer hardening (new)
- Brain Router policy layer (new)

Highest priority.

### Type B — Enhancement Tasks

Tasks that improve quality but do not block execution.

Examples:
- dashboards
- logging improvements
- UI polish
- artifact browsing tools
- Memory/Context Layer stub

Allowed only after Type A tasks complete.

### Type C — Exploratory Tasks

Tasks unrelated to the current pipeline stage.

Examples:
- new agent ecosystems
- ecommerce assistant
- short video generator
- autonomous learning system
- advanced memory systems

Moved to Backlog until the current pipeline stage stabilizes.

---

## 5. Task Approval Requirements

Before work begins, each task must include:

**Task Name** — short identifier
**Pipeline Node** — which North Star node this supports
**Task Type** — A / B / C
**Upstream Dependency** — what must exist first
**Deliverables** — exact artifacts expected
**Non-Scope Declaration** — what this task will not implement
**Acceptance Criteria** — when the task is considered complete

---

## 6. Definition of Done (DoD)

A module is considered complete only when all criteria below are satisfied:

1. Input schema defined
2. Output schema defined
3. Contract documentation exists
4. Integration tests exist
5. Error conditions defined
6. Downstream module compatibility verified
7. Module is in the correct architectural layer (NEW)
8. Module does not exceed its complexity budget (NEW)

Partial functionality does not count as completion.

---

## 7. Definition of Not Done (DoND)

A module is not complete if any of the following remain:
- outputs rely on free text
- schema not validated
- failure cases undefined
- integration tests missing
- downstream module blocked
- module placed in wrong architectural layer (NEW)
- module exceeds complexity budget without a decomposition plan (NEW)

---

## 8. Change Control Process

Scope expansion requires formal approval.

A Change Request must include:
1. Justification for change
2. Impact on North Star pipeline
3. Affected modules
4. Risk assessment
5. Layer impact (does this require cross-layer changes?) (NEW)
6. Complexity budget impact (does this push a module over its budget?) (NEW)

Changes are approved only if they improve the current pipeline stage.

---

## 9. Anti-Divergence Mechanism

To prevent project drift, the following rule is enforced:

If a task cannot map directly to the North Star pipeline, it is automatically moved to backlog.

This rule prevents engineers from implementing features based on intuition or future speculation.

---

## 10. Role Boundary Enforcement

### Product Manager
Allowed: define problem scope, acceptance criteria, milestones
Not Allowed: defining architecture, choosing frameworks, modifying system boundaries

### Architect
Allowed: module boundaries, system interfaces, dependency design, ADRs
Not Allowed: expanding product scope, introducing unrelated subsystems, writing implementation code

### Engineering Roles
Allowed: implementation within defined contracts, reporting technical risks, improving internal code structure within the layer boundary
Not Allowed: altering architecture, introducing new system domains, expanding project scope

---

## 11. Governance Enforcement Mechanisms

**Workflow Locks** — Downstream modules remain locked until upstream modules satisfy DoD.

**Contract Validation** — All outputs validated against schema before acceptance.

**Task Review** — All Type A tasks require architectural review before execution.

**Change Requests** — All scope expansion requires formal approval.

**Complexity Budget Enforcement (NEW)** — See Section 12. Violations are tracked and must be resolved as Type A tasks within the same milestone they are detected.

**vnext Completion Definition (NEW):**
The vnext refactoring phase is complete when:
- `index.js` contains no Discord.js imports
- `index.js` contains no raw SQL
- `index.js` contains no inline LLM calls
- `index.js` line count is ≤ 800
- All business logic lives in `src/vnext/` (service layer) or `src/` domain modules

**Route Deprecation Protocol (NEW):**
When a route is identified as deprecated:
1. Document it in a Route Audit record
2. Add `X-Deprecated: true` header and warning log
3. Confirm zero usage in integration tests
4. Remove the route
5. Update the Route Audit record to `removed`
No deprecated route may be left in the codebase for more than one milestone.

**Architect Output Quality Gate (NEW):**
The `arch_design` step output is not accepted unless:
- `plan/interfaces.md` exists and contains at least 1 interface definition
- `handoff/architect_to_impl.json` contains a non-empty `decisions` array with ADR IDs
- `risk/risk_report.json` contains at least 1 risk entry
A workflow that passes `arch_design` without these will be considered a governance violation.

---

## 12. Module Complexity Budget (NEW)

| Module | Layer | Max Lines | Action if Exceeded |
|--------|-------|-----------|-------------------|
| `src/index.js` | 1 (Transport) | 800 | Type A task: decompose |
| `src/workflow_engine.js` | 3 (Domain) | 600 | Type A task: decompose |
| `src/vnext/*.js` (per file) | 2 (Service) | 300 | Extract sub-module |
| `src/*.js` (other, per file) | 3 (Domain) | 500 | Extract sub-module |
| `src/adapters/*.js` (per file) | 1 (Transport) | 400 | Extract sub-module |
| `src/data/*.js` (per file) | 4 (Infra) | 250 | Split by entity |

Review process:
- Reviewed at every milestone boundary
- Violations are automatically Type A tasks for the next milestone
- No new feature work may begin in a module that exceeds budget until budget is restored

---

## 13. Architect Output Quality Standard (NEW)

The following table defines the minimum acceptable quality for the `arch_design` step across Milestones:

| Milestone | Required | Not Yet Required |
|-----------|----------|-----------------|
| M1/M2 | plan/arch.md, risk/risk_report.json, plan/workplan.md, handoff with modules/interfaces/risks/decisions | ADR format, interfaces.md, codebase scan |
| M3+ | All of the above PLUS plan/interfaces.md, at least 1 ADR per major decision, decisions array in handoff | Formal design review process |

An `arch_design` step that passes M3 validation but lacks real technology decisions (i.e., just restates PM spec in architecture format) is a quality failure even if schema validation passes. Periodic human review of architect output quality is recommended.

---

## 14. Success Criteria

Governance is considered effective when:
- development follows pipeline order
- architecture remains stable
- modules integrate without redesign
- scope expansion is controlled
- the North Star pipeline is fully operational
- module complexity budget is respected
- deprecated routes do not accumulate
- architect output produces genuine technical decisions, not placeholder documents
