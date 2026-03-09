# Execution Governance & Scope Control v3
## OpenClaw Nexus Project Governance Specification
## Date: 2026-03-08
## Supersedes: docs/01_design/system/260307/OpenClaw_Execution_Governance_Scope_Control_v2.md

---

## Changelog from v2

| Section | Change |
|---------|--------|
| Section 1 | Updated purpose to reflect multi-AI collaboration context |
| Section 3 | Added Principle 7: LLM Role Consistency |
| Section 4 | Updated Type A examples to include M4 workstreams |
| Section 10 | Added Section 10.3: Multi-AI Collaboration Protocol (NEW) |
| Section 11 | Added LLM Role Policy enforcement mechanism |
| Section 12 | Updated complexity budget table with `llm_dispatcher.js` |
| Section 13 | Added LLM Dispatcher output quality standard |
| Section 14 | Updated success criteria |

---

## 1. Purpose

This document defines the execution governance system for the OpenClaw Nexus project.

**OpenClaw Nexus is a multi-AI collaborative project.** Multiple AI agents (PM, Architect, Engineer roles) participate in the design, review, and implementation of the system itself. This governance document is the shared contract that all AI participants must read, understand, and adhere to before performing any task. It is not supplementary documentation — it is the operating protocol.

This governance layer prevents:
- Architectural drift between AI sessions
- Uncontrolled scope expansion by individual AI instances
- Premature implementation of downstream components before upstream dependencies are complete
- Structural entropy accumulation inside existing modules
- LLM model assignment scattered across codebases without audit trail
- Conflicting design decisions introduced by different AI sessions without review

This governance layer ensures:
- Stable architectural evolution across multiple AI sessions
- Controlled development sequencing
- High signal-to-noise engineering progress
- Alignment with the North Star execution pipeline
- Maintainable internal module structure
- Reproducible multi-AI collaboration with traceable decisions

---

## 2. North Star Execution Path

All development activity must ultimately support the primary operational pipeline.

```
Human Input
↓
Discord Gateway (adapter layer)
↓
Brain Router + Policy Layer (heuristic, no LLM)
↓
TaskEnvelope Normalization
↓
LLM Dispatcher (role-policy-driven model routing)
↓
OpenClaw Orchestration (4-layer internal structure)
↓
Coding Team Workflow (PM → Arch → BE → FE → QA → Release)
↓
Artifacts (docs / interfaces / implementation files / reports)
```

Any development work must prove that it shortens, stabilizes, or enables this pipeline.

If the work cannot clearly map to this pipeline, it must be categorized as Exploratory Work (Backlog).

---

## 3. Governance Principles

### Principle 1 — Upstream Completion Rule

No downstream component may begin implementation until its upstream dependency is fully completed and validated with evidence.

### Principle 2 — North Star Alignment

All tasks must demonstrate direct alignment with the North Star pipeline.

Allowed justification examples:
- Enables routing accuracy
- Enables workflow orchestration
- Enables agent role handoff
- Enables artifact delivery
- Enables correct LLM model selection for a role

Rejected justification examples:
- "Useful in the future"
- "Nice to have"
- "Makes the system more flexible"
- "General framework improvement"

### Principle 3 — Controlled Expansion

New subsystems may only be introduced after the current pipeline stage reaches Definition of Done.

Prohibited expansions (until Coding Team full workflow is operational):
- New agent teams (quant, ecommerce, research)
- New orchestration layers
- New UI systems
- LLM classification in Brain Router (deferred to future milestone)

### Principle 4 — Contract-Based Work

All modules must operate on explicit contracts.

Contracts must define: input schema, output schema, expected artifacts, validation rules, failure conditions.

No module may rely on implicit assumptions, undocumented message formats, or free-text interfaces.

### Principle 5 — Minimal Execution Surface

At each stage, implement only the minimal functionality required to support the North Star pipeline.

Feature expansion occurs only after the minimal pipeline functions reliably.

### Principle 6 — Structural Entropy Governance

Internal module structure is subject to the same governance discipline as feature scope.

Rules:
1. No source file may exceed its complexity budget (see Section 12) without an architectural review.
2. New code added to a module already at budget must first go into a correctly-layered sub-module.
3. Cross-layer imports are architectural violations — Layer 1 must not import from Layer 4 directly; Layer 2 must not call raw database.
4. At every milestone boundary, the module complexity budget table must be reviewed and violations tracked.

Structural debt is not automatically deferred — it must be explicitly tracked as a Type A task if it blocks testing, deployment, or new feature work.

### Principle 7 — LLM Role Consistency (NEW)

LLM model assignment is a governed artifact, not an implementation detail.

Rules:
1. Every agent role must have exactly one entry in `orchestrator/configs/llm_role_policy.json`.
2. No model name may be hardcoded in any source file, prompt script, or workflow definition.
3. `prompt_script_registry.json` must not contain a `model` field; it must reference `llm_role` instead.
4. All agent execution LLM calls must route through `src/vnext/llm_dispatcher.js`.
5. Changing a role's model requires only editing `llm_role_policy.json` — no code changes.
6. Brain Router routing logic is explicitly exempt from LLM Dispatcher governance (it uses no LLM).

---

## 4. Task Classification System

### Type A — Critical Path Tasks

Tasks required to enable the North Star pipeline.

Examples (cumulative across milestones):
- Brain Router schema
- TaskEnvelope definition
- Workflow Planner
- Coding Team contracts
- Artifact packaging
- Orchestrator internal decomposition (M3)
- Architect Engineer hardening (M3)
- Brain Router policy layer (M3)
- LLM Provider Registry + Dispatcher (M4)
- Coding Team handoff schemas (M4)
- Coding Team execution steps BE/FE/QA/Release (M4)

Highest priority. Must be completed before Type B tasks.

### Type B — Enhancement Tasks

Tasks that improve quality but do not block execution.

Examples:
- Dashboards
- Logging improvements
- UI polish
- Artifact browsing tools
- Memory/Context Layer (stub in M3; promoted in M4)

Allowed only after Type A tasks complete.

### Type C — Exploratory Tasks

Tasks unrelated to the current pipeline stage.

Examples:
- New agent ecosystems (quant expansion, research team)
- Ecommerce assistant
- LLM classification in Brain Router
- Advanced memory systems (vector search, semantic retrieval)
- Multi-tenant orchestration

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
**LLM Role** — if the task introduces any agent execution, which role from `llm_role_policy.json` it uses

---

## 6. Definition of Done (DoD)

A module is considered complete only when all criteria below are satisfied:

1. Input schema defined
2. Output schema defined
3. Contract documentation exists
4. Integration tests exist
5. Error conditions defined
6. Downstream module compatibility verified
7. Module is in the correct architectural layer
8. Module does not exceed its complexity budget
9. If the module involves LLM calls: role is registered in `llm_role_policy.json` and calls route through LLM Dispatcher (NEW)

Partial functionality does not count as completion.

---

## 7. Definition of Not Done (DoND)

A module is not complete if any of the following remain:
- Outputs rely on free text
- Schema not validated
- Failure cases undefined
- Integration tests missing
- Downstream module blocked
- Module placed in wrong architectural layer
- Module exceeds complexity budget without a decomposition plan
- Model name hardcoded in source code or prompt script (NEW)
- LLM calls bypass LLM Dispatcher (NEW)

---

## 8. Change Control Process

Scope expansion requires formal approval.

A Change Request must include:
1. Justification for change
2. Impact on North Star pipeline
3. Affected modules
4. Risk assessment
5. Layer impact (does this require cross-layer changes?)
6. Complexity budget impact (does this push a module over its budget?)
7. LLM role impact (does this change which model a role uses, or add a new role?) (NEW)
8. Production activation impact (does this change runtime enablement state, cohort scope, or rollback semantics for an already-implemented feature?) (NEW)

Changes are approved only if they improve the current pipeline stage.

Milestone closure does not automatically authorize production activation of a capability that remains runtime-gated. Any post-closure move from implemented-but-disabled to enabled-in-production must go through explicit change control with rollout scope, evidence, and rollback authority recorded.

---

## 9. Anti-Divergence Mechanism

To prevent project drift, the following rule is enforced:

If a task cannot map directly to the North Star pipeline, it is automatically moved to backlog.

This rule applies equally to all participants — human or AI.

---

## 10. Role Boundary Enforcement

### 10.1 Product Manager Role

Allowed:
- Define problem scope, acceptance criteria, milestones
- Propose LLM role assignments based on user cost/quality requirements
- Prioritize workstreams

Not Allowed:
- Define architecture or module boundaries
- Choose frameworks or LLM providers without architectural review
- Modify system layer boundaries

### 10.2 Architect Role

Allowed:
- Define module boundaries, system interfaces, dependency design, ADRs
- Validate and reject PM proposals that have architectural issues
- Specify LLM role assignments in `llm_role_policy.json`
- Identify config structure concerns (single source of truth, separation of concerns)

Not Allowed:
- Expand product scope
- Introduce unrelated subsystems
- Write implementation code

### 10.3 Engineering Roles

Allowed:
- Implementation within defined contracts
- Report technical risks
- Improve internal code structure within layer boundary

Not Allowed:
- Alter architecture
- Introduce new system domains
- Expand project scope
- Hardcode model names in implementation code

### 10.4 Multi-AI Collaboration Protocol (NEW)

OpenClaw Nexus is developed by multiple AI agents across multiple sessions. This section defines the collaboration protocol to ensure continuity and consistency.

#### Session Startup Protocol

Every AI session must begin by reading:
1. `docs/03_feature_development/PROGRESS_LATEST.md` — current state snapshot
2. `docs/01_design/system/260308/OpenClaw_Execution_Governance_Scope_Control_v3.md` — this document
3. `docs/01_design/system/260308/OpenClaw_Nexus_Design_Document_v3.md` — active design document
4. The current milestone task list (e.g., `OpenClaw_Nexus_Engineering_Task_List_M4.md`)

No AI may begin work without completing the startup protocol.

#### Role Handoff Protocol

When one AI session ends before a task is complete:
1. Write a progress report to `docs/03_feature_development/progress_reports/progress_{datetime}_{description}.md`
2. Update `PROGRESS_LATEST.md` to reflect current state
3. Note clearly: what was completed, what is in progress, what is blocked

The next AI session must not assume the prior session's in-progress work is correct without verification.

#### Decision Review Cycle

Multi-AI design decisions follow a two-step review cycle:

```
PM AI proposes → Architect AI reviews → PM AI revises → approved
```

A design document or task list produced by PM AI is not final until Architect AI has reviewed it and issued formal acceptance or a list of required changes. The PM AI must then incorporate the changes before the document is considered authoritative.

The conversation history of a PM → Architect review cycle must be summarized in a progress report before the session closes, so that subsequent AI sessions can understand the decisions made.

#### Conflict Resolution

If two AI sessions produce conflicting designs or implementations:
1. The more recent governance-compliant version takes precedence
2. If both are governance-compliant, escalate to human review
3. Never silently accept a conflict — it must be documented

#### Prohibited Behaviors for All AI Participants

- Silently expanding scope beyond what the current milestone defines
- Introducing a new dependency without an ADR
- Hardcoding a model name instead of using `llm_role_policy.json`
- Marking a task as "DONE" without evidence of acceptance criteria being met
- Skipping the session startup protocol
- Modifying this governance document without a Change Request

---

## 11. Governance Enforcement Mechanisms

**Workflow Locks** — Downstream modules remain locked until upstream modules satisfy DoD.

**Contract Validation** — All outputs validated against schema before acceptance.

**Task Review** — All Type A tasks require architectural review before execution.

**Change Requests** — All scope expansion requires formal approval.

**Complexity Budget Enforcement** — See Section 12. Violations are tracked and must be resolved as Type A tasks within the same milestone they are detected.

**LLM Role Policy Enforcement (NEW)**
- `validate_registry.js` must reject any prompt script entry with a `model` field
- `validate_registry.js` must reject any prompt script entry without an `llm_role` field
- CI or pre-canary validation must verify `llm_role_policy.json` and `llm_providers.json` pass schema validation
- Any grep for hardcoded model names (e.g., `"qwen-max"`, `"deepseek-r1"`) in source files outside `llm_dispatcher.js` and `llm_providers.json` is a governance violation

**vnext Completion Definition**
The vnext refactoring phase is complete when:
- `index.js` contains no Discord.js imports
- `index.js` contains no raw SQL
- `index.js` contains no inline LLM calls
- `index.js` line count is ≤ 800
- All business logic lives in `src/vnext/` (service layer) or `src/` domain modules
- Status: COMPLETE (confirmed M3)

**Route Deprecation Protocol**
When a route is identified as deprecated:
1. Document it in a Route Audit record
2. Add `X-Deprecated: true` header and warning log
3. Confirm zero usage in integration tests
4. Remove the route
5. Update the Route Audit record to `removed`
No deprecated route may remain for more than one milestone.

**Architect Output Quality Gate**
The `arch_design` step output is not accepted unless:
- `plan/interfaces.md` exists and contains at least 1 interface definition
- `handoff/architect_to_impl.json` contains a non-empty `decisions` array with ADR IDs
- `risk/risk_report.json` contains at least 1 risk entry

---

## 12. Module Complexity Budget

| Module | Layer | Max Lines | Current | Status |
|--------|-------|-----------|---------|--------|
| `src/index.js` | 1 (Transport) | 800 | 546 | OK |
| `src/workflow_engine.js` | 3 (Domain) | 600 | 431 | OK |
| `src/vnext/*.js` (per file) | 2 (Service) | 300 | ≤275 | OK |
| `src/vnext/llm_dispatcher.js` | 2 (Service) | 220 | 0 (M4) | — |
| `src/domain/*.js` (per file) | 3 (Domain) | 500 | ≤264 | OK |
| `src/adapters/*.js` (per file) | 1 (Transport) | 400 | ≤286 | OK |
| `src/data/*.js` (per file) | 4 (Infra) | 250 | compliant | OK |

Review process:
- Reviewed at every milestone boundary
- Violations are automatically Type A tasks for the next milestone
- No new feature work may begin in a module that exceeds budget until budget is restored

---

## 13. Quality Standards

### 13.1 Architect Output Quality Standard

| Milestone | Required | Not Yet Required |
|-----------|----------|-----------------|
| M1/M2 | plan/arch.md, risk/risk_report.json, plan/workplan.md, handoff with modules/interfaces/risks/decisions | ADR format, interfaces.md |
| M3+ | All of the above + plan/interfaces.md, ≥1 ADR per major decision, non-empty decisions array in handoff | Formal design review |

An `arch_design` step that passes schema validation but lacks real technology decisions is a quality failure.

### 13.2 LLM Dispatcher Quality Standard (NEW)

The LLM Dispatcher is considered production-quality when:
- All agent execution LLM calls produce structured log entries with role, provider, model, and latency
- `validateProviders()` is called at startup and results are logged
- Unknown role or provider throws a typed error (not unhandled exception)
- Unit test coverage includes all paths: cloud, local, override, error cases

---

## 14. Success Criteria

Governance is considered effective when:
- Development follows pipeline order
- Architecture remains stable across AI sessions
- Modules integrate without redesign
- Scope expansion is controlled
- The North Star pipeline is fully operational
- Module complexity budget is respected
- Deprecated routes do not accumulate
- Architect output produces genuine technical decisions
- LLM model assignments are readable from a single policy file
- No model name is hardcoded outside the infrastructure config
- AI sessions consistently read governance documents before starting work
- PM → Architect review cycle is documented in progress reports

---

## 15. Document Version History

| Version | Date | Author | Key Changes |
|---------|------|--------|-------------|
| v1 | 2026-03-06 | AI (PM+Arch) | Initial governance spec |
| v2 | 2026-03-07 | AI (PM+Arch) | Added structural entropy governance, complexity budget, architect quality standard |
| v3 | 2026-03-08 | AI (PM+Arch) | Added Multi-AI Collaboration Protocol, LLM Role Consistency principle, LLM governance enforcement |
