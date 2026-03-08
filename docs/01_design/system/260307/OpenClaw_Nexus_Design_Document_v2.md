# OpenClaw Nexus vNext
## Design Document v2
## Date: 2026-03-07
## Supersedes: docs/01_design/system/260306/OpenClaw_Nexus_vNext_Design_Document.md

---

## Changelog from v1

| Section | Change |
|---------|--------|
| Section 3 | Added Orchestrator Internal Layering (4-layer decomposition) |
| Section 5 | Brain Router: added determinism requirement and policy override layer |
| Section 7 | Architect Agent: promoted to full Engineer contract with ADR, codebase analysis, interface spec |
| Section 8 | Coding Team: workflow steps corrected, Architect output requirements hardened |
| Section 13 | Memory/Context Layer: added minimal concrete design |
| Section 16 | Observability: step-level notification requirements added |
| Section 17 | Route Consolidation: legacy route deprecation policy added |
| Section 18 | Orchestrator Complexity Budget: new section |

---

## 1. Vision

OpenClaw Nexus is a local-first multi-agent execution system designed to convert human natural-language input into structured execution workflows.

The system uses Discord as the primary interaction gateway. Incoming requests are first analyzed by a Brain Router layer, which determines user intent, task complexity, and whether orchestration is required. Simple conversational tasks are answered directly. Execution-oriented tasks are transformed into structured task envelopes and passed to OpenClaw, which acts as the orchestration trunk and dispatches specialized agents, workers, and tools.

The long-term goal is a central AI operating system that manages multiple specialized teams, beginning with the Coding Team.

---

## 2. Core Principles

### 2.1 Local-first
All core inference, orchestration, artifact generation, and task execution must run in the local environment or local Docker stack.

### 2.2 Brain before orchestration
Not every request enters OpenClaw. The Brain layer classifies requests first and only escalates requests that require workflow execution.

### 2.3 OpenClaw as trunk, not brain
OpenClaw is the orchestration trunk. It must not become a monolithic reasoning engine. It receives structured inputs and produces structured outputs.

### 2.4 Agent = contract, not persona
Each agent is defined by input schema, output schema, allowed tools, success criteria, failure handling, and escalation rules. No free-text interfaces.

### 2.5 Document-first for planning roles
PM, Architect, UI, Analyst, QA-planning: primary output is structured documentation, not immediate code.

### 2.6 Execution-first for coding roles
Frontend, Backend, Integration, Testing: primary output is executable changes, diffs, code files, tests, and runbooks.

### 2.7 Internal structure is also governed
Module size, layering, and internal coupling are subject to the same governance discipline as feature scope. Structural entropy is a first-class risk.

---

## 3. High-Level Architecture

### 3.1 Input Layer
- Human natural-language input
- Primary gateway: Discord
- Future gateways: Web UI, CLI, API, Scheduled triggers

### 3.2 Brain Router Layer
Responsibilities:
- parse user input
- classify intent via LLM
- apply deterministic policy override on ambiguous/edge cases
- detect complexity and orchestration requirement
- normalize user request into structured task envelope

Possible intent classes: chat, coding, quant, docs, research, ops, unknown

Possible decision outcomes: direct_reply, single_agent, orchestrated_workflow, human_review_required

### 3.3 OpenClaw Orchestration Layer
Responsibilities:
- receive structured task envelope
- select workflow template
- assign role-based agents
- manage state transitions
- invoke tools/workers
- collect artifacts
- return execution result

### 3.4 Execution Layer
Current executors:
- worker-coder
- worker-quant
- prompt-script agents
- Codex adapter
- OpenCode adapter

### 3.5 Artifact Layer
Stores: design docs, task lists, code patches, screenshots, reports, QA summaries, workflow logs.

### 3.6 Memory / Context Layer (Minimal Design — see Section 13)
Stores: task history, project context, reusable constraints, prior design docs, execution traces for replay.

### 3.7 Orchestrator Internal Layers (NEW)

The orchestrator is a single deployable service, but internally it must follow a strict 4-layer structure. All new code must be placed in the correct layer. Index.js violations must be resolved during Milestone 3.

```
┌─────────────────────────────────────────────┐
│  Layer 1: Transport/Adapter Layer           │
│  - Discord adapter (event in, reply out)    │
│  - HTTP route definitions (thin)            │
│  - Input normalization only                 │
├─────────────────────────────────────────────┤
│  Layer 2: Service Layer (vnext/)            │
│  - Brain Router                             │
│  - Runtime Dispatch                         │
│  - Approval Entrypoint                      │
│  - Chat Entrypoint                          │
│  - Workflow Notification Delivery           │
├─────────────────────────────────────────────┤
│  Layer 3: Domain Layer (src/)               │
│  - Workflow Engine                          │
│  - Tool Adapter Registry                   │
│  - Coding Team Validators                  │
│  - Artifact Registry                       │
│  - Policy / Risk Classifier                │
├─────────────────────────────────────────────┤
│  Layer 4: Infrastructure Layer             │
│  - PostgreSQL (pool)                        │
│  - Redis (stream)                           │
│  - S3 / MinIO                              │
│  - File system artifact store              │
└─────────────────────────────────────────────┘
```

Rules:
- Layer 1 must not contain business logic
- Layer 2 must not contain raw SQL or Redis calls
- Layer 3 must not import Discord.js or parse HTTP bodies
- Layer 4 must be accessed only through repository interfaces

---

## 4. Canonical Task Object

All non-trivial tasks must be normalized into a canonical task envelope before execution.

```json
{
  "task_id": "uuid",
  "source": "discord",
  "user_input": "Build a CRM MVP with login, customer list and notes",
  "intent": "coding",
  "sub_intent": "project_bootstrap",
  "requires_orchestration": true,
  "target_team": "coding_team",
  "expected_outputs": ["design_doc", "task_breakdown", "repo_changes"],
  "constraints": {
    "local_only": true,
    "approval_mode": "manual",
    "risk_level": "medium"
  },
  "context": {
    "channel_id": "discord-channel-id",
    "thread_id": "discord-thread-id",
    "attachments": []
  }
}
```

---

## 5. Brain Router Design

### 5.1 Responsibilities
The Brain Router answers:
1. What is the user trying to do?
2. Is this a direct-answer or execution task?
3. Does the request need orchestration?
4. Which team handles it?
5. What artifacts are expected?

### 5.2 Router Output
Structured JSON only. Never free text.

### 5.3 Routing Policy
- `chat` → direct Brain response, zero workflow records created
- `coding` + simple patch → single coding path
- `coding` + multi-role project → OpenClaw orchestration
- `quant` → quant pipeline
- `docs/research` → document-oriented agent workflow
- `unknown` → clarification or fallback

### 5.4 Determinism Requirement (NEW)

The Brain Router must operate in two phases:

**Phase A — LLM Classification**
Use LLM to produce a candidate intent class and confidence score.

**Phase B — Policy Override Layer**
Apply deterministic rules on top of LLM output:
- `/coder` prefix → force `coding` + `orchestrated_workflow`
- input contains only punctuation or < 3 tokens → force `chat`
- explicit financial keywords → force `human_review_required`
- LLM confidence < threshold → downgrade to `chat` or `unknown`

The policy layer must have its own contract file and integration tests. Routing failures must not default to silent fallback — they must emit a typed error.

### 5.5 Escalation Rules
Escalate to OpenClaw when:
- multiple roles are needed
- multiple artifacts are expected
- approval checkpoints are required
- the task spans design + implementation + verification
- the task includes external tool invocations

---

## 6. OpenClaw Role in the System

OpenClaw is the orchestration trunk.

Responsibilities:
- workflow planning
- role dispatch
- tool invocation
- execution tracking
- artifact aggregation
- retry/recovery hooks
- audit trail

Not responsible for:
- first-pass intent recognition
- direct chat handling
- unrestricted tool execution without policy
- role definition itself

---

## 7. Agent Model

### 7.1 Agent Categories

#### A. Planning Agents
- PM Agent
- Architect Engineer Agent (see Section 7.3 for full spec)
- Analyst Agent
- Research Agent

#### B. Design Agents
- UI/UX Agent
- API Design Agent
- Data Model Agent

#### C. Execution Agents
- Frontend Agent
- Backend Agent
- Integration Agent
- Quant Execution Agent

#### D. Verification Agents
- QA Agent
- Test Agent
- Reviewer Agent
- Risk/Guard Agent

### 7.2 Agent Contract Template
Each agent defines:
- name
- mission
- input schema
- output schema
- tools allowed
- forbidden actions
- dependencies
- success criteria
- retry policy
- escalation policy

### 7.3 Architect Engineer Agent — Full Specification (NEW)

The Architect Engineer is a planning agent with **elevated output requirements** distinct from all other agents. It is not a document generator — it is a design decision maker that produces binding technical constraints for all downstream implementation agents.

**Mission:**
Convert the PM specification into a concrete technical blueprint. Make explicit technology decisions, define module boundaries and interfaces, assess integration risks, and produce a workplan that implementation agents can execute without ambiguity.

**Input:**
- PM output artifacts: `plan/spec.md`, `plan/acceptance.json`, `plan/milestones.md`, `handoff/pm_to_architect.json`
- Project codebase context (existing module list, active dependencies, file tree)
- Relevant prior architecture decisions (if Memory Layer is available)

**Required Output Artifacts:**

| Artifact | Format | Required Sections |
|----------|--------|-------------------|
| `plan/arch.md` | Markdown | System overview, module breakdown, layer boundaries, technology decisions, dependency graph, integration points |
| `plan/adr/adr_NNN.md` | Markdown (ADR format) | One ADR per major technology or architecture decision |
| `plan/interfaces.md` | Markdown | Every API endpoint or internal interface contract this task introduces or modifies |
| `risk/risk_report.json` | JSON | Risk ID, category, probability, impact, mitigation per risk item |
| `plan/workplan.md` | Markdown | Per-role task breakdown for FE, BE, QA with explicit scope boundaries |
| `handoff/architect_to_impl.json` | JSON | Typed handoff manifest |

**ADR Format (per decision):**
```markdown
# ADR-NNN: [Decision Title]
## Status: Accepted
## Context: [Why this decision is needed]
## Decision: [What was decided]
## Consequences: [What becomes easier, what becomes harder]
## Alternatives Considered: [Other options and why rejected]
```

**Typed Handoff Required Fields:**
- `from_step`: `arch_design`
- `to_steps`: list of downstream steps
- `modules`: list of modules with owner (fe/be/shared)
- `interfaces`: list of API contracts or internal interfaces
- `decisions`: list of ADR IDs and short descriptions
- `risks`: list of risk IDs from risk_report.json
- `scope_constraints`: explicit list of what is OUT of scope for implementation

**Validation Rules:**
- `plan/arch.md` must contain headings: `Module Breakdown`, `Technology Decisions`, `Integration Points`
- `risk/risk_report.json` must have at least 1 risk entry
- `plan/interfaces.md` must define at least 1 interface
- Each ADR must have all 5 sections
- Handoff manifest must validate against `coding_team_arch_handoff.schema.json`

**Forbidden Actions:**
- Writing implementation code
- Creating files outside `artifacts/release/<run_id>/`
- Making technology decisions that contradict existing project constraints without an ADR

**Escalation:**
If the codebase context is insufficient to make a technology decision, the Architect must emit a `clarification_required` flag in the handoff rather than guessing.

---

## 8. Coding Team Design

### 8.1 Mission
Convert a user's product request into structured design documentation, implementation plan, code changes, tests, and verification artifacts.

### 8.2 Standard Workflow (Corrected)

```
Step 0: PM Spec          (role: pm)
Step 1: Arch Design      (role: architect_engineer)   ← elevated spec, see Section 7.3
Step 2: Backend Impl     (role: backend)
Step 3: Frontend Impl    (role: frontend)
Step 4: QA Verify        (role: qa)
Step 5: Release Pack     (role: release)
```

UI/UX step remains optional and inserts between steps 1 and 2 when project type requires it.

### 8.3 Architect Step Hardening Gap (Current State)

As of 2026-03-07, the `arch_design` step:
- Uses generic `coding.delegate` tool with 3 simple instructions
- Has no codebase analysis phase
- Does not require ADR format
- Does not produce `plan/interfaces.md`
- Does not validate that architecture decisions are grounded in actual existing code

The `Architect_Engineer_Role_Contract.md` in this directory defines the target state. The hardening work is tracked in `OpenClaw_Nexus_Engineering_Task_List_M3.md` as WS-12.

---

## 9. Non-Coding Agent Workflows

### 9.1 PM/Planning Workflow
Primary output: design doc, task list, acceptance matrix.

### 9.2 UI Workflow
Primary output: UI spec, component map, state/event matrix.

### 9.3 Research Workflow
Primary output: research brief, options matrix, recommendation memo.

---

## 10. Tooling Strategy

### 10.1 Tool Classes
- reasoning models
- code generation tools
- browser tools
- shell/sandbox tools
- quant analysis tools
- artifact generation tools

### 10.2 Current Tool Mapping
- direct conversation → Brain LLM
- structured planning docs → prompt-script agents + LLM
- coding execution → Codex / OpenCode / worker-coder
- quant execution → worker-quant
- browser evidence → OpenClaw browser tools

### 10.3 Tool Abstraction Requirement
Tools must be wrapped behind stable interfaces so that providers can be swapped without changing upstream orchestration logic.

---

## 11. Prompt Script Registry

Each script definition includes:
- script_id
- target_agent
- input schema
- output schema
- preferred model
- temperature / reasoning mode
- allowed tools
- artifact type
- validation rules

Current scripts: `pm.design_doc.v1`, `architect.system_spec.v1`, `ui.component_spec.v1`, `qa.test_plan.v1`

---

## 12. Workflow Patterns

### 12.1 Direct Reply Pattern
- chat, quick Q&A, no artifact requirement

### 12.2 Single-Agent Pattern
- simple doc generation, simple patch, narrow analysis

### 12.3 Multi-Agent Workflow Pattern
- project implementation, multi-role decomposition, design → build → verify flows

### 12.4 Human Approval Pattern
- destructive actions, risky patches, production deployments, financial actions, external communication

---

## 13. Memory / Context Layer — Minimal Design (NEW)

### 13.1 Purpose
Enable agents to access project history, prior design decisions, and recurring constraints without relying on per-request context window capacity.

### 13.2 Minimum Required Store

| Store | Key | Value | Use |
|-------|-----|-------|-----|
| Project Context | `project:{project_id}` | JSON: active tech stack, repo root, constraints | Architect input |
| Prior ADRs | `adr:{project_id}:{adr_id}` | Markdown text | Architect reference |
| Task History | `task:{run_id}` | JSON: task_id, intent, outcome, artifacts | Replay and debugging |

### 13.3 Access Pattern
- Read-only at agent runtime (agents may not write to memory directly)
- Written by orchestrator after workflow completion
- No LLM-driven memory summarization in this phase

### 13.4 Storage Backend
- Minimum: flat JSON files per project under `artifacts/memory/{project_id}/`
- Upgrade path: Redis key-value store with TTL

### 13.5 Non-Scope
- No vector search
- No semantic retrieval
- No LLM-driven memory compression
- No cross-project context sharing

---

## 14. Quality Gates

No workflow is complete without explicit quality gates:
- schema-valid output
- artifact generated
- execution log saved
- validation passed
- failure reason captured if unsuccessful

For coding workflows:
- patch generated
- tests generated or executed
- run instructions produced
- rollback note produced where applicable

---

## 15. Guardrails

### 15.1 Risk Policy
Actions labeled: low / medium / high risk.

### 15.2 High-Risk Actions
Require manual approval: deleting files, modifying secrets, production deployment, broker/trading execution, external publishing, system-wide config changes.

### 15.3 Local Safety
The system must respect local-only execution constraints and avoid unintended cloud dependency.

---

## 16. Observability

### 16.1 Required Notification Points (Updated)

The system must emit a typed notification at each of the following points:

| Event | Trigger | Output |
|-------|---------|--------|
| `workflow.started` | Workflow run created | "Workflow started: {workflow_id}, step 1 of N: {step_title}" |
| `step.completed` | Step transitions to next | "{step_title} completed. Starting {next_step_title}..." |
| `step.approval_required` | Approval gate triggered | "Step {step_title} requires approval. Use /approve or /reject." |
| `workflow.succeeded` | Terminal succeeded | "Workflow complete. Artifacts ready at {artifact_root}." |
| `workflow.failed` | Terminal failed | "Workflow failed at {step_title}: {error_code}. Details: {error_summary}" |

All notifications must:
- use deterministic template strings (no LLM generation)
- pass through the `workflow_runtime_notifier` → `workflow_notification_delivery` path
- not expose internal credentials, tokens, or file paths outside the workspace

### 16.2 Observability API (Existing)
- `/runs/:run_id/status`
- `/runs/:run_id/timeline`
- `/runs/:run_id/artifacts`
- `/workflow-runs/:workflow_run_id`

---

## 17. Route Consolidation Policy (NEW)

### 17.1 Canonical Entry Points

The following routes are the authoritative North Star entry points:

| Route | Purpose |
|-------|---------|
| `POST /chat` | vNext chat + coding dispatch |
| `POST /tasks/:id/approve` | Approval entrypoint |
| `POST /tasks/:id/reject` | Rejection entrypoint |
| `POST /workflow-runs/start` | Direct workflow start |
| `GET /workflow-runs/:id` | Workflow status query |
| `GET /runs/:run_id/timeline` | Step timeline |
| `GET /runs/:run_id/artifacts` | Artifact listing |
| `GET /health` | Health check |

### 17.2 Deprecated Routes (to be removed in M3)

| Route | Reason |
|-------|--------|
| `POST /execute-tool` | Pre-vNext ingress, replaced by `/chat` dispatch |
| `POST /debug/plan` | Debug only, not in North Star path |
| `POST /workflows` | Old workflow creation, replaced by `/workflow-runs/start` |
| `GET /ui/approvals` | UI page in non-UI phase |

### 17.3 Deprecation Process
1. Add deprecation header to response: `X-Deprecated: true`
2. Log warning on each call
3. Remove after M3 integration tests pass without calling the deprecated routes

---

## 18. Orchestrator Complexity Budget (NEW)

To prevent structural entropy from accumulating, the following hard limits apply:

| Module | Max Lines | Current | Status |
|--------|-----------|---------|--------|
| `src/index.js` | 800 | 2790 | VIOLATION — M3 priority |
| `src/workflow_engine.js` | 600 | 2131 | VIOLATION — M3 priority |
| `src/vnext/*.js` (per file) | 300 | max 235 | OK |
| `src/*.js` (other, per file) | 500 | varies | Monitor |

Enforcement:
- New code that would cause a file to exceed its budget requires architectural review
- Decomposition is tracked as a Type A task in M3

---

## 19. State Machine

Suggested task lifecycle:
- received → classified → normalized → planned → dispatched → running → waiting_for_dependency → awaiting_approval → verifying → completed → failed → canceled

Each transition must be logged and queryable.

---

## 20. Non-Functional Requirements

### NFR-01 Determinism
Routing and workflow selection must be deterministic and policy-backed. LLM output must be validated by a policy layer before routing decisions are made.

### NFR-02 Traceability
Every artifact and state transition must be attributable to a task and role.

### NFR-03 Replaceability
Tool providers must be swappable without changing top-level workflow contracts.

### NFR-04 Local-first
No workflow should require cloud-only dependencies unless explicitly approved.

### NFR-05 Recoverability
Workflow failure must not destroy task history or artifacts.

### NFR-06 Structural Maintainability (NEW)
No single module may exceed its complexity budget. Layer boundaries must be respected. Internal coupling between layers must be traceable via imports only (no shared globals outside Layer 4).

---

## 21. Definition of Success

The system is considered successful when:
1. A Discord request is correctly classified by Brain (with policy override verified).
2. Chat requests are answered directly without orchestration.
3. Coding requests trigger a structured Coding Team workflow.
4. The Architect step produces a real architecture blueprint (ADRs, interfaces, workplan) that implementation agents can execute without ambiguity.
5. PM/Architect/UI tasks produce reproducible high-quality documents.
6. Backend/Frontend tasks call coding executors through stable interfaces.
7. Quant tasks route into quant worker without contaminating coding workflows.
8. All artifacts, states, and logs are traceable end-to-end.
9. Step-level progress notifications are emitted to Discord at each workflow transition.
10. The orchestrator codebase respects the 4-layer structure and complexity budget.

---

## 22. Final Product Positioning

OpenClaw Nexus is a local-first AI operating system composed of:
- one input plane
- one routing brain (with deterministic policy layer)
- one orchestration trunk (with enforced internal layering)
- multiple specialized execution teams
- one memory/context store (minimal, grows over time)

The first production-grade team to be completed is the Coding Team, anchored by an Architect Engineer agent that produces real architecture decisions, not just placeholder documents.
