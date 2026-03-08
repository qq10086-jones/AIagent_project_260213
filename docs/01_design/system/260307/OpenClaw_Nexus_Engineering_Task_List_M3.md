# OpenClaw Nexus vNext
## Engineering Task List — Milestone 3
## Date: 2026-03-07
## Focus: Structural Hardening + Architect Hardening + Memory Stub

---

## 1. Objective

Milestone 3 addresses the structural and architectural quality gaps identified in the v2 design review.

This is **not** a feature expansion milestone. It is a hardening milestone that makes the existing North Star pipeline maintainable, trustworthy, and ready for team expansion.

Primary goals:
- Decompose the orchestrator monolith into the 4-layer structure
- Harden the Architect Engineer role to produce real architecture blueprints
- Add Brain Router policy layer for deterministic routing
- Consolidate routes and remove deprecated legacy paths
- Introduce the minimal Memory/Context Layer stub

---

## 2. Prerequisite

Milestone 3 must not start until:
- WS-10 (Observability + step-level notification) is closed with live evidence
- Milestone 2 final stage review is documented

---

## 3. Workstream Overview

| ID | Name | Type | Blocks |
|----|------|------|--------|
| WS-11 | Orchestrator Internal Decomposition | A | WS-12, WS-13, WS-14 |
| WS-12 | Architect Engineer Hardening | A | WS-14 |
| WS-13 | Brain Router Policy Layer | A | — |
| WS-14 | Route Consolidation & Legacy Cleanup | A | — |
| WS-15 | Memory / Context Layer Stub | B | — |

---

## 4. Detailed Task List

---

## WS-11 Orchestrator Internal Decomposition

**Type:** Type A / Critical Path
**Pipeline node:** All layers — internal structure enforcement

**Why now:**
`index.js` is 2790 lines and `workflow_engine.js` is 2131 lines. Both violate the design principle that "OpenClaw is the orchestration trunk, not a monolithic brain." This creates a ceiling on testability, deployability, and team-scale development.

---

### Task 11-01 Define layer boundaries and module map

Deliverable:
- `docs/01_design/system/260307/Orchestrator_Layer_Map.md`

Contents:
- Explicit mapping of every current file in `src/` and `src/vnext/` to one of the 4 layers
- Listing of all violations (files that touch multiple layers)
- Target module list after decomposition

Acceptance criteria:
- Every existing source file is assigned to exactly one layer
- All cross-layer import violations are listed

Non-scope:
- No code changes in this task

---

### Task 11-02 Extract Discord adapter from index.js

Deliverable:
- `src/adapters/discord_gateway.js`

Contents:
- Discord.js Client initialization
- Message event handlers (messageCreate, interactionCreate)
- Discord response helpers (replyChunked, safeTranslate)
- Discord attachment handling

Requirements:
- `index.js` must not import Discord.js after this task
- Discord adapter must import from `src/vnext/chat_entrypoint.js` to dispatch requests
- All Discord-specific logic is inside the adapter, not in index.js

Acceptance criteria:
- Discord event handling works correctly after extraction
- `index.js` no longer contains `import { Client } from "discord.js"`
- Integration test covering Discord message → chat entrypoint dispatch path

---

### Task 11-03 Extract data access layer from index.js

Deliverable:
- `src/data/task_repository.js`
- `src/data/event_repository.js`
- `src/data/run_repository.js`

Requirements:
- Move all raw SQL queries from index.js and workflow_engine.js into repository modules
- Repository functions must accept a `pool` parameter (no global pool import in domain layer)
- Domain layer modules must call repositories, never raw `pool.query`

Acceptance criteria:
- `src/index.js` contains zero `pool.query` calls
- Repository modules have unit tests for each query
- `src/workflow_engine.js` imports from `src/data/` not from pool directly

Non-scope:
- No ORM introduction
- No schema migration changes

---

### Task 11-04 Decompose workflow_engine.js

Deliverable:
- `src/domain/workflow_runner.js` — step execution loop
- `src/domain/workflow_state.js` — state transition logic
- `src/domain/workflow_artifact_audit.js` — artifact validation logic

Requirements:
- `workflow_engine.js` becomes an orchestrator that delegates to the three sub-modules
- Target line count for `workflow_engine.js` after decomposition: < 600 lines
- Step execution loop, state machine, and artifact checks are independently testable

Acceptance criteria:
- All existing integration/canary tests still pass after decomposition
- Line count of `workflow_engine.js` is below 600
- Each new module has at least one unit test

---

### Task 11-05 Define index.js as thin HTTP router only

Deliverable:
- Refactored `src/index.js`

Requirements:
- `index.js` contains only:
  - Express app init
  - Middleware setup
  - Route definitions (thin handlers, no business logic inline)
  - Server start
- All business logic moved to service layer (`src/vnext/`) or domain layer (`src/`)
- Target line count: < 800 lines

Acceptance criteria:
- `index.js` line count is below 800
- No raw SQL in `index.js`
- No Discord.js in `index.js`
- No inline LLM calls in `index.js`
- All existing integration tests pass

---

## WS-12 Architect Engineer Hardening

**Type:** Type A / Critical Path
**Pipeline node:** Coding Team Workflow — `arch_design` step

**Why now:**
The current `arch_design` step uses 3 simple instructions and `coding.delegate`. It produces documents but does not produce a real architecture blueprint. Implementation agents receive insufficient technical guidance, leading to low-quality handoffs. The Architect is the single highest-leverage step in the Coding Team workflow.

---

### Task 12-01 Write Architect prompt script v2

Deliverable:
- Updated `architect.system_spec.v2` prompt script in the Prompt Script Registry

Requirements:
- Include explicit instructions for:
  - Scanning and listing existing modules in the codebase context
  - Defining module boundaries and ownership (FE/BE/shared)
  - Producing at least one ADR per major technology decision
  - Producing `plan/interfaces.md` with all API endpoints or internal interfaces this task touches
  - Producing `plan/workplan.md` with per-role scope boundaries
- Prompt must be deterministic: same input → same structure of output
- Must specify artifact paths as absolute paths under the run artifact root

Acceptance criteria:
- Prompt script registered in `prompt_script_registry.json`
- Script ID: `architect.system_spec.v2`
- ADR template is embedded in the prompt

---

### Task 12-02 Add interfaces.md to arch_design required artifacts

Deliverable:
- Updated `coding_team_v0_artifacts.json`
- Updated arch_design step definition in `workflow_engine.js`

Requirements:
- `plan/interfaces.md` added to `required_artifacts` for `arch_design`
- Artifact audit must fail if `plan/interfaces.md` is absent
- Handoff validator must check for interfaces section in arch output

Acceptance criteria:
- Live workflow run fails artifact audit if `plan/interfaces.md` is missing
- Integration test confirms this behavior

---

### Task 12-03 Add ADR output to arch_design handoff validator

Deliverable:
- Updated `coding_team_arch_handoff.schema.json`
- Updated `coding_team_handoff_validators.js`

Requirements:
- Handoff manifest `handoff/architect_to_impl.json` must include a `decisions` array
- Each entry in `decisions` must have: `adr_id`, `title`, `status`
- Validator rejects handoff if `decisions` array is empty

Acceptance criteria:
- Canary test covers: valid handoff with decisions, invalid handoff with empty decisions
- Schema validation error message is human-readable

---

### Task 12-04 Architect canary test with real artifact check

Deliverable:
- Updated or new canary covering the `arch_design` step

Requirements:
- Run a live or stub workflow through `arch_design`
- Verify: `plan/arch.md`, `plan/interfaces.md`, `risk/risk_report.json`, `plan/workplan.md`, `handoff/architect_to_impl.json` all exist
- Verify: `handoff/architect_to_impl.json` contains `decisions` array with at least 1 entry
- Verify: handoff passes schema validation

Acceptance criteria:
- Canary passes with all artifacts present and validated
- Failure case covered: canary fails if `plan/interfaces.md` is absent

---

## WS-13 Brain Router Policy Layer

**Type:** Type A / Critical Path
**Pipeline node:** Brain Router

**Why now:**
The Brain Router currently relies primarily on LLM classification. Non-deterministic routing is a silent failure mode that is invisible in canary tests but observable in production. The policy layer ensures the system degrades gracefully and routes safely when LLM output is ambiguous or incorrect.

---

### Task 13-01 Define routing policy contract

Deliverable:
- `orchestrator/contracts/brain_router_policy.schema.json`
- `docs/01_design/system/260307/Brain_Router_Policy_Contract.md`

Contents:
- List of deterministic override rules
- Each rule: trigger condition, override action, log level
- Confidence threshold definition

---

### Task 13-02 Implement policy override module

Deliverable:
- `src/vnext/brain_router_policy.js`

Rules to implement:
- Input prefix `/coder` → force `orchestrated_workflow`, intent `coding`
- Input length < 3 tokens → force `direct_reply`, intent `chat`
- Input contains explicit financial/trading keywords → force `human_review_required`
- LLM returns unknown intent → downgrade to `direct_reply` with clarification prompt
- LLM returns empty or invalid JSON → emit typed error, do not crash

Acceptance criteria:
- Unit tests cover all 5 rules
- Policy module is called by `brain_router.js` after LLM classification
- All overrides are logged with rule ID and trigger value

---

### Task 13-03 Integration test for policy override paths

Deliverable:
- Integration test covering at least 3 policy override cases

Acceptance criteria:
- `/coder` prefix correctly forces orchestrated_workflow regardless of LLM output
- Short input correctly bypasses orchestration
- Invalid LLM JSON returns typed error response

---

## WS-14 Route Consolidation & Legacy Cleanup

**Type:** Type A / Critical Path
**Pipeline node:** Transport/Adapter Layer

**Why now:**
There are 4 deprecated routes in `index.js` that are not in the North Star path. Their presence creates ambiguity, inflates index.js, and misleads future developers about which paths are authoritative.

---

### Task 14-01 Audit all routes against canonical list

Deliverable:
- `docs/01_design/system/260307/Route_Audit.md`

Contents:
- List of all current routes in `index.js`
- Classification: canonical / deprecated / uncertain
- Evidence of whether deprecated routes are called by any tests or external clients

Acceptance criteria:
- All routes are classified
- Deprecated routes are confirmed unused by existing tests

---

### Task 14-02 Add deprecation headers to deprecated routes

Deliverable:
- Updated `index.js` with `X-Deprecated: true` response header on deprecated routes
- Warning log on each call

---

### Task 14-03 Remove deprecated routes after verification

Deliverable:
- `index.js` with deprecated routes removed:
  - `POST /execute-tool`
  - `POST /debug/plan`
  - `POST /workflows` (old)
  - `GET /ui/approvals`

Requirements:
- Confirm no integration tests call these routes before removal
- Update API documentation or runbook if any was referencing these routes

Acceptance criteria:
- Removed routes return 404
- All existing canonical integration tests still pass
- `index.js` line count reduced by at least 150 lines from this task

---

## WS-15 Memory / Context Layer Stub

**Type:** Type B / Enhancement
**Pipeline node:** Memory Layer

**Why now:**
Every Coding Team workflow run is currently memoryless. The Architect has no access to prior ADRs or project constraints. This reduces output quality for repeated or iterative projects. The stub introduces the minimal file-based context store.

---

### Task 15-01 Define memory store schema

Deliverable:
- `orchestrator/contracts/memory/project_context.schema.json`
- `orchestrator/contracts/memory/adr_record.schema.json`
- `orchestrator/contracts/memory/task_history_entry.schema.json`

---

### Task 15-02 Implement file-based memory reader

Deliverable:
- `src/domain/memory_reader.js`

Methods:
- `getProjectContext(project_id)` → returns project context JSON or null
- `getPriorADRs(project_id)` → returns list of ADR summaries or []
- `getTaskHistory(project_id, limit)` → returns last N task entries or []

Requirements:
- Read-only module (no writes)
- Reads from `artifacts/memory/{project_id}/`
- Returns null/[] gracefully if files do not exist

---

### Task 15-03 Wire memory reader into Architect step

Deliverable:
- Updated workflow_engine.js or arch_design step builder

Requirements:
- Before running `arch_design`, read project context from memory reader
- Append project context summary to the Architect's task prompt if available
- If no memory exists, proceed normally without error

Acceptance criteria:
- With no memory files: workflow runs normally
- With memory files present: architect prompt includes project context

---

### Task 15-04 Implement memory writer (post-workflow)

Deliverable:
- `src/domain/memory_writer.js`

Requirements:
- Called after workflow terminal `succeeded`
- Writes: task_history_entry (outcome, artifacts, run_id, date)
- Writes: any new ADRs from `plan/adr/` into the memory store
- Does not overwrite prior context — appends to history

---

## 5. Suggested Execution Order

```
WS-11-01  Layer Map (no code)
WS-13-01  Brain Router policy contract (no code)
WS-14-01  Route audit (no code)
      ↓
WS-11-02  Extract Discord adapter
WS-11-03  Extract data access layer
WS-13-02  Implement policy override module
WS-14-02  Add deprecation headers
      ↓
WS-11-04  Decompose workflow_engine.js
WS-12-01  Architect prompt script v2
WS-13-03  Policy integration tests
      ↓
WS-11-05  Finalize index.js as thin router
WS-12-02  Add interfaces.md requirement
WS-12-03  ADR handoff validator
WS-14-03  Remove deprecated routes
      ↓
WS-12-04  Architect canary test
WS-15-01  Memory schema
      ↓
WS-15-02  Memory reader
WS-15-03  Wire into Architect step
WS-15-04  Memory writer
```

---

## 6. Definition of Done for Milestone 3

Milestone 3 is complete when:
- `index.js` line count ≤ 800
- `workflow_engine.js` line count ≤ 600
- Discord adapter is in `src/adapters/discord_gateway.js`
- All raw SQL is in `src/data/` repositories
- Brain Router policy override layer exists and has unit tests
- `arch_design` step produces `plan/interfaces.md` and at least 1 ADR
- Deprecated routes are removed and 404
- Memory stub can read and write project context
- All M2 integration/canary tests still pass

---

## 7. Non-Scope for Milestone 3

- No new agent teams (no quant expansion, no ecommerce)
- No full UI dashboard
- No vector/semantic memory
- No distributed orchestrator (stays single process)
- No new workflow types
- No change to task state machine states
