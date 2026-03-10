# OpenClaw Nexus – Project State & M9 Engineering Task List

- Version: 1.0
- Date: 2026-03-10
- Milestone: M9 – Coding Precision & Sandbox Guardrails
- Status: APPROVED

---

## 1. Project Context & Current State (For New Onboardees)

Welcome to OpenClaw Nexus. Our goal is to build an intelligent, multi-agent orchestration framework around the openclaw/opencode tools.

**What we have built so far (M2 - M8):**
- **Core Orchestrator (M2/M3):** The central nervous system (`orchestrator/` folder). It receives events, triggers workflows, and manages state using Redis and Postgres.
- **Role-based LLM Dispatcher (M4):** Different agents (PM, Quant, Coder) can be assigned different sub-tasks dynamically based on user prompts.
- **Workflow DAG Engine (M5):** A powerful parallel execution engine. We can run independent tasks (like Frontend implementation and Backend implementation) at the same time.
- **Dynamic Routing & Governance (M6/M7):** An intelligent "Brain" router (`brain/` folder in Python) that classifies tasks and routes them using dynamic policies. We also built strict audit logging (Decision Logs) to ensure every LLM choice is recorded.
- **Docker Infrastructure:** The entire stack (`db`, `redis`, `orchestrator`, `worker-coder`, `worker-quant`, `brain`) runs smoothly on `docker-compose`. 

**Where we are right now:**
- The workflow **runs flawlessly at the infrastructure level**. Task dispatches are fast (<5s per LLM call), and parallel queues work properly without deadlocks.
- **The Gap:** The underlying coding agent (`worker-coder`) is currently executing "blindly". It takes raw prompts and feeds them to execution providers without task-scoped context restriction, memory-aware repair, or safety guardrails.

---

## 2. M9 Scope: Coding Precision & Sandbox Guardrails

**Objective:** Transform the raw code generation into a precise, sandboxed, and test-driven engineering loop. The execution layer must be as reliable as the orchestration layer.

### Phase 1: Context & Memory Foundation
**Domain:** `worker-coder`
*   **WS-40-01: Dynamic Context Resolver**
    *   *Task:* Before sending prompts to `opencode` or `codex`, the worker must build a task-scoped context packet instead of forwarding the raw workflow prompt directly.
    *   *Required packet fields:* `target_paths`, `candidate_files`, `entrypoints`, `related_tests`, `toolchain_facts`, `recent_changed_files`, `memory_hints`.
    *   *Goal:* Prevent hallucinated file paths, reduce irrelevant context, and make execution reproducible.
*   **WS-40-02: Repo Map Generator**
    *   *Task:* Build a lightweight repository map generator that produces a machine-readable repo digest artifact for coding steps.
    *   *Minimum output:* directory summary, key config files, app entrypoints, test locations, and import/export hints for touched modules.
    *   *Artifact:* `artifacts/context/repo_map.json`
    *   *Goal:* Give every coding step a shared structural view of the repo without paying full-index cost every run.
*   **WS-40-03: Coding Failure Memory**
    *   *Task:* Persist structured execution-failure memory for coding runs, including failed commands, stderr summary, target files, attempted fixes, and terminal outcome.
    *   *Minimum output:* append-only `coding_failures` evidence under project/run memory.
    *   *Goal:* Prevent repeated failed attempts, improve auto-fix quality, and satisfy v4 queryability requirements.
*   **WS-40-04: System Prompt Hardening**
    *   *Task:* Refactor `worker-coder/adapters/` to inject strict formatting constraints and task contract blocks rather than accepting raw conversational text.
    *   *Goal:* Ensure execution providers return deterministic patch-oriented outputs and validation evidence.

### Phase 2: Sandbox & Execution Guardrails
**Domain:** `worker-coder`
*   **WS-41-01: Read/Write Scope Restriction (Chroot)**
    *   *Task:* Implement path validation inside the `CodingService.applyPatch` and execution layers using workflow-approved `target_paths`.
    *   *Goal:* If a model tries to modify files outside the approved workspace scope (e.g., `.git/`, `docker-compose.yml`, unrelated roots), throw `E_UNAUTHORIZED_WRITE` and block the patch.
*   **WS-41-02: Pre-commit Static Linting**
    *   *Task:* Before returning `succeeded` to the orchestrator, the worker must run a fast static check based on repo/toolchain facts (e.g., `eslint`, `oxlint`, `ruff`, `tsc --noEmit`).
    *   *Goal:* Catch syntax and contract errors immediately and trigger a self-healing retry loop up to 3 times before failing the step.

### Phase 3: Test-Driven Verification (Auto-Fix Loop)
**Domain:** `orchestrator` & `worker-coder`
*   **WS-42-01: Automated Test Runner Integration**
    *   *Task:* If a task specifies a `verification_command`, the worker executes it post-patch and records the result as part of the execution evidence bundle.
    *   *Goal:* If it fails, capture the last 50 lines of `stderr`, attach the failing command, record the failure into coding memory, and feed that back to the execution provider as a follow-up task without returning control to the DAG engine immediately.
*   **WS-42-02: Auto-Fix Loop Budgeting**
    *   *Task:* Define and enforce retry budgets, stop conditions, and failure summarization for repair loops.
    *   *Required controls:* `max_attempts`, `same_error_repeat_limit`, `wall_clock_timeout`, `final_failure_summary`.
    *   *Goal:* Make auto-fix bounded, queryable, and safe under repeated failure.

### Phase 4: Queryability & Evidence
**Domain:** `worker-coder` & `orchestrator`
*   **WS-44-01: Coding Evidence Contract Upgrade**
    *   *Task:* Extend coding execution artifacts so every run can explain: what context was used, what files were in scope, what failed, what was retried, and why the final result was accepted or rejected.
    *   *Minimum artifacts:* context packet, repo map, diff bundle, lint/test logs, failure memory entries, final run summary.
    *   *Goal:* Align coding execution with Nexus v4 evidence-backed and queryable runtime principles.

---

## 3. Secondary Architecture Tasks (P1)
**Domain:** `brain/` & `orchestrator/`
*   **WS-43-01: Brain API Boundary Decoupling**
    *   *Context:* The Python `brain/` module currently connects directly to PostgreSQL. This is an architectural anti-pattern and a schema risk.
    *   *Task:* Draft and implement an API gateway within the `orchestrator` so `brain` fetches facts and posts routing decisions via HTTP/gRPC instead of direct DB inserts.

---

## 4. M9 Delivery Order

### P0
- `WS-40-01` Dynamic Context Resolver
- `WS-40-02` Repo Map Generator
- `WS-41-01` Read/Write Scope Restriction
- `WS-42-01` Automated Test Runner Integration

### P1
- `WS-40-03` Coding Failure Memory
- `WS-40-04` System Prompt Hardening
- `WS-41-02` Pre-commit Static Linting
- `WS-42-02` Auto-Fix Loop Budgeting

### P2
- `WS-44-01` Coding Evidence Contract Upgrade
- `WS-43-01` Brain API Boundary Decoupling

---

## 5. Architecture Notes

- `brain` remains responsible for reasoning, routing, and model selection.
- `worker-coder` remains responsible for execution against the workspace.
- Qwen-class models may still exist behind Nexus model routing, but they are not treated as standalone coding execution providers inside `worker-coder`.
- M9 must strengthen `worker-coder` as an execution system, not reintroduce a second reasoning plane inside the coder worker.

---

## 6. Current Implementation Snapshot (2026-03-10)

Completed:
- task-scoped context packet generation is implemented for coding steps
- lightweight repo map generation is implemented and written as artifacts
- context packet and repo map are injected into coding execution payloads
- `target_paths` write guardrails are implemented in `worker-coder`
- scoped snapshot-based `files_changed` recovery is implemented as a replacement for expensive full-repo git scanning
- fast static checks are implemented before success return for changed `.js/.json/.py` files
- `verification_command` execution is implemented and recorded as structured evidence
- coding failure memory is persisted under run-scoped artifacts
- coding failure memory is copied into orchestrator durable memory roots on successful workflow closure
- system prompt hardening is implemented as an execution contract block with scope/output/verification constraints
- bounded auto-fix controls are implemented in `worker-coder` with `max_attempts` and `same_error_repeat_limit`
- auto-fix budgeting now includes `wall_clock_timeout_s` and terminal `final_failure_summary`
- orchestrator step payloads now productize `verification_command` and retry controls for implementation steps
- release-pack evidence now indexes verification logs, retry summaries, and coding failure memory paths
- release-pack evidence now indexes prompt-contract artifacts and final-failure summaries
- local M9 coding guardrails canary exists and passes
- worker-level auto-fix canary now passes in sandbox using an inline mocked provider path
- coding runtime defaults are aligned to `provider=opencode`, `model=qwen3-coder-plus-2025-07-22`
- `brain` fact polling no longer reads PostgreSQL directly; it uses orchestrator HTTP gateway endpoints
- live container stack validation passed for orchestrator startup, brain `/run`, and fact-gateway lookup after compose refresh

Next:
- strengthen live-stack M9 validation beyond the current boundary/smoke checks into richer workflow scenarios
- expand the orchestrator-side brain gateway if additional fact/query endpoints are needed
- decide whether brain routing decision callbacks should be promoted from event ingestion to a typed contract
