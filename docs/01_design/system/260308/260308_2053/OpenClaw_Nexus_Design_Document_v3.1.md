# OpenClaw Nexus vNext
## Design Document v3.1
## Date: 2026-03-08
## Supersedes: docs/01_design/system/260308/OpenClaw_Nexus_Design_Document_v3.md

---

## Changelog from v3

| Section | Change |
|---------|--------|
| Section 9.4 | Implementation Step Output Format: added M5 diff-first execution mode and feature gate reference |
| Section 19 | Complexity Budget: added M5 new file entries (`patch_bundle_service.js`, `dag_scheduler.js`, `context_budget_policy.json`) |
| Section 23 | System Risk Register: updated R-1 status for M5 context budget tracking; added R-8 (patch anchor reliability) |
| Section 24 | Future Roadmap: updated M5 items from "deferred" to "in progress" with task list cross-references |
| Section 26 (NEW) | Context Budget Policy: externalized threshold governance specification |

Review source: `docs/90_archive/260308/OpenClaw_Nexus_Engineering_Task_List_M5_v2.md`

---

## Changelog from v2 (preserved)

| Section | Change |
|---------|--------|
| Section 5 | Brain Router: added Section 5.5 — current implementation clarification (heuristic-only, no LLM) |
| Section 5.4 | Brain Router: added unknown intent confirmation step behavior |
| Section 6 (NEW) | LLM Provider Registry: full specification for provider config, role policy, and dispatcher |
| Section 6.3 | Role Policy: added `secondary_model` + `retry_policy` fields (review adoption R-3.1 / R-3.2) |
| Section 6.5 | LLM Dispatcher: added retry mechanism and model fallback logic |
| Sections 6–22 | Renumbered to 7–23 due to insertion of new Section 6 |
| Section 8.2 (was 7.2) | Agent Contract Template: added `llm_role` field |
| Section 9.3 | QA Verify: added two-layer validation (deterministic + semantic) |
| Section 12 (was 11) | Prompt Script Registry: removed redundant `model` field; now governed by llm_role_policy |
| Section 19 (was 18) | Complexity Budget: added `llm_dispatcher.js` entry |
| Section 22 (was 21) | Definition of Success: added item 11 (LLM routing verification) |
| Section 23 (NEW) | System Risk Register: structural risks and current mitigations |

Review source: `docs/90_archive/260308/OpenClaw_Architecture_Review_and_Risk_Register.md`

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
Frontend, Backend, Integration, Testing: primary output is executable changes, implementation files, tests, and runbooks.

### 2.7 Internal structure is also governed
Module size, layering, and internal coupling are subject to the same governance discipline as feature scope. Structural entropy is a first-class risk.

### 2.8 Right model for the right role
Different execution roles have different inference requirements. Planning roles require strong reasoning (cloud models). Execution roles prioritize cost efficiency and code generation quality (local models). LLM assignment must be explicit, policy-driven, and auditable.

---

## 3. High-Level Architecture

### 3.1 Input Layer
- Human natural-language input
- Primary gateway: Discord
- Future gateways: Web UI, CLI, API, Scheduled triggers

### 3.2 Brain Router Layer
Responsibilities:
- Parse user input
- Classify intent via heuristic rules + deterministic policy override
- Detect complexity and orchestration requirement
- Normalize user request into structured task envelope

Possible intent classes: chat, coding, quant, docs, research, ops, unknown

Possible decision outcomes: direct_reply, single_agent, orchestrated_workflow, human_review_required

> Note: Brain Router does NOT call an LLM directly. See Section 5.5.

### 3.3 OpenClaw Orchestration Layer
Responsibilities:
- Receive structured task envelope
- Select workflow template
- Assign role-based agents (each with explicit LLM assignment via LLM Dispatcher)
- Manage state transitions
- Invoke tools/workers
- Collect artifacts
- Return execution result

### 3.4 Execution Layer
Current executors:
- worker-coder
- worker-quant
- prompt-script agents (dispatched via LLM Dispatcher)
- Codex adapter
- OpenCode adapter

### 3.5 Artifact Layer
Stores: design docs, task lists, implementation files, screenshots, reports, QA summaries, workflow logs.

### 3.6 Memory / Context Layer
Stores: task history, project context, reusable constraints, prior design docs, execution traces for replay.

### 3.7 Orchestrator Internal Layers

The orchestrator is a single deployable service, but internally it must follow a strict 4-layer structure:

```
┌─────────────────────────────────────────────┐
│  Layer 1: Transport/Adapter Layer           │
│  - Discord adapter (event in, reply out)    │
│  - HTTP route definitions (thin)            │
│  - Input normalization only                 │
├─────────────────────────────────────────────┤
│  Layer 2: Service Layer (vnext/)            │
│  - Brain Router                             │
│  - LLM Dispatcher (NEW)                     │
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
│  - Memory Reader / Writer                  │
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

### 5.4 Determinism Requirement

The Brain Router operates in two phases:

**Phase A — Heuristic Classification (current implementation)**
Use regex patterns and keyword matching to produce a candidate intent class and complexity estimate.

**Phase B — Policy Override Layer**
Apply deterministic rules on top of Phase A output:
- `/coder` prefix → force `coding` + `orchestrated_workflow`
- Input < 3 tokens → force `chat`
- Explicit financial keywords → force `human_review_required`
- Unknown intent with execution cues → trigger confirmation step (see below)
- Unknown intent with no execution cues → downgrade to `chat`

**Unknown Intent Confirmation Step (adopted from review R-3.4)**

When intent resolves to `unknown` but the input contains execution-oriented keywords (e.g., build, implement, create, fix, develop, 开发, 构建, 修复), the router must not silently fall back to chat mode. Instead it emits a `clarification_required` decision with a structured confirmation prompt:

> "这看起来像是一个开发任务。是否要启动 Coding Team 工作流？回复 /yes 确认，或描述更多细节。"

The user's `/yes` reply re-routes through the Brain Router with a forced `/coder` prefix applied. This prevents execution requests from being silently swallowed by the chat path (Risk R-6: Router Misclassification).

### 5.5 Brain Router Current Implementation Clarification (NEW)

**The Brain Router does NOT call any LLM.** The current implementation (`src/vnext/brain_router.js`) uses pure heuristic/regex classification. The Brain service (`http://brain:5000/run`) is a separate Docker service accessed via inter-service RPC — it is not an LLM provider and is not governed by the LLM Provider Registry.

Adding LLM-based classification (Phase A upgrade) is a future workstream. It will require an independent architectural review and is explicitly out of scope for M4.

The LLM Dispatcher (Section 6) governs agent execution calls only, not Brain Router routing.

### 5.6 Escalation Rules
Escalate to OpenClaw when:
- Multiple roles are needed
- Multiple artifacts are expected
- Approval checkpoints are required
- The task spans design + implementation + verification
- The task includes external tool invocations

---

## 6. LLM Provider Registry (NEW)

### 6.1 Purpose

Different execution roles have different inference requirements:
- Planning roles (PM, Architect) require strong structured reasoning → cloud API preferred
- Execution roles (Backend, Frontend, QA) require code generation quality and cost efficiency → local model preferred

The LLM Provider Registry provides the configuration and policy layer that routes each agent role to the appropriate LLM provider. All LLM calls in the agent execution path must go through the LLM Dispatcher (`src/vnext/llm_dispatcher.js`).

### 6.2 Provider Configuration

File: `orchestrator/configs/llm_providers.json`

This file contains infrastructure-level configuration only (endpoints, auth references, available models). It is the responsibility of the operator (DevOps/infrastructure), not the developer.

```json
{
  "cloud_qwen": {
    "type": "cloud_api",
    "endpoint_env": "QWEN_BASE_URL",
    "auth_env": "QWEN_API_KEY",
    "timeout_ms": 30000,
    "available_models": ["qwen-plus", "qwen-max", "qwen-coder-next"]
  },
  "local_ollama": {
    "type": "local_ollama",
    "endpoint_env": "OLLAMA_BASE_URL",
    "timeout_ms": 240000,
    "available_models": ["deepseek-r1:32b", "qwen2.5-coder:7b"]
  }
}
```

### 6.3 Role Policy Configuration

File: `orchestrator/configs/llm_role_policy.json`

This file contains business policy: which role uses which provider and model. It is the responsibility of the architect/PM. This is the **single source of truth** for role-model assignment.

```json
{
  "version": "1.1.0",
  "roles": {
    "pm":        { "provider": "cloud_qwen",   "model": "qwen-max" },
    "architect": { "provider": "cloud_qwen",   "model": "qwen-max" },
    "backend":   { "provider": "local_ollama", "model": "deepseek-r1:32b",  "secondary_model": "qwen2.5-coder:7b" },
    "frontend":  { "provider": "local_ollama", "model": "deepseek-r1:32b",  "secondary_model": "qwen2.5-coder:7b" },
    "qa":        { "provider": "local_ollama", "model": "deepseek-r1:32b",  "secondary_model": "qwen2.5-coder:7b" },
    "release":   { "provider": "local_ollama", "model": "deepseek-r1:32b",  "secondary_model": "qwen2.5-coder:7b" }
  },
  "retry_policy": {
    "strategy": "exponential_backoff",
    "retries": 3,
    "base_delay_ms": 2000
  },
  "fallback_policy": "model_fallback"
}
```

**`secondary_model` 使用场景（采纳自 review R-3.1）：**

本地 32B 模型（deepseek-r1:32b）在典型配置（RX 7900 XTX / 24GB VRAM）下约占用 19–21GB 显存。以下情况自动降级至 secondary_model（qwen2.5-coder:7b）：
- OOM 错误或 Ollama 返回资源不足
- 单次调用超过 latency 阈值（默认 `OLLAMA_CHAT_TIMEOUT_MS`）
- 输入 context 超过模型 `num_ctx` 配置上限

`fallback_policy` 选项：
- `model_fallback` — primary 失败 → 自动切换 secondary_model，记录降级日志
- `fail_fast` — 任何失败直接抛 typed error（可按需切换，用于调试环境）

### 6.4 Override Priority Chain

```
request-level override (explicit in task envelope)
    ↓
role-level policy (llm_role_policy.json)
    ↓
system default (fail_fast)
```

### 6.5 LLM Dispatcher

Module: `src/vnext/llm_dispatcher.js` (Layer 2 — Service)

The Dispatcher is the single entry point for all agent LLM calls within the execution path.

```
dispatch(role, messages, overrides?)
  → reads llm_role_policy.json for role assignment
  → reads llm_providers.json for provider config
  → routes to callQwenChat() or callLocalOllamaChat()
  → returns { content, model, provider, latency_ms, used_fallback: bool }
```

**重试机制（采纳自 review R-3.2）：**

传输层与提供者层策略分离：

- **传输层重试**：网络超时、临时 HTTP 错误 → 指数退避重试（3 次，基础延迟 2s）。同一 model 重试，不切换 model。
- **提供者层降级**：OOM、持续超时（超过重试耗尽）、context overflow → 切换至 `secondary_model`，记录降级事件。
- **最终失败**：secondary_model 也失败 → 抛出 typed error `{ code: 'LLM_DISPATCH_FAILED', role, provider, model, fallback_attempted: true, cause }`

这样区分了"偶发性网络抖动"（应重试）和"资源结构性不足"（应降级），避免将可恢复错误直接暴露为工作流失败。

Structured log per call:
```
[llm_dispatcher] role=backend provider=local_ollama model=deepseek-r1:32b latency=8420ms status=ok
[llm_dispatcher] role=backend provider=local_ollama model=deepseek-r1:32b status=oom_fallback → secondary=qwen2.5-coder:7b
[llm_dispatcher] role=architect provider=cloud_qwen model=qwen-max latency=1203ms status=ok retry=1
```

### 6.6 Scope Boundary

The LLM Dispatcher governs:
- PM agent execution
- Architect agent execution
- Backend implementation LLM calls
- Frontend implementation LLM calls
- QA verify LLM calls
- Release pack LLM calls

The LLM Dispatcher does NOT govern:
- Brain Router routing (heuristic-only, no LLM)
- Brain service RPC (`http://brain:5000/run`) — this is inter-service communication
- Direct chat replies via local Ollama (handled by chat_entrypoint.js directly)

### 6.7 Provider Health Check

On orchestrator startup, `validateProviders()` must be called:
- For `cloud_api`: verify API key environment variable is set (no live network call required at startup)
- For `local_ollama`: verify the configured model is listed in Ollama's `/api/tags` endpoint
- On failure: log warning per provider, do not crash startup; mark provider as `degraded` in runtime state

---

## 7. OpenClaw Role in the System

OpenClaw is the orchestration trunk.

Responsibilities:
- Workflow planning
- Role dispatch (via LLM Dispatcher for agent execution)
- Tool invocation
- Execution tracking
- Artifact aggregation
- Retry/recovery hooks
- Audit trail

Not responsible for:
- First-pass intent recognition
- Direct chat handling
- Unrestricted tool execution without policy
- Role definition itself

---

## 8. Agent Model

### 8.1 Agent Categories

#### A. Planning Agents
- PM Agent
- Architect Engineer Agent (see Section 8.3 for full spec)
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

### 8.2 Agent Contract Template

Each agent defines:
- `name`
- `mission`
- `llm_role` — key into `llm_role_policy.json` (single source of truth for model assignment)
- `input_schema`
- `output_schema`
- `tools_allowed`
- `forbidden_actions`
- `dependencies`
- `success_criteria`
- `retry_policy`
- `escalation_policy`

The `llm_role` field replaces the deprecated `model` field previously used in `prompt_script_registry.json`. No agent contract may hard-code a model name directly.

### 8.3 Architect Engineer Agent — Full Specification

See `docs/01_design/system/260307/Architect_Engineer_Role_Contract.md` for the complete specification. That document remains authoritative for M4.

---

## 9. Coding Team Design

### 9.1 Mission
Convert a user's product request into structured design documentation, implementation plan, code changes, tests, and verification artifacts.

### 9.2 Standard Workflow

```
Step 0: PM Spec          (role: pm,               llm: cloud_qwen / qwen-max)
Step 1: Arch Design      (role: architect,         llm: cloud_qwen / qwen-max)
Step 2: Backend Impl     (role: backend,           llm: local_ollama / deepseek-r1:32b)
Step 3: Frontend Impl    (role: frontend,          llm: local_ollama / deepseek-r1:32b)
Step 4: QA Verify        (role: qa,                llm: local_ollama / deepseek-r1:32b)
Step 5: Release Pack     (role: release,           llm: local_ollama / deepseek-r1:32b)
```

UI/UX step remains optional and inserts between steps 1 and 2 when project type requires it.

### 9.3 QA Verify — Two-Layer Validation (adopted from review R-3.3)

Schema validation alone only ensures valid JSON, required fields, and correct types. It cannot catch semantic errors — e.g., a Backend that produces a structurally valid API schema but with incorrect business logic, which the Frontend then consumes without error.

The QA Verify step must implement two validation layers:

**Layer 1 — Deterministic Checks**
- Required artifacts exist (`be_changes/`, `fe_changes/`, run instructions)
- All handoff schemas validate (AJV)
- Test files are present

**Layer 2 — Semantic Checks**
- API contracts declared by Backend match what Frontend actually references in its implementation
- Implemented features align with PM acceptance criteria (`plan/acceptance.json`)
- `scope_constraints` from Architect handoff are not violated

Semantic check failures are reported in `verify/qa_report.json` as `overall_status: "fail"` with per-check detail. The workflow does not proceed to Release on a semantic failure.

### 9.4 Implementation Step Output Format

Backend and Frontend implementation steps support two output modes:

**Mode A — Full-File Output (M4 default, M5 fallback)**
- `impl/be_changes/` or `impl/fe_changes/` — complete new/modified files (not diffs)
- `impl/be_notes.md` or `impl/fe_notes.md` — implementation decisions and run instructions

**Mode B — Structured Patch Output (M5 default when applicable)**
- structured patch bundle conforming to `coding_team_patch_bundle.schema.json`
- uses content-anchor addressing (not line numbers) for reliable LLM-generated patches
- operations are ordered and applied sequentially against prior operation results
- prompt scripts `backend.impl.v2` / `frontend.impl.v2` instruct LLM to produce patch format

**Mode selection logic:**
1. Check feature gate `execution.diff_first_enabled` — if disabled, use Mode A
2. Check if target files exist in workspace — if not, use Mode A (create_file operation or full-file)
3. Check context budget — if injecting target file content would push prompt into `overflow_risk`, use Mode A
4. Otherwise, use Mode B with automatic fallback to Mode A if patch application fails

The `execution_mode_used` field in step result records which mode was actually applied.

**Rationale for M4 approach (preserved):** Diff generation requires a known file baseline. In the Coding Team workflow sandbox, a reliable baseline is not guaranteed for new files. Complete file output remains the safest option for file creation scenarios.

**Rationale for M5 evolution:** For existing files, full-file output wastes tokens and increases truncation risk. Content-anchor-based patching reduces output size by ≥30% while maintaining correctness through anchor validation. Full-file fallback ensures no regression.

### 9.5 Inter-Step Handoff Contracts

Each step transition is governed by a typed handoff schema:

| Handoff | Schema File | Status |
|---------|------------|--------|
| PM → Architect | `coding_team_pm_handoff.schema.json` | Exists (M3) |
| Architect → Impl | `coding_team_arch_handoff.schema.json` | Exists (M3) |
| BE → FE | `coding_team_be_to_fe_handoff.schema.json` | **To be defined in M4 (WS-17-00)** |
| Impl → QA | `coding_team_impl_to_qa_handoff.schema.json` | **To be defined in M4 (WS-17-00)** |
| QA → Release | `coding_team_qa_to_release_handoff.schema.json` | **To be defined in M4 (WS-17-00)** |

No step may proceed without a valid handoff from the prior step.

---

## 10. Non-Coding Agent Workflows

### 10.1 PM/Planning Workflow
Primary output: design doc, task list, acceptance matrix.

### 10.2 UI Workflow
Primary output: UI spec, component map, state/event matrix.

### 10.3 Research Workflow
Primary output: research brief, options matrix, recommendation memo.

---

## 11. Tooling Strategy

### 11.1 Tool Classes
- Reasoning models (via LLM Dispatcher)
- Code generation tools
- Browser tools
- Shell/sandbox tools
- Quant analysis tools
- Artifact generation tools

### 11.2 Current Tool Mapping
- Direct conversation → Brain LLM (via Brain service, not LLM Dispatcher)
- Structured planning docs → prompt-script agents + LLM Dispatcher
- Coding execution → Codex / OpenCode / worker-coder
- Quant execution → worker-quant
- Browser evidence → OpenClaw browser tools

### 11.3 Tool Abstraction Requirement
Tools must be wrapped behind stable interfaces so that providers can be swapped without changing upstream orchestration logic.

---

## 12. Prompt Script Registry

Each script definition includes:
- `script_id`
- `target_agent`
- `llm_role` — reference to `llm_role_policy.json` key (replaces deprecated `model` field)
- `input_schema`
- `output_schema`
- `allowed_tools`
- `artifact_type`
- `validation_rules`

The `model` field previously present in registry entries is **deprecated and removed**. Model assignment is governed exclusively by `llm_role_policy.json`. This eliminates the dual-source-of-truth problem.

Current registered scripts: `pm.design_doc.v1`, `architect.system_spec.v1`, `architect.system_spec.v2`, `qa.test_plan.v1`

M4 will add: `backend.impl.v1`, `frontend.impl.v1`, `qa.verify.v1`, `release.pack.v1`

---

## 13. Workflow Patterns

### 13.1 Direct Reply Pattern
- Chat, quick Q&A, no artifact requirement

### 13.2 Single-Agent Pattern
- Simple doc generation, simple patch, narrow analysis

### 13.3 Multi-Agent Workflow Pattern
- Project implementation, multi-role decomposition, design → build → verify flows

### 13.4 Human Approval Pattern
- Destructive actions, risky patches, production deployments, financial actions, external communication

---

## 14. Memory / Context Layer

### 14.1 Purpose
Enable agents to access project history, prior design decisions, and recurring constraints without relying on per-request context window capacity.

### 14.2 Minimum Required Store

| Store | Key | Value | Use |
|-------|-----|-------|-----|
| Project Context | `project:{project_id}` | JSON: active tech stack, repo root, constraints | Architect input |
| Prior ADRs | `adr:{project_id}:{adr_id}` | Markdown text | Architect reference |
| Task History | `task:{run_id}` | JSON: task_id, intent, outcome, artifacts | Replay and debugging |

### 14.3 Access Pattern
- Read-only at agent runtime (agents may not write to memory directly)
- Written by orchestrator after workflow completion (`memory_writer.js`)
- No LLM-driven memory summarization in this phase

### 14.4 Storage Backend
- Current: flat JSON files per project under `artifacts/memory/{project_id}/`
- Upgrade path: Redis key-value store with TTL

### 14.5 Non-Scope
- No vector search
- No semantic retrieval
- No LLM-driven memory compression
- No cross-project context sharing

---

## 15. Quality Gates

No workflow is complete without explicit quality gates:
- Schema-valid output
- Artifact generated
- Execution log saved
- Validation passed
- Failure reason captured if unsuccessful

For coding workflows:
- Files generated (not diffs)
- Tests generated or executed
- Run instructions produced
- Rollback note produced where applicable

---

## 16. Guardrails

### 16.1 Risk Policy
Actions labeled: low / medium / high risk.

### 16.2 High-Risk Actions
Require manual approval: deleting files, modifying secrets, production deployment, broker/trading execution, external publishing, system-wide config changes.

### 16.3 Local Safety
The system must respect local-only execution constraints and avoid unintended cloud dependency.

---

## 17. Observability

### 17.1 Required Notification Points

| Event | Trigger | Output |
|-------|---------|--------|
| `workflow.started` | Workflow run created | "Workflow started: {workflow_id}, step 1 of N: {step_title}" |
| `step.completed` | Step transitions to next | "{step_title} completed. Starting {next_step_title}..." |
| `step.approval_required` | Approval gate triggered | "Step {step_title} requires approval. Use /approve or /reject." |
| `workflow.succeeded` | Terminal succeeded | "Workflow complete. Artifacts ready at {artifact_root}." |
| `workflow.failed` | Terminal failed | "Workflow failed at {step_title}: {error_code}. Details: {error_summary}" |

All notifications must use deterministic template strings (no LLM generation) and pass through the `workflow_runtime_notifier` → `workflow_notification_delivery` path.

### 17.2 LLM Call Observability
Every LLM Dispatcher call must produce a structured log entry:
```
[llm_dispatcher] role={role} provider={provider} model={model} latency={ms}ms status={ok|error}
```
This enables per-role LLM usage analysis and cost attribution.

### 17.3 Observability API (Existing)
- `GET /runs/:run_id/status`
- `GET /runs/:run_id/timeline`
- `GET /runs/:run_id/artifacts`
- `GET /workflow-runs/:workflow_run_id`

---

## 18. Route Consolidation Policy

### 18.1 Canonical Entry Points

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

Deprecated routes (`POST /execute-tool`, `POST /debug/plan`, `POST /workflows`, `GET /ui/approvals`) were removed in M3.

---

## 19. Orchestrator Complexity Budget

Hard limits on module size:

| Module | Layer | Max Lines | Current | Status |
|--------|-------|-----------|---------|--------|
| `src/index.js` | 1 (Transport) | 800 | 546 | OK |
| `src/workflow_engine.js` | 3 (Domain) | 600 | 431 (M5 projected: 511–551) | OK — extract to `dag_scheduler.js` if >560 |
| `src/vnext/*.js` (per file) | 2 (Service) | 300 | ≤275 | OK |
| `src/vnext/llm_dispatcher.js` | 2 (Service) | 250 | 0 (M4) | — (revised: +30 for retry+fallback) |
| `src/domain/*.js` (per file) | 3 (Domain) | 500 | ≤264 | OK |
| `src/domain/patch_bundle_service.js` (NEW) | 3 (Domain) | 400 | 0 (M5) | — |
| `src/domain/dag_scheduler.js` (conditional NEW) | 3 (Domain) | 300 | 0 (M5, only if workflow_engine.js >560) | — |
| `src/adapters/*.js` (per file) | 1 (Transport) | 400 | ≤286 | OK |
| `src/data/*.js` (per file) | 4 (Infra) | 250 | compliant | OK |

New files exceeding their budget before first commit require architectural review.

---

## 20. State Machine

Task lifecycle:
```
received → classified → normalized → planned → dispatched → running
→ waiting_for_dependency → awaiting_approval → verifying → completed
→ partial_failure → failed → canceled
```

`partial_failure` (NEW — M5): occurs when parallel steps produce mixed results (e.g., BE succeeds but FE fails). The workflow may retry only the failed step without re-executing succeeded steps.

Each transition must be logged and queryable.

---

## 21. Non-Functional Requirements

### NFR-01 Determinism
Routing and workflow selection must be deterministic and policy-backed.

### NFR-02 Traceability
Every artifact and state transition must be attributable to a task and role.

### NFR-03 Replaceability
Tool providers and LLM providers must be swappable without changing top-level workflow contracts.

### NFR-04 Local-first
No workflow should require cloud-only dependencies unless explicitly approved.

### NFR-05 Recoverability
Workflow failure must not destroy task history or artifacts.

### NFR-06 Structural Maintainability
No single module may exceed its complexity budget. Layer boundaries must be respected.

### NFR-07 LLM Assignment Auditability (NEW)
Every LLM call in the agent execution path must be attributable to a role, provider, and model. The role-model mapping must be readable from a single config file without tracing code.

---

## 22. Definition of Success

The system is considered successful when:

1. A Discord request is correctly classified by Brain Router (heuristic + policy override).
2. Chat requests are answered directly without orchestration.
3. Coding requests trigger a structured Coding Team workflow.
4. The Architect step produces a real architecture blueprint (ADRs, interfaces, workplan).
5. PM/Architect tasks produce reproducible high-quality documents using cloud LLM.
6. Backend/Frontend tasks produce complete implementation files using local LLM.
7. Quant tasks route into quant worker without contaminating coding workflows.
8. All artifacts, states, and logs are traceable end-to-end.
9. Step-level progress notifications are emitted to Discord at each workflow transition.
10. The orchestrator codebase respects the 4-layer structure and complexity budget.
11. Every agent LLM call is routed through LLM Dispatcher; model assignment is readable from `llm_role_policy.json`; changing a role's model requires editing only that file. (NEW)

---

## 23. System Risk Register

Source: `docs/90_archive/260308/OpenClaw_Architecture_Review_and_Risk_Register.md`

This register lists structural risks identified in the architecture review. Each risk has a current mitigation status.

| Risk ID | Risk | Severity | M4 Mitigation | M5 Update | Status |
|---------|------|----------|---------------|-----------|--------|
| R-1 | Context Explosion — large artifacts between steps overflow model context | High | `num_ctx` config + secondary_model fallback on overflow | M5: explicit context budget tracking per step, thresholds in `context_budget_policy.json`, release pack aggregation | M5: in progress |
| R-2 | Agent Drift — agents deviate from task scope across steps | Medium | `scope_constraints` in handoff + QA semantic validation | — | M4: semantic check adopted |
| R-3 | Model Non-Determinism — LLM responses vary across runs | Medium | Temperature control + structured prompts + deterministic validators | — | Ongoing |
| R-4 | Pipeline Garbage Propagation — bad output in step N corrupts step N+1 | High | Handoff schema validation gates + QA two-layer validation | M5: patch anchor validation adds per-operation error detection | M4: gates added; M5: enhanced |
| R-5 | Local LLM Availability — OOM, Ollama crash, VRAM exhaustion | High | secondary_model fallback + exponential retry + startup health check | — | M4: adopted (R-3.1, R-3.2) |
| R-6 | Router Misclassification — execution requests silently become chat | Medium | Unknown intent confirmation step | — | M4: adopted (R-3.4) |
| R-7 | Observability Gaps — LLM calls and state transitions not traceable | Medium | Structured LLM Dispatcher logs + workflow event logs + run_id traceability | M5: context budget reports add per-step size observability | M4: dispatcher logs; M5: enhanced |
| R-8 (NEW) | Patch Anchor Mismatch — LLM-generated content anchors fail to match target file | Medium | — | M5: content-anchor validation before each operation; typed error on mismatch; automatic full-file fallback; feature gate for global disable | M5: mitigated by design |

---

## 24. Future Roadmap (Deferred from M4)

Items explicitly deferred from M4 scope based on architecture review:

| Item | Target | Status | Reference |
|------|--------|--------|-----------|
| Structured diff / AST patching (replace full file output) | M5 | **In progress** — content-anchor patch model, feature-gated | M5 WS-19 |
| BE + FE parallel execution (workflow DAG) | M5 | **In progress** — DAG primitive + feasibility gate, default remains sequential | M5 WS-21 |
| Context budget tracking (per-step token monitoring) | M5 | **In progress** — externalized thresholds in `context_budget_policy.json` | M5 WS-20 |
| LLM classification in Brain Router (Phase A upgrade) | Future | Deferred — current heuristic routing works; LLM adds non-determinism risk before pipeline is stable | — |
| Adaptive model routing (auto-select model by task complexity) | M7 | Deferred — requires stable baseline performance data first | — |

---

## 25. Final Product Positioning

> Note: Section numbering updated in v3.1. Former Section 25 is now Section 27.

---

## 26. Context Budget Policy (NEW — M5)

### 26.1 Purpose

Context budget thresholds govern when a workflow step's prompt or artifact size is classified as `ok`, `warning`, or `overflow_risk`. These thresholds must be externalized in a policy file, consistent with the `llm_role_policy.json` governance pattern.

### 26.2 Configuration File

File: `orchestrator/configs/context_budget_policy.json`

This file is the **single source of truth** for context budget thresholds. No service may hardcode thresholds.

```json
{
  "version": "1.0.0",
  "default_thresholds": {
    "warning_prompt_chars": 80000,
    "overflow_risk_prompt_chars": 120000,
    "warning_artifact_bytes": 500000,
    "overflow_risk_artifact_bytes": 1000000
  },
  "role_overrides": {
    "architect": {
      "warning_prompt_chars": 100000,
      "overflow_risk_prompt_chars": 150000
    }
  }
}
```

### 26.3 Integration Points

- **Per-step budget reports** (M5 WS-20-02): read thresholds from this file to classify step status
- **Diff-first execution** (M5 WS-19-03): if injecting target file content would push prompt into `overflow_risk`, pre-emptively use full-file mode
- **Release pack validation** (M5 WS-20-03): missing budget reports fail validation

### 26.4 Operational Notes

- Changing thresholds requires editing only this file; no code changes or restarts needed
- Thresholds may be tuned per-role to reflect different context window sizes (e.g., cloud models have larger windows)

---

## 27. Final Product Positioning

OpenClaw Nexus is a local-first AI operating system composed of:
- One input plane (Discord gateway)
- One routing brain (heuristic + deterministic policy layer — no LLM)
- One orchestration trunk (with enforced internal layering)
- One LLM dispatch layer (role-policy-driven, cloud/local split, with retry and model fallback)
- Multiple specialized execution teams
- One memory/context store (file-based, grows over time)

The first production-grade team to be completed is the Coding Team, where planning roles use cloud reasoning models and execution roles use local inference models, all governed by a single policy file. LLM availability risks (OOM, timeout) are handled at the dispatcher layer, invisible to the workflow engine above.
