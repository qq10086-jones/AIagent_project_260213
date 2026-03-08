# OpenClaw Nexus vNext
## Engineering Task List — Milestone 4
## Date: 2026-03-08
## Focus: LLM Routing Layer + Coding Team Execution Chain

---

## 0. 前情提要 — Milestone 1 到 3 完成了什么

本节为多 AI 会话提供历史背景，确保任何新接手的 AI 参与者能理解系统的当前状态。

### Milestone 1 — vNext 基础管道（2026-03-06 关闭）

建立 OpenClaw Nexus 的核心执行管道骨架：

| 工作项 | 内容 |
|--------|------|
| WS-01 Task Envelope | 定义标准化任务信封格式，所有请求经此归一化 |
| WS-02 Brain Router | 基于启发式规则的意图分类（coding/quant/chat/ops/docs），输出结构化 JSON |
| WS-03 Runtime Dispatch | vNext 请求调度入口，将任务信封路由至正确执行路径 |
| WS-04 Coding Team 合约 | PM / Architect / QA 角色的 handoff schema 初版 |
| WS-05 Artifact 打包 | 工作流 artifact 收集、打包、存储机制 |
| WS-06 Runtime Contract Hardening | 合约 JSON Schema 校验，validate_registry.js 初版 |
| WS-07 Tool Adapter Interface | 工具适配器注册表，统一工具调用接口 |
| WS-08 Artifact Timeline Replay | 工作流步骤时间线记录，支持回放 |
| WS-09 Tool Permission Guard | 工具调用权限校验层 |
| WS-10 Observability / Notifications | 步骤级通知（workflow.started / step.completed / workflow.succeeded / failed）|

**Milestone 1 结束状态：** North Star 管道可端到端运行（PM → Architect），工具调用有权限守卫，步骤级通知推送至 Discord。

---

### Milestone 2 — vNext 端到端上线（2026-03-07 关闭）

将 M1 骨架与真实运行时打通，完成线上可用的 E2E 验证：

| 工作项 | 内容 |
|--------|------|
| E2E 运行时验证 | `validate:live_vnext_runtime` + `validate:live_workflow_runtime` 全部 pass |
| Discord 端到端 | Discord 消息 → Brain Router → TaskEnvelope → Workflow 完整链路打通 |
| 审批流 | `/approve` / `/reject` Discord 指令触发 approval entrypoint |
| Workflow Notification Delivery | 通知经 `workflow_runtime_notifier` → `workflow_notification_delivery` 路径发出 |
| Coding Team Handoff 验证 | PM → Architect handoff 校验通过，artifact 完整性校验上线 |

**Milestone 2 结束状态：** 系统可在真实 Discord 环境中接收请求、运行 PM+Architect 步骤、产出 artifact、推送通知，并支持人工审批。

---

### Milestone 3 — 结构加固（2026-03-07 关闭）

专项里程碑，不扩展功能，仅做结构质量修复：

| 工作项 | 内容 | 结果 |
|--------|------|------|
| WS-11-02 Discord 适配器提取 | `discord_gateway.js` + `discord_message_handler.js` 独立模块 | Layer 1 边界清晰 |
| WS-11-03 Repository 层提取 | `src/data/` 统一持有所有 SQL；Layer 1/2 零原始 SQL | 4 层架构合规 |
| WS-11-04 `workflow_engine.js` 拆解 | 1581 行 → **431 行**；拆出 5 个 domain 子模块 | 低于预算 600 行 |
| WS-11-05 `index.js` 瘦身 | 2424 行 → **546 行**；拆出 7 个 vnext/adapter 模块 | 低于预算 800 行 |
| WS-12 Architect Engineer Hardening | Architect 步骤强制产出 `interfaces.md` + ADR；canary 覆盖正负用例 | handoff 有实质内容 |
| WS-13 Brain Router Policy Layer | 确定性策略覆盖模块上线，`/coder` 前缀等 5 条规则 | 路由可测试 |
| WS-14 Route Consolidation | 4 条废弃路由移除（`/execute-tool`、`/debug/plan` 等） | 无历史负债 |
| WS-15 Memory Layer Stub | schema + 文件读写模块 + Architect 提示词注入（存疑，见 WS-18-00）| 最小内存层到位 |
| P1-03 Infra 连接边界 | `src/infra/runtime_connections.js` 统一管理 Redis/pg/S3 创建 | Layer 4 边界清晰 |

**Milestone 3 结束状态：** 32/32 测试通过，所有模块复杂度预算合规，4 层架构边界完整，Architect 输出有实质性技术内容。

---

### M1–M3 之后，系统的现状与空白

**已有（可用）：**
- North Star 管道：Discord → Brain Router → TaskEnvelope → Workflow Engine → PM + Architect → Artifact
- 4 层内部架构（Layer 1–4 边界清晰）
- PM → Architect handoff 全链路（含 ADR、interfaces.md、risk_report）
- 步骤级通知、审批流、工具权限守卫、artifact 打包

**空白（M4 要填补的）：**
- Backend / Frontend / QA / Release 执行步骤仍为空桩
- 没有统一的 LLM 路由层：模型名散落在代码/配置中，cloud 与 local 的选择无策略依据
- BE→FE、Impl→QA、QA→Release 三条 handoff schema 未定义
- Memory 注入是否真实工作有待核查（WS-18-00）

---

## 1. Objective

Milestone 4 addresses two parallel objectives that were blocked by M3 structural hardening:

1. **LLM Routing Layer** — Establish a policy-driven, auditable model assignment system so that different agent roles use appropriate LLM providers (cloud vs local) without hardcoded model references scattered across the codebase.

2. **Coding Team Execution Chain** — Complete the implementation steps (Backend, Frontend, QA, Release) that remain as stubs after M3. The full PM → Arch → BE → FE → QA → Release pipeline must be runnable end-to-end.

This milestone closes the gap between the system's structural foundation (M1–M3) and its stated primary value: converting a natural-language request into executable code artifacts.

---

## 2. Prerequisites

M4 must not start until:
- M3 is fully closed (confirmed: 2026-03-07)
- `PROGRESS_LATEST.md` reflects M3 closure
- WS-18-00 (Memory Layer debt check) is completed and its outcome documented

---

## 3. Design Decisions Embedded in This Milestone

### D1 — Minimum Selectable LLM Node = Agent Role
LLM model assignment is per agent role (pm, architect, backend, etc.), not per layer and not per workflow step. Override priority: request > role policy > model_fallback.

### D2 — Two Separate Config Files
`llm_providers.json` (infra: endpoints, auth) and `llm_role_policy.json` (policy: role → model). Maintained by different owners; must not be merged.

### D3 — Brain Router Is Not in LLM Dispatcher Scope
Brain Router uses heuristic/regex routing with no LLM calls. Brain service (`brain:5000`) is inter-service RPC, not an LLM provider. Both are out of LLM Dispatcher scope.

### D4 — Single Source of Truth for Role-Model Assignment
`prompt_script_registry.json` deprecated `model` field is removed. `llm_role_policy.json` is the only place role-model assignment lives.

### D5 — Implementation Output Format: Files, Not Diffs
Backend and Frontend steps output complete files in `impl/be_changes/` and `impl/fe_changes/` directories. Diff generation is deferred to M5 (structured diff / AST patching).

### D6 — Handoff Schemas Are a Prerequisite, Not Implicit Work
BE→FE, Impl→QA, and QA→Release handoff schemas must be defined before any implementation step code is written. WS-17-00 is a blocking prerequisite.

### D7 — Local Model Fallback (adopted from review R-3.1)
32B model (deepseek-r1:32b) on a typical 24GB VRAM machine carries OOM and latency risk. `llm_role_policy.json` defines `secondary_model: qwen2.5-coder:7b` for all local_ollama roles. Dispatcher falls back automatically on OOM, sustained timeout, or context overflow. `fallback_policy` changes from `fail_fast` to `model_fallback`.

### D8 — Retry and Fallback Policy Separation (adopted from review R-3.2)
Transport-layer errors (network timeout, transient HTTP failure) → exponential backoff retry on same model (3 retries, base 2s). Provider-layer errors (OOM, context overflow) → switch to secondary_model. These are different failure modes and must not be conflated.

### D9 — Unknown Intent Confirmation Step (adopted from review R-3.4)
Brain Router policy: when intent = `unknown` and input contains execution-oriented keywords, emit `clarification_required` decision with a confirmation prompt rather than silently routing to chat. This prevents R-6 (Router Misclassification).

---

## 4. Workstream Overview

| ID | Name | Type | Blocks |
|----|------|------|--------|
| WS-16 | LLM Provider & Model Router | A | WS-17 (execution calls) |
| WS-17 | Coding Team Execution Chain | A | WS-17-05 (E2E canary) |
| WS-18 | Memory Layer Closure | B | — |

WS-16 must complete before WS-17 execution steps begin. WS-17-00 (schema definitions) can run in parallel with WS-16.

---

## 5. Detailed Task List

---

## WS-16 LLM Provider & Model Router

**Type:** Type A / Critical Path
**Pipeline node:** Service Layer — all agent execution paths

---

### WS-16-01 Define LLM Provider Registry (no code)

**Deliverables:**
- `orchestrator/contracts/llm_provider_registry.schema.json` — JSON Schema for `llm_providers.json`
- `orchestrator/contracts/llm_role_policy.schema.json` — JSON Schema for `llm_role_policy.json`
- `orchestrator/configs/llm_providers.json` — initial provider definitions
- `orchestrator/configs/llm_role_policy.json` — initial role assignments

**Initial role assignments (version 1.1.0 — includes secondary_model and retry_policy):**

| Role | Provider | Primary Model | Secondary Model | Rationale |
|------|----------|---------------|-----------------|-----------|
| `pm` | `cloud_qwen` | `qwen-max` | — | Strong planning reasoning; cloud stable |
| `architect` | `cloud_qwen` | `qwen-max` | — | ADR, interface design; cloud stable |
| `backend` | `local_ollama` | `deepseek-r1:32b` | `qwen2.5-coder:7b` | Code gen; fallback on OOM |
| `frontend` | `local_ollama` | `deepseek-r1:32b` | `qwen2.5-coder:7b` | Code gen; fallback on OOM |
| `qa` | `local_ollama` | `deepseek-r1:32b` | `qwen2.5-coder:7b` | Verify; fallback on OOM |
| `release` | `local_ollama` | `deepseek-r1:32b` | `qwen2.5-coder:7b` | Packaging; fallback on OOM |

`retry_policy`: `exponential_backoff`, 3 retries, base 2s delay (transport-layer only)
`fallback_policy`: `model_fallback` (provider-layer: OOM / sustained timeout / context overflow)

**Acceptance criteria:**
- Both config files pass AJV schema validation
- `validate_registry.js` is updated to include these two files in its validation pass
- No model name appears in `prompt_script_registry.json`

**Non-scope:**
- No code changes to any calling module in this task

---

### WS-16-02 Implement LLM Dispatcher

**Deliverable:** `src/vnext/llm_dispatcher.js`

**Interface:**
```js
// Primary dispatch entry point
export async function dispatch(role, messages, overrides = {})
// Returns: { content: string, model: string, provider: string, latency_ms: number, used_fallback: boolean }
// Throws:  { code: 'LLM_DISPATCH_FAILED', role, provider, model, fallback_attempted: boolean, cause }

// Startup health check
export async function validateProviders()
// Returns: { ok: boolean, results: [{ provider, status: 'ok'|'degraded', detail }] }
```

**Requirements:**
- Load `llm_providers.json` and `llm_role_policy.json` once at module init; do not re-read per call
- Resolve provider and model from `llm_role_policy.json` for given `role`; `overrides` takes precedence
- Route to `callQwenChat()` or `callLocalOllamaChat()` from `local_llm_client.js`
- **Transport-layer retry (D8):** On network timeout or transient HTTP error, apply exponential backoff using `retry_policy` from `llm_role_policy.json`. Retry against the same model. Max 3 retries.
- **Provider-layer model fallback (D7):** On OOM error, context overflow, or retries exhausted → switch to `secondary_model` (if defined). Log `status=oom_fallback` or `status=timeout_fallback`. Set `used_fallback: true` in response.
- On unknown role: throw typed error immediately, do not default silently
- `validateProviders()`: for `cloud_api` verify env var is set; for `local_ollama` call `/api/tags` and confirm both `model` and `secondary_model` exist
- Emit structured log per call:
  ```
  [llm_dispatcher] role=backend provider=local_ollama model=deepseek-r1:32b latency=8420ms status=ok
  [llm_dispatcher] role=backend provider=local_ollama model=deepseek-r1:32b status=oom_fallback → secondary=qwen2.5-coder:7b
  [llm_dispatcher] role=architect provider=cloud_qwen model=qwen-max latency=1203ms status=ok retry=1
  ```

**Complexity budget:** ≤ 250 lines (revised upward to accommodate retry + fallback logic)

**Acceptance criteria:**
- Unit tests cover: cloud_api dispatch, local_ollama dispatch, override path, transport retry (3 attempts), OOM fallback to secondary_model, secondary_model also fails → typed error with `fallback_attempted: true`, unknown role typed error, validateProviders pass, validateProviders degraded
- `node --check src/vnext/llm_dispatcher.js` passes

---

### WS-16-03 Remove Deprecated `model` Field from Prompt Script Registry

**Deliverable:** Updated `orchestrator/configs/prompt_scripts/registry.json`

**Requirements:**
- Remove `model` field from all existing script entries (`pm.design_doc.v1`, `architect.system_spec.v1`, `architect.system_spec.v2`, `qa.test_plan.v1`)
- Add `llm_role` field to each script entry (value = role key matching `llm_role_policy.json`)
- Update `validate_registry.js`: reject any script entry that contains `model` field; require `llm_role` field

**Acceptance criteria:**
- `validate_registry.js` passes with updated registry
- No `model` field exists in any script entry

---

### WS-16-04 Wire Prompt Script Execution to LLM Dispatcher

**Deliverables:** Updated caller code in the workflow execution path

**Requirements:**
- Any location that calls `callQwenChat()` or `callLocalOllamaChat()` for a PM/Architect/QA agent execution must be updated to call `llm_dispatcher.dispatch(role, messages)` instead
- Direct calls to these functions remain allowed only in `llm_dispatcher.js` itself and in `chat_entrypoint.js` (direct chat, not agent workflow)
- Brain Router must not be modified — it has no LLM calls and must remain unchanged

**Acceptance criteria:**
- Grep for `callQwenChat` and `callLocalOllamaChat` in the execution path returns only `llm_dispatcher.js` and `chat_entrypoint.js`
- All 32 existing tests still pass

---

### WS-16-05 LLM Dispatcher Canary

**Deliverable:** `orchestrator/scripts/canary_llm_dispatcher.js`

**Coverage:**
- Cloud API dispatch path (stub provider, no real network call)
- Local Ollama dispatch path (stub provider)
- Role override path
- Unknown role → typed error
- Unknown provider → typed error
- `validateProviders()` returns expected structure

**Acceptance criteria:**
- Canary script exits 0 with all assertions passing
- Canary artifact written to `orchestrator/artifacts/canary/llm_dispatcher/`

---

### WS-16-06 Brain Router Unknown Intent Confirmation Step

**Adopted from review R-3.4 (Risk R-6: Router Misclassification)**

**Deliverable:** Updated `src/vnext/brain_router_policy.js`

**Requirement:**
Add a new policy rule (P-06) to the policy override layer:

- Trigger: `intent === 'unknown'` AND input contains execution-oriented keywords (build, implement, create, fix, deploy, develop, 开发, 构建, 实现, 修复, 部署 及类似词)
- Action: emit `decision: 'clarification_required'`, `clarification_prompt` set to:
  > "这看起来像是一个开发任务。是否要启动 Coding Team 工作流？回复 /yes 确认，或描述更多细节。"
- Logging: `[brain_router_policy] P-06 triggered: unknown intent with execution cues → clarification_required`

User `/yes` reply must re-route through Brain Router with a forced `/coder` prefix applied before classification.

**Non-scope:** This rule only applies when intent is `unknown`. All existing intents (chat/coding/quant/docs/ops) are unaffected.

**Acceptance criteria:**
- Unit test: input with execution keywords + `analyzerResult=undefined` → `clarification_required` decision
- Unit test: input without execution keywords + unknown intent → `direct_reply` (chat) unchanged
- Unit test: `/coder` prefix still takes priority over P-06
- All 32 existing tests still pass

---

## WS-17 Coding Team Execution Chain

**Type:** Type A / Critical Path
**Pipeline node:** Coding Team Workflow — steps 2–5
**Prerequisite:** WS-16-01 through WS-16-04 complete

---

### WS-17-00 Define All Missing Handoff Schemas (BLOCKING prerequisite for WS-17-01 to 17-04)

**Why explicit:** These schemas are the contracts that decouple implementation steps from each other. Without them, WS-17-01 through 17-04 cannot be developed independently or tested in isolation.

**Deliverables:**
- `orchestrator/contracts/coding_team_be_to_fe_handoff.schema.json`
- `orchestrator/contracts/coding_team_impl_to_qa_handoff.schema.json`
- `orchestrator/contracts/coding_team_qa_to_release_handoff.schema.json`

**Required fields for each schema:**

`be_to_fe_handoff`:
- `from_step: "impl_be"`
- `to_step: "impl_fe"`
- `be_changes_path` — path to `impl/be_changes/` directory
- `api_contracts` — list of API endpoints implemented (name, method, path, response shape)
- `shared_types` — list of data types shared between BE and FE
- `scope_constraints` — what BE explicitly did NOT implement

`impl_to_qa_handoff`:
- `from_steps: ["impl_be", "impl_fe"]`
- `to_step: "qa_verify"`
- `be_changes_path`, `fe_changes_path`
- `run_instructions` — how to run the implementation locally
- `known_limitations` — what was not implemented

`qa_to_release_handoff`:
- `from_step: "qa_verify"`
- `to_step: "release_pack"`
- `qa_report_path`
- `overall_status: "pass" | "pass_with_warnings" | "fail"`
- `verified_artifacts` — list of artifact paths that passed QA

**Acceptance criteria:**
- All 3 schemas pass AJV validation
- Valid fixture and invalid fixture exist for each schema
- `validate_registry.js` includes these schemas in its validation pass

---

### WS-17-01 Backend Implementation Step

**Deliverables:**
- `backend.impl.v1` prompt script in `registry.json` (`llm_role: "backend"`)
- Backend workflow step wired into `coding_team_v0` workflow definition
- Output written to `impl/be_changes/` (complete files) and `impl/be_notes.md`

**Input:** `handoff/architect_to_impl.json` (existing schema from M3)

**Output artifacts:**
- `impl/be_changes/` — new or modified files (complete content, not diffs)
- `impl/be_notes.md` — implementation decisions, assumptions, run instructions

**Acceptance criteria:**
- Step can be triggered and produces required artifacts
- Step failure produces typed error and does not advance workflow
- Step validator checks: `impl/be_changes/` directory exists and is non-empty, `impl/be_notes.md` exists

---

### WS-17-02 Frontend Implementation Step

**Deliverables:**
- `frontend.impl.v1` prompt script (`llm_role: "frontend"`)
- FE workflow step wired in
- Output: `impl/fe_changes/` + `impl/fe_notes.md`

**Input:** `handoff/architect_to_impl.json` + `handoff/be_to_fe.json` (from WS-17-00)

**Acceptance criteria:** Same pattern as WS-17-01.

---

### WS-17-03 QA Verify Step

**Adopted from review R-3.3 (Risk R-2 / R-4): two-layer validation**

**Deliverables:**
- `qa.verify.v1` prompt script (`llm_role: "qa"`) — distinct from existing `qa.test_plan.v1`
- QA verify step wired in
- Output: `verify/qa_report.json`

**`qa_report.json` required fields:**
- `overall_status`: `"pass"` | `"pass_with_warnings"` | `"fail"`
- `checks`: array of `{ check_id, layer, description, status: "pass"|"fail"|"warning", detail }`
- `verified_artifacts`: list of artifact paths reviewed

**Two validation layers (both must run):**

Layer 1 — Deterministic Checks (schema / artifact existence):
- `be_changes/` and `fe_changes/` directories exist and are non-empty
- `impl/be_notes.md` and `impl/fe_notes.md` exist
- Run instructions are present
- All upstream handoff schemas validated (AJV)

Layer 2 — Semantic Checks (LLM-assisted, via QA role model):
- API contracts declared in BE implementation match what FE implementation references
- Implemented features map to PM acceptance criteria (`plan/acceptance.json`)
- `scope_constraints` from Architect handoff (`handoff/architect_to_impl.json`) are not violated in the implementation

**Input:** `handoff/impl_to_qa.json` (from WS-17-00)

**Acceptance criteria:**
- `qa_report.json` passes schema validation with both `layer` types represented
- Workflow does not proceed to Release if `overall_status: "fail"`
- Layer 1 failure (missing artifact) → `overall_status: "fail"` immediately, Layer 2 not run
- Layer 2 semantic failure (API mismatch) → `overall_status: "fail"` with per-check detail

---

### WS-17-04 Release Pack Step

**Deliverables:**
- `release.pack.v1` prompt script (`llm_role: "release"`)
- Release step wired in
- Output: `release/release_notes.md` + `release/artifact_manifest.json`

**`artifact_manifest.json` required fields:**
- `run_id`
- `workflow_id`
- `completed_at`
- `artifacts`: array of `{ path, type, size_bytes }`

**Input:** `handoff/qa_to_release.json` (from WS-17-00)

**Acceptance criteria:**
- Both output files exist and are non-empty
- `artifact_manifest.json` passes schema validation

---

### WS-17-05 Coding Team End-to-End Canary

**Deliverable:** `orchestrator/scripts/canary_coding_team_e2e.js`

**Prerequisite:** WS-17-00 through WS-17-04 complete.

**Scope:** Full PM → Arch → BE → FE → QA → Release pipeline using stub LLM responses. No real LLM required.

**Acceptance criteria:**
- All 6 steps complete successfully with stub responses
- All required artifacts are produced and present at expected paths
- If any step produces missing artifacts, subsequent steps do not run
- Workflow reaches `succeeded` terminal state
- Failure injection test: inject artifact failure at step 3 (BE) → workflow reaches `failed` state, steps 4–6 do not execute, steps 1–2 artifacts are preserved

**Canary artifact:** `orchestrator/artifacts/canary/coding_team_e2e/`

---

## WS-18 Memory Layer Closure

**Type:** Type B / Enhancement
**Prerequisite:** None (can run in parallel with WS-16 and WS-17)

---

### WS-18-00 M3 Debt Check: WS-15-03 Actual State

**Type:** Investigation — no code output

**Task:** Read `src/domain/workflow_step_builder.js` (or wherever arch_design step is built) and verify whether project context from `memory_reader.js` is actually injected into the Architect prompt, or if it is a wired-but-empty stub.

**Output:** Written finding in `docs/03_feature_development/progress_reports/progress_20260308_ws18_debt_check.md`

**Outcome A (real injection confirmed):** WS-18-01 is classified as Type B enhancement.
**Outcome B (stub only):** WS-18-01 is classified as Type A M3 debt and must complete before M4 closes.

---

### WS-18-01 Memory Injection Verification or Completion

**Classification:** Determined by WS-18-00 outcome.

**Acceptance criteria (regardless of classification):**
- With `artifacts/memory/{project_id}/` present: Architect prompt contains prior ADR summaries formatted as read-only context block
- Without memory files: workflow runs normally, no error thrown
- Integration test covers both paths

---

### WS-18-02 Post-Workflow ADR Write-back

**Deliverable:** Updated `src/domain/memory_writer.js`

**Requirements:**
- Called after workflow reaches terminal `succeeded` state
- Writes `task_history_entry` to `artifacts/memory/{project_id}/task_history.json` (append, not overwrite)
- Extracts any `plan/adr/*.md` files from the run artifact root and copies them to `artifacts/memory/{project_id}/adrs/`
- Write failure must not change workflow terminal status (write is advisory, not transactional)

**Acceptance criteria:**
- After a successful workflow run: `task_history.json` contains a new entry, ADR files are copied
- If memory directory does not exist: it is created
- If write fails: error is logged, workflow status remains `succeeded`

---

### WS-18-03 Memory Layer Canary

**Deliverable:** `orchestrator/scripts/canary_memory_layer.js`

**Coverage:**
- Memory reader returns project context when files exist
- Memory reader returns null/[] gracefully when files do not exist
- Memory writer creates expected files after workflow succeeded
- Architect prompt includes context block when memory is present

---

## 6. Suggested Execution Order

```
WS-16-01  Provider + Role Policy config files (no code, 0.5 day)
WS-17-00  Handoff schemas × 3 (no code, can run parallel with WS-16-01)
WS-18-00  M3 debt check (can run parallel, 0.5 day)
      ↓
WS-16-02  LLM Dispatcher implementation
WS-16-03  Registry cleanup (parallel with WS-16-02)
      ↓
WS-16-04  Wire execution calls to Dispatcher
WS-16-05  Dispatcher Canary
      ↓
WS-17-01  Backend step       WS-18-01  Memory injection
WS-17-02  Frontend step      WS-18-02  ADR write-back
WS-17-03  QA Verify step
WS-17-04  Release Pack step
      ↓
WS-17-05  E2E Canary
WS-18-03  Memory Canary
```

---

## 7. Definition of Done for Milestone 4

Milestone 4 is complete when:

- `orchestrator/configs/llm_providers.json` and `llm_role_policy.json` exist and pass schema validation
- `src/vnext/llm_dispatcher.js` exists, ≤ 250 lines (revised for retry + fallback logic), all unit tests pass
- No `model` field in `prompt_script_registry.json`; all entries have `llm_role`
- All PM/Architect/QA/BE/FE/Release execution calls route through LLM Dispatcher
- All 3 new handoff schemas (`be_to_fe`, `impl_to_qa`, `qa_to_release`) exist and have fixture tests
- Backend, Frontend, QA, Release prompt scripts registered and workflow steps wired
- E2E canary (`canary_coding_team_e2e.js`) passes with full 6-step run
- WS-18-00 debt check completed and documented
- Memory injection verified (real, not stub)
- All pre-existing tests (32/32) still pass
- `node --check` passes on all new files

---

## 8. Non-Scope for Milestone 4

- No LLM classification added to Brain Router (future workstream, explicitly deferred)
- No new agent teams (quant expansion, research, ecommerce)
- No UI dashboard
- No vector/semantic memory
- No distributed orchestrator
- No cross-project memory sharing
- No streaming LLM responses
- No model performance benchmarking
- **No structured diff / AST patching** — full file output maintained for M4; deferred to M5 (review R-3.5)
- **No parallel BE + FE execution** — sequential pipeline only in M4; workflow DAG deferred to M5 (review R-3.6)
- **No adaptive model routing** — static role policy only; dynamic complexity-based routing deferred to M7

---

## 9. Risk Acknowledgment

The following risks from the architecture review are acknowledged and their M4 disposition is recorded:

| Risk | Severity | M4 Disposition |
|------|----------|----------------|
| R-1 Context Explosion | High | Partial — secondary_model fallback handles overflow; full context budget tracking in M5 |
| R-2 Agent Drift | Medium | Mitigated — QA semantic checks verify scope_constraints compliance |
| R-3 Model Non-Determinism | Medium | Ongoing — structured prompts + validators reduce variance; not fully solvable |
| R-4 Pipeline Garbage Propagation | High | Mitigated — handoff schema gates at every transition + QA two-layer validation |
| R-5 Local LLM Availability | High | Mitigated — secondary_model fallback + retry + startup health check (WS-16-01/02) |
| R-6 Router Misclassification | Medium | Mitigated — unknown intent confirmation step (WS-16-06) |
| R-7 Observability Gaps | Medium | Mitigated — LLM Dispatcher structured logs + existing workflow event logs |

Full risk register: `docs/90_archive/260308/OpenClaw_Architecture_Review_and_Risk_Register.md`
