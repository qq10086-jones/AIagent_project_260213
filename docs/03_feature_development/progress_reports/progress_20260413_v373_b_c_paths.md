# v3.7.3 B+C 路径实施 + E2E 验证 (inconclusive)

**Date**: 2026-04-13
**Workflow Run**: `f7f03662-d2c6-4f7d-a4f5-e240e786950b`
**Run ID**: `342ada6c-08f8-4c6d-b22e-92c9620979d1`
**Status**: `failed` at `impl_fe_skeleton` — OpenCode command timed out

---

## 这两天做了什么（2026-04-12 → 2026-04-13）

### v3.7 Static Audit Gate（确定性安全/契约门禁）
- 4 个 scanner 接入：xss_scanner / class_injection / delete_semantics / be_contract_checker
- 流水线新增 `static_audit` step（smoke_test → static_audit → qa_verify）
- 默认 `static_audit_mode: dry_run`，blocking 模式保留
- 效果：critical XSS 5→0，class injection 6→3

### v3.7.1 Codex Review 修复
- P0/P1 findings 合入，static_audit 硬化

### v3.7.2 Design Rules 补丁
- 目标：补齐 v3.7.2 三件套 — activity feed / read-only detail view / dashboard quick actions
- prompt 层注入 detail view 和 quick actions 描述（arch 步骤实验）

### v3.7.2 E2E 三个 bug 修复（`e2514be8` run）
1. **policy.js:51 `destructive_command` 正则误判** — `format\s+` 过度贪婪，匹配普通散文 `"label-value format including..."`。收窄为只匹 `format c:` / `format /q`。2/2 回归单测。
2. **workflow_step_validator.js:66 白名单漏 `impl_fe_skeleton`** — skeleton 校验短路，150B stub bundle 也能 "succeeded"。白名单补齐。
3. **patch_bundle_service.js:157-173 占位 bundle 时序竞态** — 无 `mode` 无 `operations` 的占位落到 `applyPatchBundle` 抛错。加 `isPlaceholder` 判定等价视为 `full_file_fallback`。

**验证**：E2E `679ac53e` 9/10 步 SUCCEEDED，零人工批准，全程 gemma4:26b。评分 ~7/10。

### v3.7.3 B+C 路径（本次新增）

**B — webapp_crm 合同扩展**
- `worker-coder/artifact_scaffold.js` BE template 加 `activityLog` store + `recordActivity` helper + `DELETE /api/customers/:id` + `GET /api/activity` + `GET /api/dashboard/stats`；所有 POST/PUT/DELETE 自动写 activity。
- FE template 加 quick actions 行（Add Customer / Refresh / View Activity）+ dashboard stats `<dl>` + activity feed 面板 + delete 按钮 + `renderCustomerDetail`（从 `renderDetail` 重命名）+ `loadActivity` / `loadDashboardStats` / `deleteSelectedCustomer`。
- `orchestrator/src/domain/workflow_state.js` `buildStepPrompt` — `projectType === "webapp_crm"` 时给 arch_design / impl_be / impl_fe_modules 三步注入 **WEBAPP_CRM MANDATORY** 指令块（activity_log 表、/api/activity、dashboard quick actions、read-only detail view 全部从 conditional 升级为 mandatory）。
- `worker-coder/tests/artifact_scaffold.test.js` — DELETE assertion 从 `doesNotMatch` 翻 `match`，新增 `/api/activity` 断言。

**C — impl 步骤读 plan/spec.md**
- `orchestrator/src/domain/workflow_step_builder.js:1008` — impl_be / impl_fe_skeleton / impl_fe_modules 在注入 arch handoff 前先注入 `plan/spec.md` 原文块（6K 字符截断），标为 "authoritative feature source"。
- 两份 `registry.json`（`configs/prompt_scripts/` + `orchestrator/configs/prompt_scripts/`）— frontend.impl.v1 和 v2 的 system_prompt 显式要求以 spec.md 为特性权威，交叉核对 workplan.json fe_tasks。

**测试**：orchestrator 306/313 PASS（基线 304/313 = 减少 2 个 pre-existing 失败）；worker-coder artifact_scaffold PASS。

---

## 本次 E2E 结果（inconclusive）

| Step | Status |
|---|---|
| pm_spec | succeeded |
| arch_design | succeeded |
| impl_be | succeeded |
| **impl_fe_skeleton** | **failed** (OpenCode command timed out) |
| 剩余 6 步 | pending |

### 产物检查

**BE (`impl/be_changes/server.js`, 57 行)**
- ✓ Customer/Ticket/Comment/File CRUD（SQLite，not in-memory）
- ✓ `/api/dashboard/stats`（total_customers / total_tickets / tickets_by_status）
- ✗ **无 `/api/activity`**
- ✗ **无 `activity_log` 表**
- ✗ **POST/PUT/DELETE 未写 activity**
- ✗ DELETE 未做 404 检查（v3.7 contract 盲区仍在）

**FE**：未产出（skeleton timed out）

### 根因分析

**B3 注入未生效** — impl_be 的 payload.task_prompt 里没有 "WEBAPP_CRM MANDATORY" 字符串。原因：orchestrator 服务进程在我改 `workflow_state.js` **之前**就已启动，跑这次 E2E 时加载的是旧代码。需要重启服务才能让 B3 生效。

**C 注入已生效** — impl_be prompt 含 "[PM Spec — plan/spec.md ...]" 块，spec.md 原文注入成功。但由于 PM 步骤本身返回 scaffold 模板（`spec.md` 仅 45 行，goal 被截断在 "newest-fi" — 未完整），spec 里没出现 activity_log / quick actions / detail view 关键词，C 无法弥补源头缺失。

**FE skeleton 超时** — 非模型问题，gemma4:26b 在本地 24GB GPU 上响应 OK（BE 成功）。timeout 推测与 OpenCode adapter 交互或 skeleton prompt 长度有关。待下次 run 观察。

### 模型核验（重要）

用户报告 VRAM 100%。从 tasks 表 payload 确认：
- `model`: **`ollama/gemma4:26b`**
- `execution_lane`: **`stable_gemma4_lane`**
- 全程未升级到 stable_cloud_lane，无 MiniMax 介入

**结论**：没有误用重型模型。gemma4:26b 在 24GB GPU 上跑满 VRAM 是正常水线，非异常。

---

## 遗留问题

### 高优（阻塞评分提升）
1. **PM 步骤返回 scaffold 而非真 LLM spec**（v3.6→v3.7.2→v3.7.3 持续未解决）
   - 证据：本次 spec.md 45 行，goal 截断，模板三段式 "CRUD operations for X / List view, detail view, create/edit forms"
   - 影响：下游 arch/impl 步骤读到的 spec 没有 v3.7.2 关键词，C 路径失效
2. **B3 需重启服务验证** — 下次 E2E 前必须重启 orchestrator 容器
3. **impl_fe_skeleton OpenCode timeout** — 本次第一次出现，需观察是否可复现

### 中优
4. **pre-existing 7 个测试失败**（非 B+C 引入）：registry validator / workflow 集成测试 3 个 / arch interface contract regex / context packet / smoke_test run_acceptance_test.mjs regex
5. **deploy_preview 外部凭据 queued** — 独立问题
6. **BE DELETE 未返 404** — static_audit delete_semantics scanner 能查出，但当前 dry_run 不阻塞

---

## 接下来做什么

### 立即（解锁 B+C 验证）
1. 重启 orchestrator 容器让 B3 生效（`docker compose restart orchestrator`）
2. 重跑 E2E，确认 `WEBAPP_CRM MANDATORY` 在 impl_be prompt 中，且产物出现 activity_log + /api/activity

### 本阶段（提升评分到 8/10）
3. **修 PM 步骤真调 LLM**（根因层修复）— 替换 scaffold 返回，让 goal 完整传入 LLM 产 spec。这是 v3.6 起持续性的阻塞点。
4. **static_audit dry_run → blocking 切换** + retry 反馈回路 — 让 DELETE 缺 404 / POST 缺 validation 的 finding 驱动 impl_be 重试。

### 后续
5. QA 步骤深度语义检查（**全本地 gemma4:26b**，按用户 2026-04-11 决定不切 MiniMax）
6. pre-existing 7 测试失败分诊

---

## 修改文件清单（本次）

- `orchestrator/src/domain/workflow_state.js` — webapp_crm MANDATORY 注入
- `orchestrator/src/domain/workflow_step_builder.js` — impl 步骤 spec.md 注入
- `configs/prompt_scripts/registry.json` — frontend.impl.v1/v2 spec.md 指令
- `orchestrator/configs/prompt_scripts/registry.json` — 同上
- `worker-coder/artifact_scaffold.js` — CRM BE/FE 模板扩展
- `worker-coder/tests/artifact_scaffold.test.js` — 断言更新
- `memory/feedback_no_minimax.md` + `MEMORY.md` — 记录 MiniMax 不续费决定
