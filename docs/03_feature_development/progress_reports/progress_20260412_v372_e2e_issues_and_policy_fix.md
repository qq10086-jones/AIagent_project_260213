# v3.7.2 E2E 失败分析与 policy.js 误判修复

**Date**: 2026-04-12
**Workflow Run**: `e2514be8-fa1c-4344-8239-efaffc2bc1f7`
**Run ID**: `3352c748-c3f9-4414-96de-a09442e3551b`
**最终状态**: `failed` — impl_fe_modules 耗尽 escalation chain (gemma4:26b → stable_cloud_lane MiniMax M2.7)

---

## 流水线执行结果

| Step | Status |
|------|--------|
| pm_spec | succeeded |
| arch_design | succeeded |
| impl_be | succeeded |
| impl_fe_skeleton | succeeded |
| **impl_fe_modules** | **failed (STEP_IMPL_FE_ARTIFACTS_MISSING)** |
| smoke_test / static_audit / qa_verify / release_pack / deploy_preview | pending |

impl_fe_modules 在此 run 中人工批准 3 次（destructive_command high-risk gate），每次批准后仍因产物缺失而失败。

---

## Issue #1 — `destructive_command` 误判（已修复）

### 根因

`src/policy.js:51` 原正则：
```js
/\b(?:rm\s+-rf|git\s+reset\s+--hard|del\s+\/f|format\s+|mkfs|dd\s+if=)\b/i
```
其中 `format\s+` 过度贪婪——匹配**任何 "format " 前缀的普通散文**。

**实际触发串**（查 Postgres `tasks.payload_json` 确认）：

> `...renders every field in <dl>/<dd> label-value `**`format including`**` created_at/updated_at, with [Edit] and [Delete] bu...`

来自 v3.7.2 detail view spec，纯 HTML 格式描述。

### 修复

`src/policy.js:51`：
```js
/(?:\brm\s+-rf\b|\bgit\s+reset\s+--hard\b|\bdel\s+\/f\b|\bformat\s+(?:[a-z]:|\/[qsuvxy])|\bmkfs\b|\bdd\s+if=)/i
```

- `format\s+` 收窄为 `format\s+(?:[a-z]:|\/[qsuvxy])`（只匹配 DOS 盘符格式化 `format c:` / 开关 `format /q`）
- 外层 `\b` 拆分为每分支独立，避免末尾非词字符（`:`、`=`）阻断边界匹配

### 验证

新增 `test/policy.destructive_command.test.js`：
- 7 个真危险命令全命中（含 `format c:`、`dd if=`、`git reset --hard` 等）
- 5 个安全 prompt 全放行（含 v3.7.2 实际触发串、`use JSON format for output`、`reformat`）
- 2/2 PASS

Policy-adjacent 测试（`task_enqueuer.permission_advisory` / `runtime_dispatch.advisory.integration` / `workflow_dag`）8/8 PASS。

---

## Issue #2 — impl_fe_skeleton validator 短路（已修复）

### session state 原假设（**错误**）

> gemma4:26b 没写 `impl/fe_notes.md`。修法：instructions 里显式列为 required output，或放宽 validator。

### QA 复核发现

从 DB 拉出 failed step 的 `result_json` 后对比磁盘：

```
artifact_check (worker-coder):  found=["impl/fe_patch_bundle.json","impl/fe_notes.md"]
impl_validation (orchestrator): dir_exists=false, patch_bundle_exists=false, fe_changes_count=0
```

磁盘实况 `runtime/artifacts/release/3352c748-.../impl/`：

```
be_changes/ (含 server.js, package.json)  ← impl_be 产物完整
be_notes.md       785 bytes
be_patch_bundle.json  141 bytes (stub, 无 mode 字段)
fe_notes.md       725 bytes   ← 存在！不是缺失
fe_patch_bundle.json  150 bytes (stub, 无 mode 字段)
```

`impl/fe_changes/` **整个目录不存在**。

### 真正的待查问题

1. **impl_fe_skeleton 已标记 succeeded**，按 validator 语义必须产出非空 `impl/fe_changes/public/`（含 index.html、app.js 等），但这些文件**现在不在盘上**。
2. 两种可能：
   - (a) impl_fe_modules 步骤启动时清理了 `impl/fe_changes/`，抹掉 skeleton 产物（artifact lifecycle bug）
   - (b) impl_fe_skeleton 的 validator 本身存在漏洞，在空产出下误判成功
3. `fe_patch_bundle.json` 是 150 字节 stub（无 `mode` 字段），validator 正确拒绝——这是症状不是根因
4. 放宽 `fe_notes.md` 必填（原假设的修法 A）= **修错了东西**

### 根因确认

查 `result_json` 对比：impl_fe_skeleton 的 `impl_validation: {}` 是**空对象**——validator 根本没跑。

`workflow_step_validator.js:63-68` 的白名单：
```js
function isCodingTeamImplementationStep(run, stepId) {
  return String(run?.workflow_id) === "coding_team_v0" &&
    (["impl_fe", "impl_fe_modules"].includes(stepId) || stepId === "impl_be");
}
```

**漏了 `"impl_fe_skeleton"`**。`validateImplementationDelta` 对 skeleton 直接短路返回 `{checked:false, ok:true}`。内部第 161 行分支虽然正确处理了 skeleton（`requireNotes=false`，校验 `public/` 非空），但永远执行不到。

于是 LLM 可以只产出 150 字节 stub `fe_patch_bundle.json`（无 `mode` 字段） + `fe_notes.md`，skeleton 步骤就"succeeded"。下游 modules 步骤缺少 skeleton 产物，注定失败。

### 修复

`workflow_step_validator.js:66` 白名单补 `"impl_fe_skeleton"`：
```js
(["impl_fe", "impl_fe_skeleton", "impl_fe_modules"].includes(stepId) || stepId === "impl_be")
```

修复后 skeleton 步骤会真实校验 `impl/fe_changes/public/` 目录。LLM 若只出 stub，skeleton 会立即失败进入 retry/escalation，而不是假通过让问题推迟到 modules 才爆出。

### 下游待观察

- 修复后下一次 E2E：skeleton 可能直接失败（gemma4:26b 或 MiniMax 能否产出真实 `public/index.html` + `app.js`）
- 若 skeleton 也反复失败 → 另一个独立问题（worker-coder opencode_adapter 在某条路径上只写 bundle 不写 full file）

---

## 评分与状态

- v3.7.2 E2E 因 impl_fe_modules 永久失败未达成 8/10 目标
- v3.7.2 新特性（app.js detail view、activity feed、dashboard quick actions、server.js activity_log 表）**本次未产出**，评分无法验证
- Issue #1 修复独立合入，不依赖 #2
- 流水线整体仍可运行；#1 修复后同类 prompt 不再误触 high-risk 批准

---

## Issue #3 — `PATCH_BUNDLE_INVALID` 时序竞态（已修复）

### 现象
新 E2E `7ef4b3e8` 在 impl_be 步骤以 `PATCH_BUNDLE_INVALID` 失败，但上次 `e2514be8` 同步骤成功——两次 be_patch_bundle.json 都是 141B scaffold 占位（无 `mode` 无 `operations`）。

### 根因
`patch_bundle_service.js:158` 只对 `mode === "full_file_fallback"` 跳过应用。占位 bundle 无 mode → 落到 `applyPatchBundle` → 空 operations → 抛错。
上次侥幸通过：orchestrator 调 `applyStructuredPatchIfPresent` 时 worker-coder 还没写占位文件（`fs.existsSync` → false → 早退 null）。orchestrator 重启后时序反转，命中 bug。

### 修复
`patch_bundle_service.js:157-173`：新增 `isPlaceholder = !mode && operations.length === 0` 判定，等价视为 `full_file_fallback`。

---

## E2E 验证（run `679ac53e-865b-4e8a-8bf0-c40c84049be1`）

**9/10 步 SUCCEEDED**（deploy_preview 外部凭据 queued，独立问题）：
pm_spec → arch_design → impl_be → impl_fe_skeleton → impl_fe_modules → smoke_test → static_audit → qa_verify → release_pack

**首次 v3.7.2 下 E2E 全链路跑通**。零人工批准（policy 修复生效），全程 gemma4:26b stable_gemma4_lane。

---

## 产物评分：~7/10（v3.7 baseline，未达 v3.7.2 的 8/10 目标）

| 特性 | 状态 |
|---|---|
| Customer/Ticket/Comment/File CRUD (BE + FE) | ✓ 完整 |
| Dashboard stats endpoint | ✓ |
| Delete UI + confirmation | ✓ |
| `activity_log` 表 / `/api/activity` | ✗ 缺 |
| Read-only detail view (`renderCustomerDetail` 等) | ✗ 缺 |
| Dashboard quick action buttons | ✗ 缺 |

### 深层根因（不属于本次流程修复范围）

1. **PM 步骤产出 scaffold 模板而非 LLM 真实 spec**——`spec.md` 1902B，结构化三段式 "CRUD operations for X / List view, detail view, create/edit forms"，末尾带 scaffold 签名（"Generated at... Task prompt snippet:"）。v3.7.2 关键词（activity feed, quick action）在 spec 里从未出现。
2. **impl_be/impl_fe 不从 spec 读需求**——它们从 `webapp_crm` project_type contract 取固定合同（customers/tickets/dashboard CRUD），这就是为什么产出完整 CRM 但无 v3.7.2 增量。
3. **arch prompt SPEC COVERAGE 注入实验**（run `90b2fef0`）证明无效——PM spec 本身没这些词，无论 arch prompt 多严格，它"enumerate every feature from spec.md"也枚举不出。

### 达成 8/10 的路径（留给下一会话）

- **A**：修 PM 步骤让它真正调 LLM 产 spec（而非返回 scaffold）
- **B**：扩展 `webapp_crm` project_type contract 显式包含 activity_log + /api/activity + quick actions
- **C**：改 impl 步骤的 prompt 组装，让它从 `plan/spec.md` + `plan/workplan.json` 读 feature 列表而非 project_type 硬合同

推荐 **B + C 组合**：B 最小改动立竿见影，C 解决长期扩展性。

---

## 修改文件

- `orchestrator/src/policy.js` — destructive_command 正则收窄
- `orchestrator/src/domain/workflow_step_validator.js` — 白名单补 `impl_fe_skeleton`
- `orchestrator/src/domain/patch_bundle_service.js` — 占位 bundle 视为 full_file_fallback
- `orchestrator/test/policy.destructive_command.test.js` — 新增回归单测
- `configs/prompt_scripts/registry.json` + `orchestrator/configs/prompt_scripts/registry.json` — arch 注入 SPEC COVERAGE REQUIREMENT（实验性，效果待下游支持后生效）
