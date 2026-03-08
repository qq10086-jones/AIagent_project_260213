# Progress Report — WS-17-05 E2E Canary Closure

## Date
2026-03-08

## Scope
WS-17-05: Coding Team End-to-End Canary（`scripts/canary_coding_team_e2e.js`）

## Status
**DONE** — canary passes both test cases

---

## 完成内容

### WS-17-05 E2E Canary
完整 PM → Arch → BE → FE → QA → Release 六步流水线，stub LLM，in-memory DB pool。

**两个测试用例：**
- `happy_path`：全 6 步成功，workflow=succeeded，6 个 checkpoint 记录，release 包写盘验证通过
- `be_failure`：BE 产出无效 artifacts → workflow=failed，FE/QA/Release 不触发（只调度 3 个 task）

---

## 修复的 Bug（调试过程中发现）

### Bug 1 — PM `acceptance.json` criteria schema 类型错误
- **现象**：PM 终端处理后 arch task 未调度（No task at index 1）
- **原因**：canary 写入 `criteria: [{ id, description }]`（对象数组），但 `coding_team_pm_acceptance.schema.json` 要求字符串数组
- **修复**：改为 `criteria: ["AC-001: User can log in"]`

### Bug 2 — Arch handoff `decisions` schema 类型错误
- **原因**：canary 写入 `decisions: ["postgres"]`（字符串数组），但 `coding_team_arch_handoff.schema.json` 要求对象数组 `{adr_id, title, status}`
- **修复**：改为 `decisions: [{ adr_id: "ADR-001", title: "Use Postgres for auth storage", status: "accepted" }]`

### Bug 3 — BE-to-FE handoff `shared_types` schema 类型错误
- **原因**：canary 写入 `shared_types: ["User"]`（字符串数组），但 schema 要求对象数组 `{name}`
- **修复**：改为 `shared_types: [{ name: "User", description: "{ id, email }" }]`

### Bug 4 — QA report `verified_artifacts` ID 不匹配
- **现象**：artifact pack 校验失败（`ARTIFACT_INVALID:verify/qa_report.json:missing acceptance ids A1`）
- **原因**：`deriveAcceptanceIds`（`artifact_pack_validator.js`）对字符串 criteria 生成 fallback ID `"A1"`，而 canary 写入 `verified_artifacts: ["AC-001"]`
- **修复**：`verified_artifacts` 改为 `["A1"]`（匹配 deriveAcceptanceIds 输出）

### Bug 5 — `release` 角色缺少 tool permission
- **现象**：QA terminal 后 release task 未调度（`TOOL_PERMISSION_DENIED`）
- **原因**：`src/vnext/tool_permission_guard.js` `DEFAULT_MATRIX` 缺少 `release`/`release_agent` 角色
- **修复**：添加 `"release": ["coding.delegate", "document.read", "document.write", "file.read"]` 及 `release_agent`

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `orchestrator/scripts/canary_coding_team_e2e.js` | 新建（WS-17-05 canary） + 4 项 stub artifact 修复 |
| `orchestrator/src/vnext/tool_permission_guard.js` | DEFAULT_MATRIX 添加 release/release_agent 角色 |

---

## 验证证据

```
node scripts/canary_coding_team_e2e.js
# Coding Team E2E Canary (WS-17-05)
- happy_path: all 6 steps succeeded, workflow=succeeded
  checkpoints=6 release_root=...
- be_failure: workflow=failed, be_error=STEP_IMPL_BE_ARTIFACTS_MISSING, tasks_dispatched=3
- report: .../coding_team_e2e_canary.json
exit: 0

node scripts/canary_tool_permission_guard.js     → pass
node scripts/canary_coding_team_workflow_integration.js → pass
node scripts/canary_backend_execution_adapter.js  → pass
node scripts/canary_frontend_execution_adapter.js → pass
node scripts/canary_qa_verifier.js                → pass
node scripts/canary_agent_contract_layer.js       → pass
node scripts/canary_runtime_contract_hardening.js → pass
node --test --experimental-test-isolation=none test/workflow_step_validator_backend.test.js  → pass
node --test --experimental-test-isolation=none test/workflow_step_validator_frontend.test.js → pass
node --test --experimental-test-isolation=none test/artifact_pack_validator.test.js          → pass
```

---

## M4 工作项完成状态

| WS | 状态 |
|----|------|
| WS-16-01 ~ WS-16-06 | **全部 DONE** |
| WS-17-00 ~ WS-17-05 | **全部 DONE** |
| WS-18-00 | **DONE** |
| WS-18-01 ~ WS-18-03 | pending |

---

## 下一步
- WS-18-01：Memory layer strict closure（Type B）
- WS-18-02：ADR write-back
- WS-18-03：Memory layer canary
