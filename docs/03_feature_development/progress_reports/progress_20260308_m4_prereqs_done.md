# OpenClaw Nexus Progress Report
## M4 前置任务完成（WS-16-01 / WS-17-00 / WS-18-00）

- Date: `2026-03-08`
- Phase: `Milestone 4 — LLM Routing Layer + Coding Team Execution Chain`
- Session: 前置无代码任务执行

---

## 执行摘要

本次会话完成了 M4 的三个前置任务，全部为"无代码"任务。
所有新文件通过 JSON 语法验证（13/13 ok）。

---

## WS-18-00 — M3 债务核查：WS-15-03 实际状态

**结论：WS-15-03 是真实注入，不是 stub。**

核查文件：`orchestrator/src/domain/workflow_step_builder.js`，Lines 140–161

实际代码行为：
- 在 `arch_design` 步骤构建 payload 时，调用 `getProjectContext` / `getPriorADRs` / `getTaskHistory`
- 若任一返回非空，则构建 `[Project Memory Context]` 文本块追加到 `payload.task_prompt`
- 无内存文件时，代码正常跳过，无报错

**分类裁决：WS-18-01 定性为 Type B 新功能（非 M3 债务）。**

---

## WS-16-01 — LLM Provider Registry 配置文件

**状态：DONE**

创建文件：

| 文件 | 说明 |
|------|------|
| `orchestrator/contracts/llm_provider_registry.schema.json` | 基础设施配置 JSON Schema |
| `orchestrator/contracts/llm_role_policy.schema.json` | 角色策略 JSON Schema（含 secondary_model + retry_policy 字段） |
| `orchestrator/configs/llm_providers.json` | 两个 provider（cloud_qwen / local_ollama），端点和模型白名单 |
| `orchestrator/configs/llm_role_policy.json` | v1.1.0，6 个角色分配，local 角色均配 secondary_model=qwen2.5-coder:7b，fallback_policy=model_fallback |

**validate_registry.js 未修改说明：**
`validate_registry.js` 专用于 capability registry（工具/工作流注册），混入 LLM 配置验证不符合职责边界。LLM 配置验证改为由 `llm_dispatcher.validateProviders()`（WS-16-02）在运行时启动阶段执行。此为架构设计决策，不是遗漏。

---

## WS-17-00 — 全部 Handoff Schemas 定义

**状态：DONE（阻塞 WS-17-01~04 的前置任务已解除）**

创建文件：

| 文件 | 说明 |
|------|------|
| `orchestrator/contracts/coding_team_be_to_fe_handoff.schema.json` | BE→FE 交接：api_contracts / shared_types / scope_constraints |
| `orchestrator/contracts/coding_team_impl_to_qa_handoff.schema.json` | Impl→QA 交接：be/fe_changes_path / run_instructions / known_limitations |
| `orchestrator/contracts/coding_team_qa_to_release_handoff.schema.json` | QA→Release 交接：overall_status（pass/pass_with_warnings/fail） / verified_artifacts |
| `orchestrator/contracts/fixtures/be_to_fe_handoff_valid.json` | CRM MVP 示例，2 个 API contract |
| `orchestrator/contracts/fixtures/be_to_fe_handoff_invalid.json` | 缺必填字段 + 错误 from_step |
| `orchestrator/contracts/fixtures/impl_to_qa_handoff_valid.json` | 含 run_instructions + known_limitations |
| `orchestrator/contracts/fixtures/impl_to_qa_handoff_invalid.json` | 缺 be_changes_path + 错误 to_step |
| `orchestrator/contracts/fixtures/qa_to_release_handoff_valid.json` | overall_status=pass_with_warnings + warnings 列表 |
| `orchestrator/contracts/fixtures/qa_to_release_handoff_invalid.json` | overall_status 值不在 enum + 缺必填字段 |

---

## 验证结果

```
node -e "JSON.parse each file" → 13/13 ok, 0 failed
```

---

## 关键架构设计记录

### be_to_fe_handoff 中的 api_contracts 结构

采用对象数组（含 method/path/response_shape），而非字符串数组。这让 QA 语义验证（WS-17-03 Layer 2）可以直接对比 FE 实现中的 API 调用路径，而不需要解析自由文本。

### qa_to_release_handoff 中的 overall_status=fail 阻塞

Release 步骤在消费此 handoff 时必须检查 `overall_status`，若为 `"fail"` 则拒绝启动。这是防止 Pipeline Garbage Propagation（Risk R-4）的关键门控。

---

## 未完成（下次会话继续）

| 任务 | 状态 | 备注 |
|------|------|------|
| WS-16-02 `llm_dispatcher.js` | 待开始 | ~250 行代码，需新会话 |
| WS-16-03 registry.json 清理 | 待开始 | 依赖 WS-16-02 完成后方向确定 |
| WS-16-04 Wire execution calls | 待开始 | 依赖 WS-16-02 |
| WS-16-05 Dispatcher canary | 待开始 | 依赖 WS-16-02 |
| WS-16-06 Brain Router P-06 | 待开始 | 独立，可与 WS-16-02 并行 |
| WS-17-01~05 | 待开始 | 依赖 WS-16-01~04 完成 |
| WS-18-01~03 | 待开始 | Type B，可并行 |

---

## Source Of Truth

- 活跃设计文档：`docs/01_design/system/260308/`
- 最新状态：`docs/03_feature_development/PROGRESS_LATEST.md`
