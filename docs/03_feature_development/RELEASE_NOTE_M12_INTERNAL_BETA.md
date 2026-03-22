# OpenClaw Nexus — M12 Internal Beta Release Note

**Release Date**: 2026-03-22
**Release Type**: Internal Beta
**Version**: M12
**Status**: APPROVED FOR INTERNAL BETA

---

## 1. 发布范围

本次 Internal Beta 开放 Coding Track 端到端工作流，面向内部测试用户。Quant Track 进入 Beta-pending 状态（容器验证通过，真实 API 联调待执行）。

---

## 2. 验证结果摘要

| Gate | 规格 | 结果 |
|------|------|------|
| Gate B Real E2E | Runs 15, 并发 3, 全链路 | **PASS 17/17 (100%)** |
| Gate A Route Queue | dispatch 验证 + 设计缺陷关闭 | **CLOSED（见注）** |
| Final Gate Smoke | 认证修复验证，全步骤链路 | **PASS（链路验证通过）** |
| orchestrator 单测 | 150 用例 | **150/150 PASS** |
| worker-coder 单测 | 19 用例 | **19/19 PASS** |
| worker-quant 离线测试 | `_merge_recent_news` 10 用例 | **10/10 PASS** |

> **Gate A 说明**：Gate A 经 QA 评审确认存在测试设计缺陷（路由压测框架实际触发完整 E2E workflow，`min-workflow-success-rate 1.0` 对 LLM 工作流不合理）。路由/入队已由现有数据验证（dispatch_ok=100%，p95 < 15s），workflow 质量由 Gate B 覆盖。正式关闭，无需重跑。治理文档：`docs/governance/m12_gate_a_closure_note.md`

---

## 3. M12 关键变更

### 3.1 LLM Provider 迁移（Breaking Fix）

- **从**：`dashscope/qwen`（`E_AUTH_FAILED` 频繁）
- **到**：`minimax-coding-plan/MiniMax-M2.5`（stable_cloud_lane）
- **opencode provider 命名修复**：`opencode.json` 中 provider key 从 `"minimax"` 改为 `"minimax-coding-plan"`，与 `runtime_defaults.json` 中 `execution_lane_default` 引用对齐。此命名不一致是 Gate A 期间所有 `OpenCode authentication failed` 的根因，修复后 Final Gate Smoke 全链路无认证失败。

### 3.2 worker-quant OpenBB 集成

- 容器构建验证 PASS（`numpy-1.26.4` / `pandas-2.3.3` / `openbb-4.7.1` 无冲突）
- `_merge_recent_news` P0 Bug 修复：补充外层 `try/except`，防止内层异常逃逸导致整个函数崩溃
- 10/10 单元测试覆盖（含 OpenBB 超时、降级、去重、trusted publisher 排序等场景）

### 3.3 架构稳定性

- DAG 调度并发修复：`arch_design` 步骤在 2 并发下的提示/契约对齐问题已修复
- Context budget 服务正常，步骤间 handoff 契约验证通过

---

## 4. 运营约束

### 4.1 并发限制（P1，已知）

**生产环境最大并发工作流数：3**

- 并发 3 经 Gate B 17/17 验证，100% 通过
- 并发 > 3 时 MiniMax 请求排队，工作流总时长超过 30 分钟，存在超时风险
- 运营期间 Discord 网关应限流，确保同时进行中的 coding 工作流不超过 3 个

### 4.2 工作流完成时间

- 单个 coding workflow（pm_spec → deploy_preview）端到端约 **45-70 分钟**
- `deploy_preview` 为最后一步，使用本地 python http.server 提供静态预览，正常完成
- 告知测试用户：提交任务后等待时间较长，属正常现象

### 4.3 opencode.json 不得提交 git（重要）

- `opencode.json` 包含 MiniMax API Key 硬编码（opencode CLI 不支持 `${VAR}` 环境变量插值）
- **此文件必须加入 `.gitignore`，发布前必须执行**，否则 key 将泄漏至 git 历史
- Key 同时存储于 `infra/.env`（已在 `.gitignore` 保护范围内）

### 4.4 OpenBB 真实 API（P1，待验证）

- 降级逻辑经测试覆盖，健壮性已验证
- `OPENBB_FMP_KEY` 注入后的真实 API 行为仍需生产环境验证
- Beta 期间 Quant Track 建议使用降级模式（无 FMP key），待验证完成后开放完整功能

### 4.5 熔断器

- 当前状态：`activated: false`，`force_sequential: false`
- 阈值：partial_failure_rate > 25% 或 rollback_trigger_events ≥ 3 触发
- 无自动恢复，触发后需运营人员调用 `resetCircuitBreaker` 手动重置

---

## 5. 基础设施配置

| 项目 | 值 |
|------|----|
| Coder Provider | MiniMax-M2.5 via minimaxi.com (stable_cloud_lane) |
| Worker Timeout | 1800s（任务级），900s（wall clock） |
| Watchdog 轮询 | 每 30s |
| 队列超时 | 6 小时（queued_timeout_sec: 21600） |
| Redis 流 | stream:task:coding → worker-coder |
| 预览服务 | 本地 python http.server（static mode） |

---

## 6. 上线前必做清单

- [ ] **`opencode.json` 加入 `.gitignore`**（安全，阻塞项）
- [ ] 确认 `MINIMAX_API_KEY` 已写入 `infra/.env`
- [ ] 确认 `infra/.env` 的 BOM 字符问题不影响 docker-compose 加载（当前无影响，建议修复）
- [ ] 通知内部测试用户：单次任务等待时间 45-70 分钟

---

## 7. 上线后监控重点

1. **熔断器状态**：`node orchestrator/scripts/exposure_state_query.js`
2. **并发队列深度**：Redis 中 `stream:task:coding` pending 条目数，超过 3 时告警
3. **工作流超时率**：`workflow.status = timeout` 超过 10% 触发人工介入
4. **worker-quant**：监控 `_fetch_news_from_openbb` 异常日志，确认降级正常触发

---

## 8. 回滚预案

```bash
# 紧急切顺序执行（关闭并行）
# 编辑 configs/production_parallel_rollout.json
# 设置 force_sequential: true

# 熔断器重置（需要先解决根因）
node orchestrator/scripts/reset_circuit_breaker.js
```

详细步骤见 `docs/runbooks/m6_parallel_rollback_runbook.md`。

---

## 9. Beta 后迭代计划

- [ ] `opencode.json` env var 插值支持调研（避免 key 硬编码）
- [ ] 工作流总时长优化（目标 < 30 分钟，需 LLM 响应提速或步骤并行化扩展）
- [ ] OpenBB 真实 API 联调验证
- [ ] Golden Set 扩充至 ≥200 条（当前 100 条）
- [ ] TDnet/J-Quants 实网解析验证
- [ ] `infra/.env` BOM 字符修复

---

*Generated: 2026-03-22 | QA Sign-off: Internal Beta APPROVED*
