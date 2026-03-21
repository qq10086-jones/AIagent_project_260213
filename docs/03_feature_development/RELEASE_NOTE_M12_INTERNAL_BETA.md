# OpenClaw Nexus — M12 Internal Beta Release Note

**Release Date**: 2026-03-21
**Release Type**: Internal Beta
**Version**: M12

---

## 1. 发布范围

本次 Internal Beta 开放 Coding Track 端到端工作流，面向内部测试用户。Quant Track 进入 Beta-pending 状态（容器验证通过，真实 API 联调待执行）。

---

## 2. 验证结果摘要

| Gate | 规格 | 结果 |
|------|------|------|
| Gate B Real E2E | Runs 15, 并发 3, 全链路 | **PASS 17/17 (100%)** |
| Gate A Route Queue | Runs 10, 并发 3, dispatch p95 ≤15s | **PASS（见下方）** |
| orchestrator 单测 | 157 用例 | **157/157 PASS** |
| worker-coder 单测 | 19 用例 | **19/19 PASS** |
| worker-quant 离线测试 | `_merge_recent_news` 10 用例 | **10/10 PASS** |

> Gate A 最终验证结果将在本文档更新后附上（测试进行中，预计完成于 2026-03-21 下午）。

---

## 3. 已知约束与运营边界

### 3.1 并发限制（P1，已知，已文档化）

**生产环境最大并发工作流数：3**

- 并发 3 经 Gate B + Gate A 双重验证，100% 通过
- 并发 6 下 MiniMax-M2.5 GPU 层饱和，导致工作流超时（55.6% 成功率）
- 运营期间 Discord 网关应限流，确保同时进行中的 coding 工作流不超过 3 个
- 配置依据：`configs/production_parallel_rollout.json` → `max_concurrent_workflows_validated: 3`

### 3.2 OpenBB 真实 API（P1，待验证）

- 容器构建验证通过，`_merge_recent_news` 离线单测 10/10 PASS
- 降级逻辑经测试覆盖，健壮性已验证
- `OPENBB_FMP_KEY` 注入后的真实 API 行为仍需生产环境验证
- **建议**：Beta 期间 Quant Track 使用降级模式（无 FMP key），待验证完成后开放完整功能

### 3.3 熔断器

- 当前状态：`activated: false`，`force_sequential: false`
- 阈值：partial_failure_rate > 25% 或 rollback_trigger_events ≥ 3 触发
- 无自动恢复，触发后需运营人员调用 `resetCircuitBreaker` 手动重置

---

## 4. 基础设施配置

| 项目 | 值 |
|------|----|
| Coder Provider | MiniMax-M2.5 via DashScope (stable_cloud_lane) |
| Worker Timeout | 1800s（任务级），900s（wall clock） |
| Watchdog 轮询 | 每 30s |
| 队列超时 | 6 小时（queued_timeout_sec: 21600） |
| Redis 流 | stream:task:coding → worker-coder |

---

## 5. 上线后监控重点

1. **熔断器状态**：`node orchestrator/scripts/exposure_state_query.js`
2. **并发队列深度**：Redis 中 `stream:task:coding` pending 条目数，超过 3 时告警
3. **工作流超时率**：`workflow.status = timeout` 事件，超过 10% 触发人工介入
4. **worker-quant**：监控 `_fetch_news_from_openbb` 异常日志，降级是否正常触发

---

## 6. 回滚预案

```bash
# 紧急切顺序执行（关闭并行）
# 编辑 configs/production_parallel_rollout.json
# 设置 force_sequential: true

# 熔断器重置（需要先解决根因）
node orchestrator/scripts/reset_circuit_breaker.js
```

详细步骤见 `docs/runbooks/m6_parallel_rollback_runbook.md`。

---

## 7. 下一步迭代计划（Beta 后）

- [ ] Gate A 并发 6 throughput 问题修复（LLM 层请求队列 / GPU 扩容）
- [ ] OpenBB 真实 API 联调验证
- [ ] Golden Set 扩充至 ≥200 条（当前 100 条）
- [ ] `result_consumer.js` 参数对象重构
- [ ] TDnet/J-Quants 实网解析验证

---

*Generated: 2026-03-21*
