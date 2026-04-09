# Nexus Coder v3.3 — 任务清单

> **对应设计文档**: `nexus_coder_v3.3_project_planner.md`
> **日期**: 2026-04-09
> **估算基准**: v3.2 Phase 0-3 实施节奏

---

## Phase A: Planner Core (MVP)

> 目标: 能拆解、能验证、能人工确认
> 前置: 无（可立即开始）

| 任务 ID | 标题 | 文件 | 依赖 | 验收标准 |
|---------|------|------|------|---------|
| P-A1 | Schema 验证 + DAG 无环校验 | `orchestrator/src/vnext/project_plan_contract.js` | 无 | 1) 合法 plan 通过验证 2) 缺少 required field 拒绝 3) 有环依赖图拒绝 (C-04) 4) run_key 唯一性 (C-03) 5) task_class 从 `worker_coding_task_classes.json` 动态读取并校验 (C-02) 6) target_paths 跨 run 不重叠 (C-06) 7) shared_context.artifacts 限标准路径 (C-09) |
| P-A2 | LLM 拆解核心 | `orchestrator/src/vnext/project_planner.js` | P-A1 | 1) 复用 `callQwenChat()` 调用 LLM 2) prompt 模板运行时注入 task_classes / project_type / workspace 3) JSON 提取 (```json``` fence + fallback) 4) schema 验证失败时最多重试 1 次 (prompt 追加错误反馈) 5) LLM 超时/失败 → single-run fallback 6) 保留原始 LLM 响应用于调试 |
| P-A3 | Feature Flag 配置 | `configs/runtime/runtime_defaults.json` | 无 | 新增 5 个 flags: `project_planner_enabled`, `_max_runs`, `_max_parallel_runs`, `_failure_policy`, `_confirm_mode` |
| P-A4 | Planner 单元测试 | `orchestrator/test/project_plan_contract.test.js` + `orchestrator/test/project_planner.test.js` | P-A1, P-A2 | contract ≥8 用例 (每条 C 规则至少 1 个), planner ≥5 用例 (mock LLM, 含中文/英文/fallback/retry 场景) |

**Phase A 交付物**:
- 给定任意产品需求文本 → 输出合法的 `project_plan.json`
- 可在终端手动运行: `node scripts/test_planner.js "做一套客诉管理系统"`

---

## Phase B: Executor Engine

> 目标: 能按 plan 依次启动 workflow runs
> 前置: Phase A 完成

| 任务 ID | 标题 | 文件 | 依赖 | 验收标准 |
|---------|------|------|------|---------|
| P-B1 | 多 Run 编排执行器 | `orchestrator/src/vnext/project_executor.js` | P-A1 | 1) 接受任意合法 project_plan.json 2) topo sort 计算执行波次 (waves) 3) 按 config `max_parallel_runs` 控制并发 4) 每个 run 调用 `startWorkflowRun()` 5) run 完成回调触发下游 6) 状态机: CREATED→VALIDATED→SCHEDULED→CONFIRMED→RUNNING→COMPLETED/PARTIAL_FAILURE→REPORTED |
| P-B2 | workflow_engine 扩展 | `orchestrator/src/workflow_engine.js` | P-B1 | `startWorkflowRun()` 接受可选 `project_context` 参数 (project_id, run_key, upstream_artifacts[]), 透传到 pm_spec 步骤的 payload。无 project_context 时行为不变 |
| P-B3 | 跨 Run 上下文注入 | `project_executor.js` (内) | P-B1, P-B2 | 1) `injectSharedContext()` 按 `shared_context.from_runs` + `.artifacts` 从上游 artifact_dir 提取文件 2) 注入到下游 `context_packet.upstream_artifacts` 3) 文件不存在时 warn log 但不 block (graceful degradation) |
| P-B4 | runtime_dispatch 集成 | `orchestrator/src/vnext/runtime_dispatch.js` | P-A2, P-B1 | 1) orchestrated_workflow + complex + flag → planner 路径 2) runs.length > 1 → executor 3) runs.length == 1 → 退化单 run 4) flag=false → 跳过 planner, 完全不影响现有逻辑 |
| P-B5 | 集成测试 | `orchestrator/test/project_executor.test.js` | P-B1~B4 | 1) 2-run 依赖链: R-01 完成后 R-02 自动启动 2) 跨 run artifact 传递验证 3) 并行 wave: 无依赖 runs 同时启动 4) 失败传播: stop_dependents 策略 5) flag=false 回归 |

**Phase B 交付物**:
- 完整的 project_plan.json → N 个 workflow runs 自动编排
- 跨 run artifact 传递验证通过

---

## Phase C: Governance & Reporting

> 目标: 产出项目级治理文档和汇总报告
> 前置: Phase B 完成

| 任务 ID | 标题 | 文件 | 依赖 | 验收标准 |
|---------|------|------|------|---------|
| P-C1 | 项目汇总报告 | `project_executor.js` (内) | P-B1 | 1) 所有 run 完成后生成 `project_summary.json` 2) 包含: 每个 run 的 status/duration/failure_attribution 3) 包含: 总验收标准达成率 4) 包含: 合并后的 risk_report |
| P-C2 | 人工确认模式 | `project_planner.js` (内) | P-A2 | 1) `decompose()` 返回 plan 后不自动执行 2) 支持 `confirm_mode: "manual"` — 输出 plan 等待用户确认 3) 用户可修改 plan (增删改 runs) 后再提交执行 |
| P-C3 | 断点续跑 | `project_executor.js` (内) | P-B1 | 1) project 中断后重新启动时，跳过已完成的 runs 2) 从最后未完成的 run 继续 3) 持久化 project 状态到文件系统 |
| P-C4 | 回归验证 | 全量测试 | P-C1~C3 | 1) 现有 104 测试全绿 2) 新增测试 ≥15 用例 3) flag=false 时行为完全不变 |

**Phase C 交付物**:
- 完整的项目级治理能力
- 人工确认 + 断点续跑 + 汇总报告

---

## 实施顺序与依赖图

```
Phase A (Planner Core)
  P-A1 ──→ P-A2 ──→ P-A4
  P-A3 (独立)       ↓
                    ↓
Phase B (Executor Engine)
  P-B1 ──→ P-B2 ──→ P-B3
  P-B1 ──→ P-B4
  P-B1~B4 → P-B5
                    ↓
Phase C (Governance)
  P-C1 (独立于 C2/C3)
  P-C2 (独立于 C1/C3)
  P-C3 (独立于 C1/C2)
  P-C1~C3 → P-C4
```

---

## 关键设计决策记录 (ADR)

| ADR | 决策 | 理由 |
|-----|------|------|
| ADR-01 | Planner 使用与 brain 相同的 MiniMax-M2.7 | 无需新增 provider，复用现有 LLM 基础设施 |
| ADR-02 | project_plan.json 走 schema 硬验证 | LLM 输出不可信，必须 contract 校验后才能执行 |
| ADR-03 | 跨 run 上下文通过文件系统传递 (不走 Redis) | artifact 已经是文件，直接读取最简单，无需序列化 |
| ADR-04 | 默认 max_parallel_runs=2 | 与 production_parallel_rollout.json 的 max_concurrent_workflows=3 对齐，留 1 slot 给非 project 任务 |
| ADR-05 | 失败策略 stop_dependents (不是 stop_all) | 并行分支不应因无关 run 的失败而停止 |
| ADR-06 | Phase A 产出 plan 但不自动执行 | 人先看一遍拆解结果，确认合理后再跑，降低 LLM 拆解错误的风险 |
| ADR-07 | 单 run 退化为现有行为 | 如果 LLM 判断不需要拆解（需求够小），直接走单 run，不强制拆 |
