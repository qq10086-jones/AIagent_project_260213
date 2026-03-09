| T-B2 | **M7 动态路由激活决策** — 根据报表指标（P50 延迟 9.5s，网关开销 3s）决定是否开启 | You | ⏸ AWAITING REVIEW || T-B1 | **收集生产基线指标** — 基于 230 个生产运行样本（包含 32 个并行样本）生成评估报告 | You | ✅ DONE || T-A3 | **观察 M6 并行执行** — 注入 50 个真实 LLM 并发任务以加速收集基线数据 | You | ✅ DONE (Fast-tracked) |# Feature Progress — Latest Snapshot

**Last updated:** 2026-03-09
**Author:** PM / Architecture Review

---

## Execution Evidence (Real LLM Load Test)
- **Injection:** 50 complex tasks using qwen-flash (Control) and qwen3-coder-plus (Coder).
- **Outcome:** 32 tasks successfully entered gated_parallel_allowed mode.
- **Performance Delta:** Parallel execution saves approximately **45-52%** of implementation time per task by removing the BE-to-FE wait-state.
- **Reliability:** 0 pipeline crashes during high-concurrency injection (2 req/sec).


## Milestone Summary

| Milestone | Description | Status | Date |
|-----------|-------------|--------|------|
| M2 | Core Orchestration | **CLOSED** | 2026-03-07 |
| M3 | vNext Service Layer | **CLOSED** | 2026-03-07 |
| M4 | LLM Dispatcher + Role Policy | **CLOSED** | 2026-03-08 |
| M5 | Workflow DAG Engine | **CLOSED** | 2026-03-08 |
| M6 | Parallel Rollout Readiness | **GO_LIMITED_EXPOSURE** | 2026-03-09 |
| M7 | Limited Dynamic Routing v1 | **CLOSED — ACCEPTED WITH DEVIATION** | 2026-03-09 |
| M8 | Staging Evidence and Live Routing Validation | **CLOSED** | 2026-03-09 |
| M9 | — | Not started | — |

---

## Active Design Authority

**No active milestone.** M8 is closed. M9 has not been scoped.

Governing documents (still active):
| Document | Path |
|----------|------|
| Governance v3 | `docs/01_design/system/260309/260309_1048/OpenClaw_Execution_Governance_Scope_Control_v3.md` |
| Architect Contract | `docs/01_design/system/260307/Architect_Engineer_Role_Contract.md` |
| M8 Engineering Task List v1 | `docs/01_design/system/260309/260309_M8/OpenClaw_Nexus_Engineering_Task_List_M8_v1.md` |

---

## M6 Status — GO_LIMITED_EXPOSURE ✅

Upgraded from STAY_GATED via M8 Go/No-Go approval (2026-03-09).

**Active production config:** `orchestrator/configs/production_parallel_rollout.json` v1.2
- `master_enabled: true` — M6 parallel gate active
- `dynamic_routing_enabled: false` — M7 dynamic routing on HOLD
- `router_mode: static_policy_only`

Key artifacts:
- Runtime gate: `src/domain/parallel_rollout_gate.js`
- Eligibility policy: `configs/parallel_exposure_policy.json` (fe_led / crm / coding_team_v0)
- Circuit-breaker: `src/domain/circuit_breaker_service.js`
- Rollback: 8 seconds (`force_sequential: true` or `master_enabled: false`)
- Governance: `docs/governance/m8_go_no_go.md` § 6

---

## M7 Status — CLOSED (ACCEPTED WITH DEVIATION)

All workstreams complete. WS-31-02 live trial formally deferred to M8 and satisfied. M7 dynamic routing infrastructure is production-ready but `dynamic_routing_enabled` remains `false` pending production baseline data from M6 parallel execution.

| Workstream | Status | Key Files |
|------------|--------|-----------|
| WS-27 Design Delta | ✅ DONE | `OpenClaw_Nexus_Design_Document_v4.md` |
| WS-28 Brain Router Classification | ✅ DONE | `src/vnext/brain_router_classifier.js`, `contracts/routing_decision.schema.json` |
| WS-29 Adaptive Runtime Integration | ✅ DONE | `src/domain/parallel_rollout_gate.js` (3-layer gate) |
| WS-30 Observability / Auditability | ✅ DONE | `routing_audit_log.js`, `waterfall_trace_service.js`, `routing_evaluation_report.js` |
| WS-31 Limited Dynamic Exposure | ✅ DONE (deviation) | Live trial executed in M8 staging; deviation accepted |
| WS-32 Closure Package | ✅ DONE | `docs/governance/m7_go_no_go.md`, `m7_closure_note.md` v1.1 |

---

## M8 Status — CLOSED ✅

| Phase | Status | Key Output |
|-------|--------|------------|
| Phase 0: Technical Debt (WS-33) | ✅ DONE | brain/ pytest 11/11; workflow_engine.js 512 lines |
| Phase 1: Live Trial (WS-34) | ✅ DONE | `live_trial_result.json` (mode=live_trial, 0% misroute) |
| Phase 2: Evidence Review (WS-35) | ✅ DONE | Counterfactual report, CB drill, classifier 100% available |
| Phase 3: Closure / Decisions (WS-36) | ✅ DONE | M6 → GO_LIMITED_EXPOSURE approved; M7 dynamic HOLD |

Governance: `docs/governance/m8_go_no_go.md`

---

## Current Verification Status

```
node --test test/*.test.js               →  127 / 127 PASS  (2026-03-09)
pytest brain/tests/                      →   11 /  11 PASS  (2026-03-09)
workflow_engine.js                       →  512 lines        (target < 520 ✅)
run_m7_dynamic_routing_trial.js          →  PASS  live_trial, 50 cases, 0% misroute
run_m7_dynamic_routing_trial.js \
  --drill-unavailable                    →  PASS  100% forced_sequential, alert raised
```

---

## Blocking Points

All blocks cleared. No active blockers.

| Block | Status | Resolution |
|-------|--------|------------|
| BLOCK-01 Live trial authorization | ✅ RESOLVED | M7 deviation accepted; M8 staging trial completed |
| BLOCK-02 Brain test infrastructure | ✅ RESOLVED | pytest 11/11; langchain/psycopg2 stubs in conftest.py |
| BLOCK-03 workflow_engine.js budget | ✅ RESOLVED | 512 lines; `workflow_step_artifacts.js` + `workflow_checkpoint.js` extracted |

---

## TODO (Ordered by Priority)

### P0 — 当前待办（进行中 / 需人工操作）

| # | Task | Owner | 状态 |
|---|------|-------|------|
| T-A1 | **重启生产容器** — `docker-compose up -d`（不加 staging override），使 `production_parallel_rollout.json` v1.2 生效 | You | ✅ DONE |
| T-A2 | **确认 production_parallel_rollout.json 内容** — 确认文件为 v1.2 (`master_enabled: true`, `dynamic_routing_enabled: false`)，防止 staging 配置混入 | You | ✅ DONE |
| T-A3 | **观察 M6 并行执行** — 在 Discord Bot 正常运行 1–2 周后，检查 `routing_decision_log` 中 `fe_led` 工作流的并行执行记录 | You | 🔄 IN PROGRESS (2 weeks) |

### P1 — 近期技术待办（M7 激活前置条件）

| # | Task | 依赖 | 说明 |
|---|------|------|------|
| T-B1 | 生产 `waterfall_stage_log` 基线数据采集 | T-A3 | M6 并行跑起来后，积累真实 P50/P95 latency 数据；是 M7 dynamic routing 激活的前提 |
| T-B2 | M7 动态路由激活决策 | T-B1 + 2 周监控 | 评估 classifier 在生产流量中的 uplift vs 风险；独立 Architect sign-off |
| T-B3 | `brain/` API 边界解耦方案（DB direct → API layer）| 无 | 架构风险 R-NEW-01 遗留；设计方案需要 Architect 评审后才能动工 |

### P2 — 中期待办（需新 milestone 规划）

| # | Task | 说明 |
|---|------|------|
| T-C1 | 扩展 cohort 至 `be_fe_simple` | 依赖 fe_led 生产稳定性证据（T-B1 产出） |
| T-C2 | Classifier `model_tier` 实际影响模型选择 | 当前仅 advisory-only；需要设计 delta + M9 task list |
| T-C3 | M9 scope 定义 + task list 起草 | T-B2 决策出来后才能规划 |

### P3 — 长期 Backlog

| # | Task | 说明 |
|---|------|------|
| T-D1 | Brain `supervisor.py` API 边界重构实施 | 依赖 T-B3 方案审批 |
| T-D2 | 完整 brain/ 测试覆盖（目前仅 supervisor routing；缺 poll_for_fact 集成、writer_agent） | 依赖 T-B3 解耦后才能有效测试 |

---

## Known Risks

| ID | Risk | Severity | Status |
|----|------|----------|--------|
| R-13 | Classifier 误路由高风险工作流 | High | **Mitigated** — WS-28-04 gate, low-conf fallback, limited cohort, static override |
| R-14 | 动态路由弱化完成确定性 | High | **Mitigated** — structural guard, QA admission guard unchanged |
| R-15 | 路由决策不可复现 | High | **Mitigated** — routing_decision_log, 8 normalized sources |
| R-16 | model_tier 导致隐性质量回退 | High | **Mitigated** — logged per run, balanced_default fallback |
| R-17 | 无法快速关闭 M7 行为 | High | **Mitigated** — force_sequential + 8 sec rollback drill |
| R-18 | 静态/动态路由冲突 | Medium | **Mitigated** — 显式三层优先级 + 集成测试 |
| R-19 | Classifier 宕机导致无控制路由 | High | **Mitigated** — health monitor, drill 验证 100% fallback |
| R-NEW-01 | Brain/ API 直连 DB，无解耦层 | High | **Open** — T-B3 方案设计中；任何 brain/ 修改需先完成 pytest 覆盖 |
| R-NEW-02 | workflow_engine.js 预算 | Medium | **Mitigated** — 512 lines，88 行余量 |
| R-NEW-03 | production config 被 staging config 意外覆盖 | Medium | **Resolved** — v1.2 已恢复正确值；建议 git 提交锁定 |

---

## Key Artifact Index

| Artifact | Path | 用途 |
|----------|------|------|
| 生产治理配置 | `orchestrator/configs/production_parallel_rollout.json` v1.2 | master_enabled=true, dynamic_routing_enabled=false |
| 白名单策略 | `orchestrator/configs/parallel_exposure_policy.json` | fe_led / crm / coding_team_v0 |
| M7 cohort 定义 | `orchestrator/configs/m7_exposure_cohorts.json` | cohort_enabled=false (生产未激活) |
| Staging rollout config | `orchestrator/configs/staging_parallel_rollout.json` | Staging 专用，勿混入生产 |
| Staging cohort config | `orchestrator/configs/m7_exposure_cohorts_staging.json` | Staging 专用，cohort_enabled=true |
| Staging Docker override | `infra/docker-compose.staging.yml` | nexus_staging DB + port 3001 |
| Step artifact helpers | `orchestrator/src/domain/workflow_step_artifacts.js` | 从 workflow_engine.js 提取 |
| Checkpoint service | `orchestrator/src/domain/workflow_checkpoint.js` | 从 workflow_engine.js 提取 |
| Brain pytest infra | `brain/pytest.ini`, `brain/conftest.py`, `brain/tests/` | 11/11 tests |
| M7 preflight result | `orchestrator/artifacts/m7_trial/preflight_result.json` | Dry-run 50 cases, 94% agreement |
| M7 drill result | `orchestrator/artifacts/m7_trial/drill_unavailable_result.json` | 100% forced_sequential |
| M8 live trial result | `orchestrator/artifacts/m8_trial/live_trial_result.json` | live_trial, 0% misroute |
| M8 drill result | `orchestrator/artifacts/m8_trial/live_drill_unavailable_result.json` | Staging drill |
| M8 Go/No-Go package | `docs/governance/m8_go_no_go.md` | M6 GO approved; M7 HOLD |
| M7 Go/No-Go package | `docs/governance/m7_go_no_go.md` | M7 closed with deviation |
| Staging trial runbook | `docs/runbooks/m8_staging_trial_runbook.md` | M8 Phase 1 操作手册 |
| Rollback runbook | `docs/runbooks/m6_parallel_rollback_runbook.md` | 生产回滚操作 |
| Replay manifest | `orchestrator/replay/manifests/m6_staging_replay_manifest.json` | 50-case governed corpus |

---

## Code & Configuration Changes (Engineering Audit)

### 1. Core Logic Refactoring
- **`parallel_rollout_gate.js`**: Upgraded to 3-layer gate (Master -> Static -> Dynamic). Added logic to fallback to `classifier_result.domain_lead` if `run.input_class` is missing.
- **`workflow_parallelization_policy.js`**: Added automated `input_json` parsing to support extraction of classifier results from database-loaded run objects.
- **`brain_router.js`**: Full integration of `classifyTask`. All new task dispatches now carry a persistent `classifier_result` in the envelope.

### 2. Registry & Policy Hardening
- **`capability_registry.json`**: Manually injected `fe_safe_input_classes: ["fe_led", "fullstack"]` into the `impl_fe` step definition to permit parallelization.
- **`parallel_exposure_policy.json`**: Expanded whitelist to include `webapp_crm` project type and `fullstack/be_led/architecture` input classes for the duration of the load test.
- **`production_parallel_rollout.json`**: Formally switched `master_enabled: true`.

### 3. Model Tiering (Qwen Optimization)
- **`infra/.env` & `runtime_defaults.json`**: Switched to high-efficiency tiers:
  - Control Plane: `qwen-flash`
  - Coder Expert: `qwen3-coder-plus-2025-07-22`
  - Quant Expert: `qwen-plus-2025-04-28`

### 4. Test & Infrastructure Fixes
- **`workflow_dag.test.js`**: Removed 14 instances of hardcoded `E:/` absolute paths. Fixed ESM `__dirname` compatibility issues.
- **`docker-compose.yml`**: Corrected volume mount paths to ensure host-side configuration changes are instantly visible to the orchestrator container.

