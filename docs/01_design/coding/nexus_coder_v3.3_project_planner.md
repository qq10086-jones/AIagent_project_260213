# Nexus Coder v3.3 — Project Planner (产品级任务拆解层)

> **状态**: DESIGN REVIEW
> **日期**: 2026-04-09
> **作者**: PM / QA
> **基线**: v3.2 Capability Enhancement (全 Phase 已启用)
> **定位**: brain_router 与 workflow_engine 之间插入通用产品级拆解层

---

## 变更日志

| 版本 | 变更说明 |
|------|---------|
| v3.3 | 初稿。Project Planner 通用设计 — 补齐"产品需求 → 多 Run 编排"的缺失层 |

---

## 一、问题陈述

### 1.1 Nexus 执行模型的层次缺口

当前系统的执行决策链条：

```
用户输入 → brain_router → [路由决策] → workflow_engine → [单 run 8步] → 产出
```

`brain_router` 有三级路由：
- `direct_reply` — 问答（无执行）
- `single_agent` — 简单修复（单 LLM 调用）
- `orchestrated_workflow` — 复杂任务（8 步流水线）

问题在于**第三级内部没有再分级**。无论是"加一个 API endpoint"还是"做一套 ERP 系统"，都走同一个 `coding_team_v0` 单 run。8 步流水线在 `impl_be` / `impl_fe` 步骤只能做 1 次 LLM 调用，无法在单次调用中生成整个多模块系统的代码。

### 1.2 缺失的层：产品级编排

| 粒度 | 现有能力 | 示例 |
|------|---------|------|
| 命令级 | `composite_planner.js` — 分号/换行拆分多条指令 | "查新闻; 跑发现; 设仓位" |
| **功能级** | **`coding_team_v0` — 单 workflow run (8 步)** | "加一个登录 API" |
| **产品级** | **缺失** | "做一套客户管理系统" |

### 1.3 解决方案

在 `brain_router` 路由决策之后、`workflow_engine.startWorkflowRun()` 之前，插入 **Project Planner** 层。该层将产品级需求拆解为 N 个功能级 workflow runs，每个 run 复用现有的 `coding_team_v0` 流水线。

```
用户输入
    ↓
brain_router.routeTaskRequest()
    ↓ decision = "orchestrated_workflow" && complexity = "complex"
project_planner.decompose()          ← 新增: 拆解为 N 个 runs
    ↓ 产出 project_plan.json
project_executor.executeProjectPlan() ← 新增: 按依赖图编排
    ↓ 跨 run 上下文传递
workflow_engine.startWorkflowRun() × N  ← 现有: 每个 run 复用 8 步流水线
```

**关键原则**: planner 只做拆解和编排，不改变单 run 内部的任何行为。每个 run 仍然走完整的 pm_spec → arch_design → impl → test → qa → release 流程。

---

## 二、现有系统能力审计（复用清单）

planner 必须复用（而非重建）的现有基础设施：

| 组件 | 文件 | 复用方式 |
|------|------|---------|
| LLM 调用基础设施 | `nlp/router.js` `callQwenChat()` | planner 复用同一 LLM client 做需求拆解 |
| 复杂度判定 | `brain_router.js` `inferComplexity()` | complex 时触发 planner |
| 项目类型推断 | `coding_project_type.js` `inferCodingProjectType()` | 从用户输入推断 project_type，决定 artifact 要求和风控级别 |
| 任务分类 | `brain_router_classifier.js` `classifyTask()` | 每个子 run 独立分类 work_shape |
| Task class 枚举 | `worker_coding_task_classes.json` | planner 产出的 runs 只能使用已注册的 task_class |
| Capability 注册表 | `capability_registry.json` | 按 project_type 查询 risk_profile / required_artifacts / parallelization 策略 |
| Workflow 定义 | `coding_team_v0.json` | 每个 run 走同一 8 步 workflow |
| DAG 并行调度 | `workflow_parallelization_policy.js` + `parallel_rollout_gate.js` | run 内部 BE/FE 并行策略不变 |
| Handoff 合约 | `step_artifact_contract.js` | 步骤间交接 schema 不变 |
| Refinement Re-entry | `workflow_refinement_service.js` | 失败的 run 可用 lineage 重试 |
| 事件总线 | `recordEvent()` | 新增 project-level 事件类型 |

---

## 三、架构设计

### 3.1 模块拓扑

```
orchestrator/src/vnext/
├── brain_router.js              (现有 — 不修改)
├── brain_router_classifier.js   (现有 — 不修改)
├── brain_router_policy.js       (现有 — 不修改)
├── composite_planner.js         (现有 — 不修改, 命令级拆解)
├── coding_project_type.js       (现有 — 不修改)
├── project_planner.js           ← 新增: 产品级需求拆解 (LLM 驱动)
├── project_plan_contract.js     ← 新增: project_plan.json schema 验证
├── project_executor.js          ← 新增: 多 run 编排执行器
└── runtime_dispatch.js          (现有 — 修改: 插入 planner 决策点)
```

### 3.2 数据流（通用）

```
                    ┌─────────────────────────────┐
                    │  任意产品级需求 (raw_input)     │
                    │  "做一套XX系统" / "build XX"   │
                    └────────┬────────────────────┘
                             ↓
                    ┌────────────────────────┐
                    │   brain_router         │
                    │   → intent: coding     │
                    │   → complexity: complex │
                    │   → project_type: 推断  │
                    └────────┬───────────────┘
                             ↓
              ┌──────────────────────────────────────┐
              │     project_planner.decompose()      │
              │                                      │
              │  输入:                                │
              │    raw_input, project_type,           │
              │    capability_registry, task_classes  │
              │                                      │
              │  处理:                                │
              │    1. 构建 decomposition prompt       │
              │    2. 调用 LLM (MiniMax / Ollama)    │
              │    3. 提取 JSON + schema 验证         │
              │    4. DAG 无环校验 + topo sort        │
              │    5. 推断 tech_stack (from registry) │
              │    6. 验证 target_paths 不重叠        │
              │                                      │
              │  失败回退:                             │
              │    LLM 失败 / schema 不通过           │
              │    → 返回 single-run fallback         │
              │    → 退化为现有行为                    │
              └──────────┬───────────────────────────┘
                         ↓
              ┌───────────────────────────────────┐
              │   project_plan.json               │
              │                                   │
              │  - project_id                     │
              │  - modules[] (逻辑模块分组)        │
              │  - runs[] (可执行单元)             │
              │  - dependency_graph               │
              │  - execution_strategy             │
              │  - project_constraints            │
              └──────────┬────────────────────────┘
                         ↓
              ┌───────────────────────────────────┐
              │   [可选] 用户确认                   │
              │   confirm_mode: "manual" → 暂停    │
              │   用户可增删改 runs 后再执行         │
              └──────────┬────────────────────────┘
                         ↓
              ┌───────────────────────────────────────┐
              │     project_executor                   │
              │     .executeProjectPlan(plan, context) │
              │                                       │
              │  1. topo sort → 计算执行波次 (waves)   │
              │  2. Wave N: 启动当前可执行的 runs      │
              │  3. 每个 run → startWorkflowRun()      │
              │  4. run 完成 → 提取 artifact           │
              │  5. injectSharedContext() → 下游 run   │
              │  6. 触发下一批可执行 runs               │
              │  7. 全部完成 → project_summary.json    │
              └───────────────────────────────────────┘
```

### 3.3 project_plan.json Schema（通用定义）

```jsonc
{
  // === 项目元数据 ===
  "project_id": "string (auto-generated, e.g. proj-<timestamp>-<slug>)",
  "project_title": "string (用户需求的一句话摘要)",
  "created_at": "ISO 8601 timestamp",
  "decomposition_model": "string (LLM model used, e.g. MiniMax-M2.7)",
  "decomposition_confidence": "number 0-1 (LLM 自评分解置信度)",

  // === 逻辑模块 (分组, 不影响执行) ===
  "modules": [
    {
      "module_id": "string (e.g. mod-auth, mod-dashboard)",
      "title": "string",
      "description": "string"
    }
  ],

  // === 可执行单元 ===
  "runs": [
    {
      "run_key": "string (e.g. R-01, R-02, ...)",
      "module_id": "string (对应 modules[].module_id)",
      "task_class": "enum: fe_create | fe_modify | be_create | bug_fix | artifact_completion",
      "title": "string (< 80 chars)",
      "prompt": "string (200-500 chars, 给 PM 步骤的详细需求描述)",
      "target_paths": ["string[] (workspace 相对路径, 不同 run 之间不重叠)"],
      "depends_on": ["string[] (上游 run_key 列表)"],
      "shared_context": {
        "from_runs": ["string[] (上游 run_key)"],
        "artifacts": ["string[] (要从上游提取的 artifact 相对路径)"]
      },
      "estimated_complexity": "enum: simple | medium | complex",
      "acceptance_criteria": ["string[] (AC-xxx: 具体可验证条件, 2-5 条)"]
    }
  ],

  // === 依赖图 (冗余, 用于快速校验) ===
  "dependency_graph": {
    "<run_key>": ["<upstream_run_key>", "..."]
  },

  // === 执行策略 ===
  "execution_strategy": {
    "max_parallel_runs": "number (default: from config, cap: 3)",
    "failure_policy": "enum: stop_dependents | stop_all | continue_all",
    "retry_failed_runs": "boolean",
    "max_retries_per_run": "number (default: 1)"
  },

  // === 项目约束 (从 registry + 用户输入推断) ===
  "project_constraints": {
    "project_type": "string (from coding_project_type.js)",
    "workspace_root": "string (e.g. workspace/sandbox/<project-slug>/)",
    "tech_stack_hints": "object | null (LLM 推断, 仅建议性, 不强制)",
    "naming_convention": "string | null"
  }
}
```

**Schema 设计原则**:
1. **`task_class` 枚举来源于 `worker_coding_task_classes.json`** — 不硬编码，从注册表读取
2. **`project_type` 由 `coding_project_type.js` 推断** — 不由用户指定
3. **`tech_stack_hints` 仅建议性** — 不写入 PM prompt 的硬约束，由架构师步骤自行决策
4. **`target_paths` 不重叠校验** — contract 层强制，防止并行 runs 冲突写文件
5. **`shared_context.artifacts` 使用现有 artifact 路径** — 复用 `step_artifact_contract.js` 定义的标准路径（如 `handoff/be_to_fe.json`、`plan/interfaces.md`）

### 3.4 Planner 触发条件

在 `brain_router.routeTaskRequest()` 返回后，增加判定逻辑：

```javascript
// runtime_dispatch.js — 修改点
const routeResult = routeTaskRequest({ ... });

if (routeResult.decision === "orchestrated_workflow"
    && routeResult.route.complexity === "complex"
    && isProjectPlannerEnabled()) {

  const plan = await projectPlanner.decompose({
    raw_input,
    project_type: routeResult.route.execution_plan.project_type,
    classifier_result: routeResult.route.classifier_result,
    registry,          // capability_registry.json
    task_classes,      // worker_coding_task_classes.json
  });

  if (plan.runs.length > 1) {
    return projectExecutor.executeProjectPlan(plan, context);
  }
  // 单 run 退化为普通 workflow — 不启动 executor
}

// 原有逻辑: 单 workflow run
```

**触发条件**（三者同时满足）：
1. `decision === "orchestrated_workflow"` — brain_router 已决定需要编排
2. `complexity === "complex"` — 复杂度为复杂级（`COMPLEX_CUE_RE` 匹配 或 >220字符 或 >35词）
3. `project_planner_enabled === true` — feature flag 开启

**不触发时**: 退化到现有行为（单 run），零侵入。

### 3.5 LLM Decomposition Prompt

```markdown
<role>
你是 Nexus 项目规划器。你的职责是将产品级需求拆解为多个可独立执行的功能模块（Run）。
每个 Run 将由一条完整的 coding_team_v0 流水线执行（PM→架构→实现→测试→QA→发布）。
</role>

<task>
将以下需求拆解为多个独立的 Runs。
需求: {{RAW_INPUT}}
</task>

<constraints>
已注册 task_class: {{TASK_CLASSES_JSON}}
项目类型: {{PROJECT_TYPE}}
目标 workspace: {{WORKSPACE_ROOT}}
</constraints>

<rules>
1. 粒度控制: 每个 Run 必须是一个 LLM 在单次调用中能完成的功能单元
   - 好的粒度: "用户认证 API（注册+登录+JWT验证）" — 1个模块, 3-5个文件
   - 坏的粒度: "整个后端" — 太大; "加一行配置" — 太小
2. task_class 必须从已注册枚举中选择，不能自创
3. depends_on 声明因果依赖:
   - 后端 API 必须先于调用它的前端页面
   - 共享数据模型必须先于使用它的业务逻辑
   - 不相关的模块之间不要加假依赖
4. target_paths 在不同 Run 之间不得重叠
5. 每个 Run 包含 2-5 条具体的验收标准（AC），可验证、可自动化
6. 总 Run 数量: 3-12 个（太少覆盖不全，太多管理成本高）
7. shared_context.artifacts 只能引用标准 artifact 路径:
   handoff/pm_to_architect.json, handoff/architect_to_impl.json,
   handoff/be_to_fe.json, handoff/impl_to_qa.json,
   plan/interfaces.md, plan/spec.md, plan/arch.md, plan/workplan.json
8. prompt 字段: 写给 PM 角色看的需求描述 (200-500字)，包含:
   - 要实现的功能
   - 关键数据结构/接口
   - 与其他模块的边界约束
</rules>

<output_format>
返回 JSON，严格符合以下结构:
{
  "modules": [{ "module_id": "mod-xxx", "title": "...", "description": "..." }],
  "runs": [{
    "run_key": "R-01",
    "module_id": "mod-xxx",
    "task_class": "be_create",
    "title": "...",
    "prompt": "...",
    "target_paths": ["{{WORKSPACE_ROOT}}子路径/"],
    "depends_on": [],
    "shared_context": { "from_runs": [], "artifacts": [] },
    "estimated_complexity": "medium",
    "acceptance_criteria": ["AC-R01-1: ..."]
  }],
  "tech_stack_hints": { "backend": "...", "frontend": "...", "test": "..." }
}
</output_format>
```

**Prompt 设计原则**:
1. **`{{TASK_CLASSES_JSON}}` 运行时注入** — 从 `worker_coding_task_classes.json` 读取，不硬编码
2. **`{{PROJECT_TYPE}}` 运行时注入** — 从 `coding_project_type.js` 推断结果
3. **`{{WORKSPACE_ROOT}}` 运行时注入** — 从 project_slug 生成
4. **`shared_context.artifacts` 限定标准路径** — 不允许 LLM 自创 artifact 路径
5. **`tech_stack_hints` 定义为 hints** — 不是硬约束，架构步骤可以覆盖

### 3.6 project_plan_contract.js 验证规则

LLM 输出不可信，必须经过以下硬验证才能执行：

| 规则 | 校验内容 | 失败行为 |
|------|---------|---------|
| C-01 | `runs.length >= 2 && runs.length <= max_runs` | reject (太少退化单 run, 太多拒绝) |
| C-02 | 每个 `run.task_class` 存在于 `worker_coding_task_classes.json` | reject |
| C-03 | 每个 `run.run_key` 唯一且匹配 `R-\d+` 格式 | reject |
| C-04 | `dependency_graph` 无环 (topo sort 成功) | reject |
| C-05 | 每个 `run.depends_on[]` 引用的 run_key 存在于 runs 中 | reject |
| C-06 | `target_paths` 跨 run 不重叠 | reject |
| C-07 | 每个 `run.acceptance_criteria.length >= 1` | reject |
| C-08 | 每个 `run.prompt.length >= 50` | reject (prompt 太短无法产出质量) |
| C-09 | `shared_context.artifacts[]` 只包含标准 artifact 路径 | warn + strip 非法路径 |
| C-10 | `modules[]` 中每个 `module_id` 至少被一个 run 引用 | warn (孤立模块) |

### 3.7 Project Executor 状态机

```
PROJECT_CREATED
    ↓ contract 验证 (C-01 ~ C-10)
PROJECT_VALIDATED
    ↓ topo sort → 计算 waves
PROJECT_SCHEDULED
    ↓ [可选] 用户确认 (confirm_mode: "manual")
PROJECT_CONFIRMED
    ↓ 启动 Wave 1 (无依赖 runs)
PROJECT_RUNNING
    ↓ 每个 run 完成后:
    │   1. 提取 artifacts
    │   2. injectSharedContext() → 下游 runs
    │   3. 启动新解锁的 runs (下一 wave)
    │   4. 记录 project-level event
    ↓ 全部 runs 完成 OR 不可恢复失败
PROJECT_COMPLETED / PROJECT_PARTIAL_FAILURE
    ↓ 生成 project_summary.json
PROJECT_REPORTED
```

**失败策略**:

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `stop_dependents` (默认) | 失败 run 的下游 blocked，并行分支继续 | 大多数情况 |
| `stop_all` | 任何 run 失败即停止所有 | 强一致性要求 |
| `continue_all` | 记录失败但继续执行所有 runs | 探索性原型 |

失败的 run 可通过 v3.2 Phase 1.5 的 refinement re-entry 重试（带 lineage 上下文）。

### 3.8 跨 Run 上下文传递

```javascript
// project_executor.js — 通用上下文注入
async function injectSharedContext(downstreamRun, completedRuns) {
  const shared = downstreamRun.shared_context;
  if (!shared?.from_runs?.length) return downstreamRun;

  const upstream_artifacts = [];
  for (const fromKey of shared.from_runs) {
    const upstream = completedRuns[fromKey];
    if (!upstream?.artifact_dir) continue;

    for (const relPath of (shared.artifacts || [])) {
      const absPath = path.join(upstream.artifact_dir, relPath);
      if (fs.existsSync(absPath)) {
        upstream_artifacts.push({
          source_run: fromKey,
          path: relPath,
          content: fs.readFileSync(absPath, "utf8"),
        });
      }
    }
  }

  // 注入到下游 run 的 workflow input
  return {
    ...downstreamRun,
    context_packet: {
      ...downstreamRun.context_packet,
      upstream_artifacts,
      project_id: downstreamRun.project_id,
      run_key: downstreamRun.run_key,
    },
  };
}
```

注入的 `upstream_artifacts` 通过 `workflow_engine.startWorkflowRun()` 的 `project_context` 参数传入 PM 步骤的 payload，使 PM 能看到上游 run 产出的接口定义和数据模型。

---

## 四、与现有系统的集成点

### 4.1 修改清单

| 文件 | 修改类型 | 描述 | 影响范围 |
|------|---------|------|---------|
| `runtime_dispatch.js` | 修改 | orchestrated_workflow + complex + flag → planner 路径 | 新增 if 分支，else 不变 |
| `runtime_defaults.json` | 修改 | 新增 `project_planner_*` flags | 纯配置 |
| `workflow_engine.js` | 修改 | `startWorkflowRun()` 接受可选 `project_context` 参数 | 可选参数，不影响现有调用 |
| `coding_service.js` | 修改 | `delegateTask()` 的 context_packet 可含 `upstream_artifacts` | 现有字段扩展，不影响无 artifacts 的调用 |
| `project_planner.js` | **新增** | LLM 驱动的需求拆解 | 无侵入 |
| `project_plan_contract.js` | **新增** | plan schema 验证 | 无侵入 |
| `project_executor.js` | **新增** | 多 run 编排执行器 | 无侵入 |

### 4.2 不修改的文件

所有现有模块保持不变：`brain_router.js`、`brain_router_classifier.js`、`brain_router_policy.js`、`composite_planner.js`、`coding_project_type.js`、`step_artifact_contract.js`、`surgical_patch.js`、`context_resolver.js`、`refinement_context_builder.js`、`subtask_generator.js`、`parallel_rollout_gate.js`、`coding_team_v0.json`、`capability_registry.json`、`worker_coding_task_classes.json`。

### 4.3 Feature Flags

```json
{
  "worker_coder": {
    "project_planner_enabled": false,
    "project_planner_max_runs": 12,
    "project_planner_max_parallel_runs": 2,
    "project_planner_failure_policy": "stop_dependents",
    "project_planner_confirm_mode": "manual"
  }
}
```

| Flag | 默认 | 含义 |
|------|------|------|
| `project_planner_enabled` | false | 总开关 |
| `project_planner_max_runs` | 12 | 单个 project 最大 run 数 |
| `project_planner_max_parallel_runs` | 2 | 同时执行的 run 数上限 (留 1 slot 给 `max_concurrent_workflows=3`) |
| `project_planner_failure_policy` | stop_dependents | 失败策略 |
| `project_planner_confirm_mode` | manual | manual=人工确认后执行, auto=拆解后自动执行 |

---

## 五、验收标准

### 5.1 功能验收

| ID | 条件 | 验证方法 |
|----|------|---------|
| AC-01 | 输入任意产品级需求，planner 拆解为 2-12 个 runs | 多场景单元测试 (中文/英文/混合) |
| AC-02 | 每个 run 的 task_class 属于 `worker_coding_task_classes.json` 枚举 | contract 验证 |
| AC-03 | runs 依赖图无环 (DAG) | topo sort 测试 (含环检测 reject) |
| AC-04 | target_paths 跨 run 不重叠 | contract 验证 |
| AC-05 | 跨 run 上下文传递: 上游 artifact 出现在下游 context_packet | 集成测试 |
| AC-06 | 失败传播: 上游 run 失败 → 下游 blocked，并行分支不受影响 | 状态机测试 |
| AC-07 | flag=false 时行为与 v3.2 完全一致 | 回归测试 |
| AC-08 | LLM 拆解失败时退化为单 run | fallback 测试 |
| AC-09 | 完成后产出 project_summary.json 汇总报告 | 输出验证 |
| AC-10 | confirm_mode=manual 时暂停等待用户确认 | 交互测试 |

### 5.2 质量门禁

| 门禁 | 要求 |
|------|------|
| 单元测试 | contract ≥8, planner ≥5, executor ≥5 |
| 集成测试 | ≥1 个 2-run 依赖链端到端 |
| 回归 | 现有 104 测试全绿 |
| Schema | 任何 plan 必须通过 contract 验证才能执行 |

---

## 六、风险评估

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| LLM 拆解质量不稳定 — 粒度不当、依赖遗漏、task_class 错误 | 高 | 1) contract 硬验证 reject 不合规 2) confirm_mode=manual 人工审批 3) fallback 到单 run |
| 跨 run 上下文丢失 — 下游 run 缺少上游接口信息 | 中 | 显式 shared_context 声明 + artifact 存在性校验 + warn log |
| 长时间执行中断 — N 个 runs 串行可能耗时很长 | 中 | 1) 状态持久化支持断点续跑 2) 并行 wave 减少总时间 |
| 并行 run 冲突写同一文件 | 低 | target_paths 不重叠校验 (contract C-06) |
| 单 run 失败级联 — 一个 run 失败阻塞大量下游 | 中 | stop_dependents 策略 + refinement re-entry 重试 |
| Planner prompt 与 registry 不同步 — 新增 task_class 后 prompt 仍用旧枚举 | 低 | prompt 运行时从 registry 注入，不硬编码 |

---

## 七、实施阶段划分

### Phase A: Planner Core (MVP)

**目标**: 能拆解、能验证、可人工检查拆解结果

| 任务 | 文件 | 要点 |
|------|------|------|
| P-A1 | `project_plan_contract.js` | C-01~C-10 验证规则 + topo sort + 无环校验 |
| P-A2 | `project_planner.js` | LLM prompt 构建 + JSON 提取 + schema 验证 + fallback |
| P-A3 | `runtime_defaults.json` | 新增 feature flags |
| P-A4 | `test/project_plan_contract.test.js` + `test/project_planner.test.js` | 各 5+ 用例 |

**Phase A 交付物**: 输入需求 → 输出合法 project_plan.json（不自动执行）

### Phase B: Executor Engine

**目标**: 能按 plan 自动编排多个 workflow runs

| 任务 | 文件 | 要点 |
|------|------|------|
| P-B1 | `project_executor.js` | 状态机 + wave 调度 + 失败策略 |
| P-B2 | `project_executor.js` | `injectSharedContext()` 跨 run artifact 注入 |
| P-B3 | `workflow_engine.js` | `startWorkflowRun()` 接受 `project_context` |
| P-B4 | `runtime_dispatch.js` | 插入 planner 决策点 |
| P-B5 | `test/project_executor.test.js` | 2-run 依赖链 + 并行 + 失败传播 |

**Phase B 交付物**: plan → 自动编排 N 个 runs → 收集结果

### Phase C: Governance & UX

**目标**: 产品级治理完整性

| 任务 | 文件 | 要点 |
|------|------|------|
| P-C1 | `project_executor.js` | `project_summary.json` 汇总报告 |
| P-C2 | `project_planner.js` | confirm_mode=manual 交互流 |
| P-C3 | `project_executor.js` | 断点续跑 (从最后完成的 run 继续) |
| P-C4 | 全量测试 | 现有 104 + 新增 ≥18 测试全绿 |

**Phase C 交付物**: 完整的项目级治理能力

---

## 八、附录

### A. 已支持的项目类型 (from capability_registry.json)

| project_type | 推断关键词 | risk_profile | required_artifacts |
|-------------|-----------|-------------|-------------------|
| `webapp_crm` | CRM, 客户管理, sales pipeline | medium | spec, arch, diff, verification, manifest |
| `generic_app` | app, dashboard, 系统, 全栈 | medium | spec, arch, diff, verification, manifest |
| `single_file_html` | landing page, 单页, html only | low | spec, diff, verification, manifest |
| `generic_coding_task` | (default fallback) | dynamic | spec, arch, diff, verification, manifest |
| `coding_task` | (base type) | dynamic | diff, plan, tests, run_summary |

Planner 将根据推断出的 project_type 从 registry 获取 `required_artifacts` 和 `risk_profile`，注入每个 run 的执行配置。

### B. 已注册的 task_class (from worker_coding_task_classes.json)

| task_class | 含义 | 典型粒度 |
|-----------|------|---------|
| `fe_create` | 前端新建页面/组件 | 1 页面 + 2-3 组件 |
| `fe_modify` | 前端修改已有页面 | 1-3 文件修改 |
| `be_create` | 后端新建 API/服务 | 1 模块 + 3-5 文件 |
| `bug_fix` | 跨层 bug 修复 | 1-2 文件定点修复 |
| `artifact_completion` | Artifact 补全 | 文档/配置生成 |

### C. 示例: 不同规模的预期拆解

**小型项目** (3-4 runs): "做一个待办事项应用"
```
R-01 [be_create] TODO CRUD API + SQLite
R-02 [fe_create] TODO 列表 + 添加 + 完成切换
R-03 [fe_modify] 添加筛选/排序/批量操作
```

**中型项目** (5-7 runs): "做一套客诉管理系统 + 质保书发行系统"
```
R-01 [be_create] 客诉数据模型 + CRUD API         depends: []
R-02 [be_create] 客诉工作流引擎 (状态机)          depends: [R-01]
R-03 [fe_create] 客诉管理前端                     depends: [R-01, R-02]
R-04 [be_create] 质保书模板 + PDF 生成 API         depends: []
R-05 [fe_create] 质保书前端                       depends: [R-04]
R-06 [be_create] 客诉↔质保书联动                  depends: [R-01, R-04]
R-07 [fe_modify] 统一仪表盘                       depends: [R-03, R-05, R-06]
```

**大型项目** (8-12 runs): "做一个多租户 SaaS 电商平台"
```
R-01 [be_create] 多租户认证 + RBAC
R-02 [be_create] 商品目录 CRUD + 图片上传
R-03 [be_create] 购物车 + 订单引擎
R-04 [be_create] 支付集成 (Stripe/PayPal)
R-05 [fe_create] 商品浏览 + 搜索前端
R-06 [fe_create] 购物车 + 结算流前端
R-07 [fe_create] 租户管理后台
R-08 [be_create] 库存管理 + 低库存告警
R-09 [fe_create] 订单管理 + 物流追踪前端
R-10 [be_create] 报表 + 数据导出 API
R-11 [fe_create] 分析仪表盘
R-12 [fe_modify] 统一导航 + 通知中心
```

### D. 关键设计决策记录 (ADR)

| ADR | 决策 | 理由 |
|-----|------|------|
| ADR-01 | Planner 复用现有 LLM client (`callQwenChat`) | 不新增 provider 依赖 |
| ADR-02 | plan 走 schema 硬验证 | LLM 输出不可信，必须 contract 校验后才能执行 |
| ADR-03 | 跨 run 上下文通过文件系统传递 | artifact 已是文件，直接读取最简，无需额外序列化层 |
| ADR-04 | max_parallel_runs 默认 2 | 与 `max_concurrent_workflows_validated=3` 对齐，留 1 slot |
| ADR-05 | 失败策略 stop_dependents | 并行分支不应因无关 run 失败而停止 |
| ADR-06 | 默认 confirm_mode=manual | 人先审拆解结果，降低 LLM 拆解错误的风险 |
| ADR-07 | 单 run 退化为现有行为 | LLM 判断不需要拆（需求够小）时直接走单 run |
| ADR-08 | task_class / project_type 从 registry 运行时读取 | 新增类型时 planner 自动适配，无需改代码 |
| ADR-09 | tech_stack 为 hints 而非约束 | 不同 run 的架构步骤应有自主决策权 |
