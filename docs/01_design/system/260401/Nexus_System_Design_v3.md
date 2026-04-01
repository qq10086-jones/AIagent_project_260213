# Nexus / OpenClaw — System Design v3.1
## Brain-Routed Multi-Agent Execution Platform

**版本**: 3.1（v3.0 同行评审后修订）
**日期**: 2026-04-01
**状态**: 正式设计稿
**主要变更**: 重新定位 Shared Layer（contracts + observability，非新执行内核）；Permission Council 降为 advisory；single_agent 加 micro-workflow spec；worker-quant 排除在 AgentSession 统一之外

---

## 1. 设计背景与版本演进

| 版本 | 核心变化 |
|------|---------|
| v1.x | Brain + OpenClaw + Worker 基础架构，Discord 接入 |
| v2.0 | 工作流 DAG 完整实现，artifact 层，QA/release 自动化 |
| v3.0 | 提出 agent-runtime substrate + AI Permission Council（已被 v3.1 修订） |
| **v3.1** | **修正 v3.0 过度设计：Shared Layer 重新定位为 contracts + observability；Permission Council 降为 advisory；明确复用 OpenClaw pi session 而非自建；single_agent 加 micro-workflow 规范** |

---

## 2. 系统定位（不变）

```
Nexus/OpenClaw = Control Plane（路由、编排、合同、审计、可观测性）
执行内核       = OpenClaw pi-embedded-runner（已有，createAgentSession() via pi SDK）
外部执行内核   = OpenCode / quant_briefing.py / 未来 vertical worker
```

**关键原则**：Nexus 不重造执行内核。OpenClaw 已经通过 `@mariozechner/pi-coding-agent` SDK 内嵌了完整的 agent session 能力（turn loop、compact、session 持久化、tool 注入、事件订阅），见 `openclaw/src/agents/pi-embedded-runner/`。

v3.1 的新增层是**控制平面基础设施**（合同、可观测性、事件适配），而不是又一个执行内核。

---

## 3. 整体架构（v3.1）

```
┌─────────────────────────────────────────────────────────┐
│                    Input Layer                          │
│  Discord / CLI / API / Scheduled Trigger                │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  Brain Router                           │
│  意图分类 → 路由决策                                     │
│  direct_reply / single_agent / orchestrated_workflow /  │
│  human_review_required                                  │
└───────────────────────┬─────────────────────────────────┘
                        │ canonical task_envelope
┌───────────────────────▼─────────────────────────────────┐
│              OpenClaw Orchestrator                      │
│  DAG 生成 → 步骤调度 → artifact 收集 → QA → Release      │
└──────┬──────────────────────────────────────────────────┘
       │
       │  分发到各 Worker（通过统一 WorkerContract）
       ▼
┌──────────────────────────────────────────────────────────┐
│          Shared Contracts + Observability Layer ★REVISED │
│                                                          │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  ToolSchema     │  │  WorkerResult Schema         │  │
│  │  工具接口合同    │  │  统一输出格式合同             │  │
│  └─────────────────┘  └──────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Permission Council ★ADVISORY ONLY in v3        │     │
│  │  SafetyAuditor + ContextValidator + RiskScorer  │     │
│  │  → 预筛选建议，不是权威门；                      │     │
│  │  → write/exec/network 操作人工审批仍是权威门     │     │
│  └─────────────────────────────────────────────────┘     │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  AuditHooks     │  │  StreamAdapter               │  │
│  │  统一审计埋点    │  │  对接 pi 事件 → Discord 推流  │  │
│  └─────────────────┘  └──────────────────────────────┘  │
└──────┬─────────────────────┬───────────────────────────┘
       │                     │
       ▼                     ▼
┌─────────────────┐  ┌───────────────────────────────────┐
│  worker-quant   │  │  worker-coder                     │
│  批处理 pipeline │  │  → 对接 OpenClaw pi-embedded-runner│
│  queue/DLQ/     │  │    (createAgentSession via pi SDK) │
│  artifact 录制  │  │  → Superpowers 增强（Track A/B/C） │
│  不纳入 session  │  │                                   │
│  统一           │  └───────────────────────────────────┘
└─────────────────┘
```

---

## 4. Shared Contracts + Observability Layer（重新定位后）

### 4.1 这一层的正确定位

这一层是**控制平面基础设施**，不是执行内核。它的作用是：

1. 定义合同（所有 Worker 遵守同一 schema，但各自实现）
2. 提供可观测性钩子（所有 Worker 用同一方式上报执行事件）
3. Advisory 权限预筛（不阻断，只建议）
4. 对接 Discord 推流（不自己 run agent，只转发 pi 事件）

**不做的事**：不自建 turn loop；不替代 pi-embedded-runner；不拥有 session 生命周期。

### 4.2 ToolSchema — 工具接口合同

为所有工具定义统一 schema，供 Permission Council 和 audit log 使用。这是一个**合同定义**，不是运行时注册表。

```javascript
// 工具合同 schema（定义在 shared/contracts/tool_schema.js）
{
  name: string,                    // 全局唯一工具名
  description: string,
  input_schema: JSONSchema,
  risk_profile: {
    level: 'safe' | 'low' | 'medium' | 'high' | 'critical',
    reversible: boolean,
    scope: 'local' | 'workspace' | 'external' | 'production'
  }
  // 注意：execute() 不在这里定义，由各 Worker 自己实现
}
```

工具风险分级参考：

| 工具类 | 示例 | risk_level |
|--------|------|------------|
| 读操作 | read_file, query_db | safe |
| 可逆写操作 | write_file, create_branch | low |
| 工作区写操作 | delete_file, merge_branch | medium |
| 外部调用 | call_api, run_test | high |
| 生产操作 | deploy_production, drop_table | critical |

### 4.3 WorkerResult Schema — 统一输出格式

所有 Worker 返回相同结构，OpenClaw 统一处理。

```javascript
// 定义在 shared/contracts/worker_result.js
{
  run_id: string,
  status: 'success' | 'partial' | 'failed',
  content: Array<{ type: 'text' | 'artifact_ref', ... }>,
  metadata: {
    duration_ms: number,
    tool_calls: number,
    permission_decisions: PermissionDecisionSummary[],
    tokens_used: number
  },
  artifacts: ArtifactRef[]
}
```

### 4.4 Permission Council — v3 为 Advisory Only

**v3 阶段定位：预筛选建议层，不是权威门。**

```
权威门（不变）：
  write / exec / network 敏感操作 → 人工审批（Discord approve/reject）

Advisory 层（v3 新增）：
  Permission Council 在人工审批请求发出之前先跑
  → 如果 Council 判断 safe/low → 建议直通，减少不必要的人工干预
  → 如果 Council 判断 medium/high → 提供风险摘要，辅助人工判断
  → 如果 Council 判断应 deny → 在人工审批前先拦截明显的错误操作

Council 的决策写入 audit log，但不能覆盖人工审批要求。
目标是减少人工误判和无意义的审批，而不是替代人工。
```

三个 Advisory Agent（实现方式不变，但职能降级）：

```
SafetyAuditor  → 判断操作是否安全可逆（轻量，haiku 级别）
ContextValidator → 判断操作是否符合当前任务上下文（轻量，haiku 级别）
RiskScorer     → 综合风险摘要，供人工参考（中等，sonnet 级别）
```

**扩权路径（v4+）**：在 GOV-02 积累 30 天 Council 决策数据 + 人工 feedback 后，评估是否将特定操作类型提升到 Council 独立决策。这需要数据支撑，不在 v3 范围内。

### 4.5 AuditHooks — 统一审计埋点

每个 Worker 在执行前后调用标准审计钩子，写入 `execution_audit_log`：

```javascript
// 所有 Worker 在关键节点调用
AuditHooks.onTaskStart(run_id, task_envelope)
AuditHooks.onToolCall(run_id, tool_name, input, permission_decision)
AuditHooks.onToolResult(run_id, tool_name, result_summary)
AuditHooks.onTaskComplete(run_id, worker_result)
AuditHooks.onTaskError(run_id, error)
```

### 4.6 StreamAdapter — 对接 pi 事件到 Discord

worker-coder 通过 OpenClaw pi-embedded-runner 执行任务，pi 的事件系统（`pi-embedded-subscribe.ts`）已经有完整的事件流。StreamAdapter 的工作是**订阅 pi 事件，转换格式，推送到 Discord thread**，不自己产生事件。

```javascript
// 对接 openclaw/src/agents/pi-embedded-subscribe.ts 的现有事件
piSession.on('tool_use', (event) => discordThread.post(formatToolUse(event)))
piSession.on('tool_result', (event) => discordThread.update(formatResult(event)))
piSession.on('agent_message', (event) => discordThread.update(formatMessage(event)))
```

---

## 5. worker-coder 执行路径（v3.1）

worker-coder 的 session 执行**复用 OpenClaw 现有的 pi-embedded-runner**，而不是重建。

```
任务到达 worker-coder
    ↓
task_lifecycle.js 初始化 run_id + AuditHooks.onTaskStart()
    ↓
Permission Council advisory 预筛（如需，medium+）
    ↓
（如果需要人工审批）→ Discord approve/reject 流程
    ↓
调用 openclaw/src/agents/pi-embedded-runner/run.ts
  → runEmbeddedPiAgent() 内含：
    - createAgentSession() via pi SDK
    - turn loop（pi-agent-core 提供）
    - compact（compact.ts）
    - session 持久化（session-manager-init.ts）
    - tool 注入（tool-split.ts，含 Superpowers 工具）
    - 事件流（pi-embedded-subscribe.ts）
    ↓
StreamAdapter 订阅 pi 事件 → Discord 推流
    ↓
AuditHooks.onTaskComplete() → 打包 WorkerResult
```

---

## 6. worker-quant 执行路径（v3.1，不纳入 session 统一）

worker-quant 是批处理 pipeline，执行模型根本不同于 turn-loop session。

**不做的事**：不要求 worker-quant 实现 AgentSession；不改变其 queue-driven、retry/DLQ、artifact 录制的运行模式。

**做的事**：统一输出为 WorkerResult schema；接入 AuditHooks；接入 StreamAdapter（推送 briefing 结果到 Discord）。

---

## 7. single_agent 路由 — micro-workflow 规范

single_agent 不是 DAG bypass，是一个**轻量但有质量门控的执行路径**。

### 7.1 适用场景

- 编码 bug fix（预估影响 < 3 个文件）
- quant 查询分析（非批量 pipeline）
- 简单 web 研究

### 7.2 必须满足的 micro-workflow 四项门控

以下四项是 single_agent 路径的最低质量保证，**不可跳过**：

| 门控 | 说明 | 实现位置 |
|------|------|---------|
| `evidence_id` | 每次执行有唯一可追溯 ID，关联所有 audit log 记录 | task_lifecycle.js |
| `replay_tag` | 输入参数快照，支持完整重放 | WorkerResult.metadata |
| `output_hash` | 输出内容摘要（SHA256），用于一致性校验 | WorkerResult.metadata |
| `bounded_validation` | 至少一项输出格式检查（不需要完整 QA，但不能为零） | coding_service.js / quant_briefing.py |

没有这四项，single_agent 路径不允许上线。

### 7.3 single_agent vs orchestrated_workflow

| | single_agent | orchestrated_workflow |
|---|---|---|
| 质量门数量 | 4 项（轻量） | 完整 DAG QA 体系 |
| 审计 | evidence_id + AuditHooks | 完整 artifact + replay |
| 适用规模 | 简单任务（< 60 秒预期） | 多角色多步骤项目 |
| 人工审批 | write/exec/network 操作仍需 | 按 policy 配置 |

---

## 8. Superpowers 深度融合（不变）

### 8.1 当前状态（2026-04-01）
- Track A（插件注册）：opencode.json 未配置 plugins 字段 — **P0**
- Track C（模型分级）：代码框架就绪，未激活 — **P0**
- Track B（微任务注入）：未实施 — **P1**
- 验证：superpowers_configured_steps = 0，superpowers_available_steps = 0

### 8.2 注意：plugin 路径以代码为准

`detectSuperpowersPlugin()`（`worker-coder/adapters/opencode_adapter.js:51`）检测以下路径：
1. `/root/.config/opencode/plugins/superpowers.js`
2. `{cwd}/vendor/superpowers/.opencode/plugins/superpowers.js`

SP-01 的实施必须以这两个路径为准，不要以文档中的路径字符串为准。**验收标准是 `detectSuperpowersPlugin()` 返回 `{ available: true }`**，而不是文件存在。

---

## 9. 关键路径数据流（v3.1）

```
1. 用户在 Discord 发送 /coder: 修复登录 bug

2. Brain Router
   → intent: coding, sub_intent: bug_fix, complexity: simple
   → decision: single_agent

3. OpenClaw 创建 task_envelope（run_id, evidence_id, replay_tag）

4. worker-coder 接收任务
   → AuditHooks.onTaskStart()
   → Permission Council advisory 预筛（write_file = low → 建议通过）
   → 无需人工审批（low risk）

5. 调用 runEmbeddedPiAgent()（pi-embedded-runner）
   → pi session turn loop 自动处理工具调用
   → StreamAdapter 订阅 pi 事件 → Discord thread 实时更新

6. 完成后
   → bounded_validation（格式检查）
   → output_hash 写入 WorkerResult.metadata
   → AuditHooks.onTaskComplete()
   → WorkerResult 返回 OpenClaw → Discord 最终回复

总时长：~45 秒，write 操作为 low risk，Council advisory 通过，无人工干预
```

---

## 10. 不做的事（明确范围边界）

- **不自建 turn loop**（复用 pi-embedded-runner）
- **不重建 compact / session 持久化**（pi SDK 已提供）
- **不用 Permission Council 替代人工审批**（v3 仅 advisory）
- **不强制 worker-quant 采用 AgentSession**（仅统一 schema + hooks）
- 不重造 Claude Code 完整产品形态
- 不做独立 MCP server（可后期扩展）

---

*文档版本：v3.1 | 2026-04-01 | 基于同行评审修订*
