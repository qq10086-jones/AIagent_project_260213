# Nexus v3.1 — 完整任务清单

**版本**: 3.1（v3.0 同行评审后修订）
**日期**: 2026-04-01
**主要变更**: ARS 重新定位为 contracts + observability；删除自建 AgentSession；修复 ARS-03/04 循环依赖；SP-01 验收以 detectSuperpowersPlugin() 为准；M4 目标改为 schema + audit hooks 统一

---

## 说明

- 任务 ID 格式：`[模块]-[序号]`
- 优先级：P0（本周）/ P1（本月）/ P2（下季度）
- 规模：S（<4h）/ M（4-16h）/ L（>16h）
- 前置：必须先完成的任务 ID

---

## EPIC A：Superpowers 落地

### [SP-01] Track A：插件注册
**优先级**: P0 | **规模**: S | **前置**: 无

**目标**：让 `detectSuperpowersPlugin()` 在运行时返回 `{ available: true }`

**⚠️ 路径说明**：检测代码（`worker-coder/adapters/opencode_adapter.js:51`）查找的是：
1. `/root/.config/opencode/plugins/superpowers.js`
2. `{cwd}/external/vendor/superpowers/.opencode/plugins/superpowers.js`

实施时以代码路径为准，不要以文档字符串为准。

**步骤**：
1. 在 `worker-coder/opencode.json` 中添加 `plugins` 字段：
   ```json
   "plugins": ["{cwd}/external/vendor/superpowers/.opencode/plugins/superpowers.js"]
   ```
2. 重新构建 worker-coder Docker 镜像
3. 在容器内运行检测验证

**验收标准**：
- [ ] `detectSuperpowersPlugin()` 在容器内返回 `{ available: true, path: "..." }`（运行代码验证，不是检查文件是否存在）
- [ ] canary 报告中 `superpowers_configured_steps >= 1`
- [ ] 工作流整体 verdict 仍为 GO

---

### [SP-02] Track C：模型分级
**优先级**: P0 | **规模**: S | **前置**: [SP-01]

**目标**：release_pack / deploy_preview 使用快速模型，降低延迟

**步骤**：
1. 在 `orchestrator/src/domain/workflow_step_builder.js` 步骤配置中：
   - `release_pack` → `model: "MiniMax-M2.7"`
   - `deploy_preview` → `model: "MiniMax-M2.7"`
   - `pm_spec / arch_design` → 保持高能力模型
2. 运行 canary 确认分级生效

**验收标准**：
- [ ] canary log 显示对应步骤使用了快速模型
- [ ] 整体工作流耗时降低 ≥ 20%
- [ ] verdict 仍为 GO

---

### [SP-03] Track B：architect 微任务注入
**优先级**: P1 | **规模**: M | **前置**: [SP-01] [SP-02]

**目标**：architect 产出结构化 workplan，impl 步骤按 workplan 执行

**步骤**：
1. 修改 `configs/prompt_scripts/registry.json` 中 `architect.system_spec.v2`，要求输出 JSON workplan
2. 在 `workflow_step_builder.js` 中解析 arch artifact，提取 workplan，注入到 impl 步骤 context
3. 实现 graceful fallback（arch 不输出 workplan 时，impl 正常运行）
4. 运行 3 次 canary 验证稳定性

**验收标准**：
- [ ] arch artifact 包含 `workplan` 字段（JSON 数组，每项有 title/files/acceptance）
- [ ] impl 步骤 context 包含 `injected_workplan`
- [ ] 3/3 canary 通过，product_fidelity = demo_usable

---

### [SP-04] Superpowers 运行时证据上报
**优先级**: P1 | **规模**: S | **前置**: [SP-01]

**目标**：在 release_pack manifest 和 Discord summary 中显示 superpowers 真实证据

**步骤**：
1. 在 `workflow_artifact_pack.js` 中收集 `superpowers_steps_used`（来自 coding_service 日志）
2. 在 `release_notes.md` 模板中增加 superpowers 使用摘要
3. 在 Discord final message 中展示

**验收标准**：
- [ ] `artifact_manifest.json` 包含 `superpowers_steps_used > 0`
- [ ] Discord 最终消息包含 superpowers 工具使用证据

---

## EPIC B：Shared Contracts + Observability（重新定位后）

> **定位说明**：这一层是控制平面基础设施（合同 + 可观测性），不是新的执行内核。
> OpenClaw 的 pi-embedded-runner 已提供 turn loop、compact、session 持久化，不在这里重建。

### [SCO-01] 定义 ToolSchema 合同
**优先级**: P0 | **规模**: S | **前置**: 无

**目标**：建立 `shared/contracts/` 目录，定义工具接口合同 schema

**步骤**：
1. 创建 `shared/contracts/tool_schema.js`
   - ToolSchema 合同：name / description / input_schema / risk_profile
   - 不包含 execute()（各 Worker 自己实现）
2. 为 worker-coder 的 3 个核心工具（read_file, write_file, bash）定义合规 schema
3. 为 worker-quant 的 quant_briefing 工具定义合规 schema
4. 编写 schema 校验器（JSON Schema validation）

**验收标准**：
- [ ] `shared/contracts/tool_schema.js` 存在，包含 ToolSchema 定义和 validate()
- [ ] 4 个工具的 risk_profile 定义完整且合理
- [ ] 单元测试：schema 校验通过/失败场景

---

### [SCO-02] 定义 WorkerResult 统一输出合同
**优先级**: P0 | **规模**: S | **前置**: 无（可与 SCO-01 并行）

**目标**：所有 Worker 返回相同格式，OpenClaw 统一处理

**步骤**：
1. 创建 `shared/contracts/worker_result.js`
   - WorkerResult schema（见设计文档 4.3 节）
   - 包含 metadata.permission_decisions 字段（为 Permission Council 预留）
2. worker-coder 输出适配到新格式
3. worker-quant 输出适配到新格式（quant_briefing.py 返回值包装）

**验收标准**：
- [ ] 两个 Worker 的返回值都通过 WorkerResult schema 校验
- [ ] OpenClaw 接收两者结果时走同一处理路径
- [ ] metadata 字段完整（duration_ms, tool_calls）

---

### [SCO-03] AuditHooks — 统一审计埋点
**优先级**: P1 | **规模**: S | **前置**: [SCO-02]

**目标**：所有 Worker 在关键节点写入统一 audit log

**步骤**：
1. 创建 `shared/contracts/audit_hooks.js`：
   - `onTaskStart(run_id, task_envelope)`
   - `onToolCall(run_id, tool_name, input, permission_advice)`
   - `onToolResult(run_id, tool_name, result_summary)`
   - `onTaskComplete(run_id, worker_result)`
   - `onTaskError(run_id, error)`
2. 在 PostgreSQL 创建 `execution_audit_log` 表
3. worker-coder 接入 AuditHooks
4. worker-quant 接入 AuditHooks

**验收标准**：
- [ ] 每次 worker-coder 任务执行，audit log 有完整记录
- [ ] 每次 worker-quant briefing，audit log 有完整记录
- [ ] 可按 run_id 查询完整执行链路

---

### [SCO-04] StreamAdapter — 对接 pi 事件到 Discord
**优先级**: P1 | **规模**: S | **前置**: 无（可并行）

**目标**：agent 执行过程实时推流到 Discord，复用 pi 现有事件系统

**⚠️ 实现说明**：这是订阅 pi 事件并转发，不是自己产生事件。
参考：`external/openclaw/src/agents/pi-embedded-subscribe.ts` 的现有事件机制。

**步骤**：
1. 创建 `shared/stream_adapter.js`
2. 订阅 pi-embedded-runner 事件（tool_use / tool_result / agent_message）
3. 转换为 Discord thread 消息格式
4. 实现节流（同类事件 1 秒内最多更新 1 次，避免刷屏）
5. worker-quant 的 briefing 结果通过同一 StreamAdapter 推送（非 pi 事件，直接调用）

**验收标准**：
- [ ] Discord thread 实时显示 worker-coder 执行步骤
- [ ] worker-quant briefing 结果推送到 Discord
- [ ] 不刷屏（节流生效）
- [ ] session_error 时显示清晰错误信息

---

### [SCO-05] Permission Council — Advisory MVP
**优先级**: P1 | **规模**: M | **前置**: [SCO-01]

**目标**：在人工审批前增加 AI advisory 预筛，减少不必要的人工干预

**⚠️ v3 定位**：Advisory only。Council 建议不能覆盖人工审批要求。
write/exec/network 敏感操作的人工审批机制**保持不变**。

**步骤**：
1. 创建 `shared/permission_council.js`
2. 实现 `SafetyAuditorAgent`（prompt-based，haiku 级别）
3. 实现 `ContextValidatorAgent`（prompt-based，haiku 级别）
4. 实现 `RiskScorerAgent`（输出风险摘要供人工参考，sonnet 级别）
5. Advisory 逻辑：
   - safe/low → Council 建议直通，附上 audit record，**不阻塞人工流程**
   - medium/high → 生成风险摘要，附在 Discord 审批请求里供人工参考
   - 明显错误操作（Council 一致判 deny）→ 在人工审批前先拦截，节省人工时间
6. 所有 Council 决策写入 `permission_audit_log` 表

**验收标准**：
- [ ] safe/low 工具：Council 附 advisory record，不增加人工审批步骤
- [ ] medium/high 工具：Discord 审批请求附带 Council 风险摘要
- [ ] Council 一致 deny 的操作：提前拦截，不再发出人工审批请求
- [ ] Council 决策不会覆盖或绕过人工审批配置
- [ ] 所有决策记录入库，可按 run_id 查询

---

## EPIC C：single_agent micro-workflow 规范

### [SA-01] 定义 single_agent micro-workflow 规范并实现四项门控
**优先级**: P1 | **规模**: M | **前置**: [SCO-02] [SCO-03]

**目标**：single_agent 路径有最低质量保证，不成为无审计的快车道

**四项强制门控**（缺任何一项不允许上线）：

| 门控 | 说明 |
|------|------|
| `evidence_id` | 每次执行唯一 ID，关联所有 audit log |
| `replay_tag` | 输入参数快照，支持重放 |
| `output_hash` | 输出内容 SHA256，一致性校验 |
| `bounded_validation` | 至少一项输出格式检查 |

**步骤**：
1. 在 `task_lifecycle.js` 中为 single_agent 路径生成 evidence_id + replay_tag
2. 在 WorkerResult 中写入 output_hash 和 bounded_validation 结果
3. 在 Brain Router 新增 `single_agent` 路由出口（brain/supervisor.py）
4. 定义 single_agent 适用范围（见设计文档 7.1）

**验收标准**：
- [ ] 所有 single_agent 任务的 WorkerResult 包含四项门控字段
- [ ] audit log 可按 evidence_id 追溯完整执行记录
- [ ] 简单 bug fix 走 single_agent 路径完成，耗时 < 60s
- [ ] 无四项门控的 single_agent 任务被拒绝执行

---

## EPIC D：治理与可观测性

### [GOV-01] permission_audit_log 表和查询接口
**优先级**: P1 | **规模**: S | **前置**: [SCO-05]

**步骤**：
1. PostgreSQL 创建 `permission_audit_log` 表：
   ```sql
   run_id, tool_name, risk_level, council_advice,
   safety_verdict, context_verdict, risk_score,
   final_human_decision, duration_ms, created_at
   ```
2. 实现按 run_id / risk_level / final_human_decision 的查询接口

**验收标准**：
- [ ] 每次 Permission Council 运行写入记录（含 advisory 建议）
- [ ] 每次人工审批决策写入 final_human_decision 字段
- [ ] 可对比 Council 建议与人工决策的一致率

---

### [GOV-02] Council Advisory 质量监控（扩权前置条件）
**优先级**: P2 | **规模**: M | **前置**: [GOV-01]

**目标**：建立 Council advisory 准确率基线，为未来扩权提供数据依据

**步骤**：
1. 积累 30 天 Council advisory 数据
2. 分析 Council 建议与人工最终决策的一致率
3. 生成周报：Council 建议的 allow 率 / deny 率 / 人工 override 率
4. 达到以下阈值后，提出 v4 扩权提案：
   - Council advisory 与人工决策一致率 ≥ 90%
   - 过去 30 天零 false negative（Council 建议 allow 但人工判断应 deny）

**验收标准**：
- [ ] 30 天数据完整
- [ ] 一致率报告可自动生成
- [ ] 阈值达标后提出 v4 扩权提案（write/exec/network 低风险场景）

---

## 里程碑汇总

| 里程碑 | 包含任务 | 目标日期 | 关键验收 |
|--------|---------|---------|---------|
| **M1: Superpowers 激活** | SP-01, SP-02 | 本周 | detectSuperpowersPlugin() = true；superpowers_configured_steps > 0；verdict=GO |
| **M2: Shared Contracts MVP** | SCO-01, SCO-02, SCO-04 | 2 周内 | 两个 Worker 输出 WorkerResult；Discord 实时推流 |
| **M3: 可观测性 + Advisory** | SCO-03, SCO-05, GOV-01 | 本月 | audit log 覆盖所有 Worker；Permission Council advisory 运行 |
| **M4: single_agent 上线** | SA-01, SP-03, SP-04, BR-01（见下） | 下月 | single_agent 四项门控齐全；Superpowers Track B 完成 |
| **M5: Council 质量基线** | GOV-02 | Q2 | 30 天数据，评估 v4 扩权可行性 |

---

### [BR-01] Brain Router 新增 single_agent 路由
**优先级**: P1 | **规模**: S | **前置**: [SA-01]

**步骤**：
1. 在 `brain/supervisor.py` 路由逻辑中新增 `single_agent` decision
2. 路由规则（见设计文档 7.1）
3. OpenClaw 接收 single_agent 决策后，确认 micro-workflow 四项门控就绪

**验收标准**：
- [ ] 简单 coding / quant / web 任务正确路由到 single_agent
- [ ] 复杂多角色任务仍走 orchestrated_workflow

---

*任务清单版本：v3.1 | 2026-04-01*
