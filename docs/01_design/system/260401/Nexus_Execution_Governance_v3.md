# Nexus 执行治理文档 v3.1
## 员工执行手册 — 防跑偏指南

**版本**: 3.1（v3.0 同行评审后修订）
**日期**: 2026-04-01
**适用对象**: 所有参与 Nexus 开发的工程师
**强制级别**: 本文档所有"禁止"条款为硬性约束，"建议"条款为最佳实践
**主要变更**: 修正执行内核边界描述；Permission Council 明确为 advisory only；补充 single_agent 质量门要求

---

## 0. 开始工作前必读

在开始任何开发任务前，确认你已阅读并理解：
1. 本文档全文
2. `docs/01_design/system/260401/Nexus_System_Design_v3.md`（最新设计）
3. 你负责模块的最新合同文档（`docs/01_design/system/260306/` 下对应文件）

**如果设计文档和你的理解有出入，先问，再动手。**

---

## 1. 系统定位红线（绝对不能违反）

### 1.1 Nexus 是控制平面，不是执行内核

```
✅ 正确：Nexus 负责路由、编排、合同定义、可观测性、审计、制品收集
❌ 错误：在 Nexus/OpenClaw 层直接实现代码生成、文件修改、shell 执行

执行内核已存在：
  - worker-coder 的 agent session → OpenClaw pi-embedded-runner（createAgentSession via pi SDK）
  - worker-quant 的批处理 pipeline → quant_briefing.py / ss7_sqlite_news_overlay.py

shared/contracts/ 里的代码是合同定义和 observability hooks，不是执行引擎。
如果你在 shared/ 里写了 execute()、run()、loop() 这类方法，停下来，问一下自己
这是合同还是执行。
```

### 1.2 不重造执行内核

```
✅ 正确：worker-coder 通过 runEmbeddedPiAgent() 调用 pi-embedded-runner 执行
✅ 正确：worker-quant 运行 quant_briefing.py / daily_run.py 执行
❌ 错误：在 shared/contracts/ 里实现 turn loop
❌ 错误：在 orchestrator 里直接 exec shell 命令
❌ 错误：绕过 pi-embedded-runner 自己管理 LLM 对话循环

OpenClaw 的 pi-embedded-runner（src/agents/pi-embedded-runner/）
已经提供了：turn loop、compact、session 持久化、tool 注入、事件流。
不要重建这些。
```

### 1.3 Agent = Contract，不是 Persona

```
✅ 正确：每个 agent 有明确的 input schema / output schema / allowed_tools / success_criteria
❌ 错误：用自然语言描述 agent 职责，没有明确的输入输出规范
❌ 错误：agent 的输出是自由文本，没有可验证的结构

修改任何 agent 的行为前，先更新对应的 contract 文档。
```

---

## 2. 代码修改规则

### 2.1 核心文件保护

以下文件修改需要 PR review（不得直接推 main）：

| 文件 | 原因 |
|------|------|
| `orchestrator/src/workflow_engine.js` | DAG 核心，改错影响所有任务 |
| `orchestrator/src/domain/workflow_step_builder.js` | 步骤构建核心 |
| `brain/supervisor.py` | 路由决策核心 |
| `configs/prompt_scripts/registry.json` | 所有 prompt 模板 |
| `configs/capability_registry.json` | 能力注册表 |
| `shared/permission_council.js` | 权限建议核心（advisory only） |

### 2.2 不得修改的内容

```
❌ 禁止修改已通过 canary 验证的工作流步骤定义，除非有新的 canary 基线
❌ 禁止在 prompt_scripts 里内嵌具体业务逻辑（prompt 只管结构和角色）
❌ 禁止在 workflow_engine.js 里添加业务判断逻辑（应该在 step_builder 里）
❌ 禁止绕过 Worker 自身的工具适配层直接调用工具
❌ 禁止绕过既有执行内核直接操作 LLM（coding 通过 pi session；quant 通过既有 pipeline）
```

### 2.3 新增功能规则

1. **先写 contract，再写代码**
   - 新 Tool：先定义 schema（name / input_schema / risk_profile）
   - 新 Agent：先写 contract 文档（input/output/tools/success_criteria）
   - 新 Worker：先实现 AgentWorkerInterface，再填充逻辑

2. **新 Tool 必须定义 risk_profile**
   - 不允许 `risk_level: undefined`
   - 保守原则：不确定时选更高 risk_level

3. **不允许静默降级**
   ```javascript
   // ❌ 错误：
   try {
     result = await superPowersExecution()
   } catch {
     result = await simpleExecution()  // 静默 fallback，没有告知用户
   }

   // ✅ 正确：
   try {
     result = await superPowersExecution()
   } catch (e) {
     logger.warn('superpowers 执行失败，降级到标准模式', e)
     streamEmitter.emit('degraded', { reason: e.message })
     result = await simpleExecution()
   }
   ```

---

## 3. Permission Council 使用规则

### ⚠️ v3 阶段定位：Advisory Only

**Permission Council 在 v3 是预筛选建议层，不是权威门。**

```
权威门（不变，v3 不能绕过）：
  write / exec / network 敏感操作 → 人工审批（Discord approve/reject）
  这个门在 v3 不变，不能用 Council 决策替代。

Advisory 层（v3 新增）：
  Council 在人工审批请求发出前先跑
  → 建议直通的（safe/low）：附 advisory record，人工审批流程仍按 policy 执行
  → 提供风险摘要的（medium/high）：附在 Discord 审批请求里供人工参考
  → Council 一致判 deny 的：提前拦截明显错误，不再触发人工审批流程

Council 建议 ≠ 最终决策。人工仍是 write/exec/network 操作的最终裁判。
```

**扩权路径**：v4+ 阶段，在 GOV-02 积累 30 天数据且一致率 ≥ 90% 后，
才评估是否将特定低风险操作类型升级为 Council 独立决策。

---

### 3.1 不允许用 Council advisory 绕过人工审批

```javascript
// ❌ 错误：Council 建议 allow 就直接执行，跳过人工审批
const advice = await council.evaluate(toolCall)
if (advice.recommendation === 'allow') {
  await tool.execute(input)  // 错误：write/exec 操作必须走人工审批
}

// ✅ 正确：Council 是预筛，不是放行门
const advice = await council.evaluate(toolCall)
// 无论 Council 建议什么，write/exec 操作仍走人工审批
// Council 结果附在审批请求里供人工参考
await requestHumanApproval(toolCall, { councilAdvice: advice })
```

### 3.2 不允许修改 risk_level 来规避审查

```javascript
// ❌ 错误：为了让工具快速执行，把 risk_level 改低
{
  name: 'delete_production_table',
  risk_profile: { level: 'safe' }  // 这是错的，会被 review 打回
}

// ✅ 正确：诚实定义风险级别
{
  name: 'delete_production_table',
  risk_profile: { level: 'critical', reversible: false, scope: 'production' }
}
```

### 3.3 Council advisory 不允许被 hardcode

```javascript
// ❌ 错误：hardcode Council 建议为 allow
if (toolName === 'my_tool') {
  councilAdvice = { recommendation: 'allow' }  // 绕过了 advisory 本身的意义
}

// ✅ 正确：如果某个工具总是被 Council 误判，
//         修改 tool 的 risk_profile 描述，或者改进 SafetyAuditor 的 prompt
```

---

## 4. Canary 验证规则

### 4.1 何时必须运行 canary

以下情况必须在合并代码前运行完整 canary：
- 修改了任何 workflow 步骤定义
- 修改了任何 prompt_scripts
- 修改了 Permission Council 逻辑
- 修改了 pi-embedded-runner 的调用方式或参数
- 修改了 worker-coder 或 worker-quant 的核心执行逻辑
- 修改了 shared/contracts/ 下任何合同 schema

### 4.2 canary 通过标准

```
必须全部满足（不允许跳过任何一项）：
  ✅ workflow_status = succeeded
  ✅ go_no_go = GO
  ✅ smoke_root_status = 200
  ✅ smoke_api_status = 200
  ✅ product_fidelity = demo_usable 或 fully_functional
  ✅ 无 critical 级别错误日志
```

如果 superpowers 相关指标（M1 里程碑后）：
```
  ✅ superpowers_configured_steps > 0
  ✅ superpowers_available_steps > 0
```

### 4.3 canary 失败处理

```
1. 不允许在 canary 失败的情况下合并代码
2. 先找到 root cause，再修复
3. 不允许通过"重试几次"来绕过不稳定的失败
4. 如果 canary 不稳定（偶发失败），先修复稳定性再说其他
```

---

## 5. 模块边界规则

### 5.1 Brain Router 边界

```
✅ Brain Router 可以做：意图分类、路由决策、task_envelope 构建
❌ Brain Router 不做：具体业务逻辑、代码分析、文件操作
❌ Brain Router 不做：直接调用 LLM 完成任务（只做路由判断）

路由决策必须输出结构化 JSON，不允许输出自由文本再由下游解析。
```

### 5.2 OpenClaw Orchestrator 边界

```
✅ Orchestrator 可以做：DAG 生成、步骤调度、状态管理、artifact 收集、QA 评估
❌ Orchestrator 不做：具体的代码修改、文件读写
❌ Orchestrator 不做：直接与用户进行多轮对话

Orchestrator 的每个步骤只能调用 Worker 或 Tool，不能直接操作文件系统。
```

### 5.3 Worker 边界

```
✅ Worker 可以做：通过既有执行内核和自身工具适配层执行具体任务
✅ Worker 可以做：调用外部执行内核（pi-embedded-runner / quant_briefing.py）
❌ Worker 不做：绕过既有执行内核直接调用 LLM
❌ Worker 不做：直接操作 PostgreSQL（通过 task_lifecycle.js 接口）
❌ Worker 不做：直接发送 Discord 消息（通过 StreamAdapter / 统一事件通道）
```

### 5.4 Shared/contracts 与 Shared utilities 边界

```
✅ shared/contracts 可以做：schema、审计 hooks、结果对象、权限 advisory 接口
✅ shared utilities 可以做：StreamAdapter、Permission Council、通用校验与格式转换
❌ shared/ 不做：turn loop、AgentSession 管理、Tool runtime 注册表
❌ shared/ 不做：业务逻辑（不知道"coding"是什么，"quant"是什么）
❌ shared/ 不做：直接连接任何数据库（通过抽象接口）

shared/ 必须是业务无关的通用基础设施。
新增 shared/ 功能前，确认其他垂类（quant / coding / future）都能复用。
```

---

## 6. 常见跑偏模式（历史教训）

### 6.1 "我直接在 orchestrator 里做就行了"

**现象**：把本该在 Worker 里的执行逻辑放到 orchestrator，或者把本该在共享合同/共享工具层的通用判断逻辑散落在各处。

**后果**：orchestrator 变成上帝对象，所有修改都需要改 workflow_engine.js，风险极高。

**正确做法**：orchestrator 只管调度，执行逻辑全部下沉到 Worker + 既有执行内核。

---

### 6.2 "这个 prompt 改一下应该没问题"

**现象**：随意修改 prompt_scripts，不跑 canary，不更新版本号。

**后果**：工作流输出格式改变，下游解析失败，canary 挂掉，但不知道是哪次改动导致的。

**正确做法**：
1. prompt 修改必须新建版本（v1 → v2），不覆盖旧版本
2. 新版本必须跑 canary 验证
3. 旧版本保留至少 2 周，确认新版本稳定后再删除

---

### 6.3 "先跑起来再说，schema 以后再定"

**现象**：Tool 没有明确的 input_schema，Worker 没有明确的输出格式，靠约定而非 schema 传递数据。

**后果**：接口不稳定，上下游频繁因为格式不匹配出错，调试困难。

**正确做法**：
1. 先定 schema，再写实现
2. 所有跨模块数据传递必须有 JSON schema 定义
3. 开发期可以宽松校验，但 schema 本身必须存在

---

### 6.4 "Permission 太麻烦，先 hardcode allow"

**现象**：为了跑通任务，把工具的 risk_level 设为 safe，或者 hardcode `decision = allow`。

**后果**：权限边界崩溃，agent 可以做任何事，一旦出现幻觉或恶意输入，后果不可控。

**正确做法**：
1. 永远不 hardcode allow
2. 如果 Council 判断太严，修改 Council 的 prompt 或调整 risk_profile 的描述
3. 开发测试期可以设置 `PERMISSION_COUNCIL_DRY_RUN=true`（只记录不拦截），但生产必须关闭

---

### 6.5 "这是临时方案，以后再重构"

**现象**：用临时方案解决问题，但临时方案变成了长期代码。

**识别信号**：代码里有 `// TODO: fix later` 超过 2 周没动；功能"凑合能用"但不满足 contract。

**正确做法**：
1. 临时方案必须在代码里打 `// TEMP: [issue_id] 描述` 标签，并创建对应 issue
2. 临时方案不允许进入 shared/contracts 或 shared utilities（基础设施必须是正式方案）
3. 每次 sprint 开始时清理超过 1 个月的 TEMP 标签

---

## 7. 开发流程规范

### 7.1 任务开始前

1. 从 `Nexus_Tasklist_v3.md` 找到对应任务 ID，确认前置任务已完成
2. 阅读相关 contract 文档（确认自己理解 input/output/boundary）
3. 如果任务影响核心文件（2.1 节列表），告知 team lead

### 7.2 开发中

1. 保持小步提交，每个 commit 对应一个明确的改动
2. 不要在一个 PR 里混入多个功能
3. 遇到设计文档没有覆盖的情况，先写 issue 讨论，不要自行决策

### 7.3 任务完成时

1. 运行相关 canary（如需，见 4.1 节）
2. 更新对应的 contract 文档（如有接口变化）
3. 更新 CHANGELOG.md
4. 如果完成了里程碑，更新 `orchestrator/artifacts/canary/` 下的最新报告

---

## 8. 紧急情况处理

### 8.1 生产问题

```
优先顺序：
1. 先恢复服务（回滚到上一个稳定版本）
2. 再找 root cause
3. 最后修复并验证

不允许在生产问题期间推未经测试的 hotfix。
```

### 8.2 canary 持续失败

```
1. 停止新功能开发，优先修复 canary
2. 检查最近 3 个 commit 哪个引入了失败
3. 如果找不到原因超过 2 小时，回滚到最后一个通过 canary 的版本
4. 记录失败模式，更新到本文档"常见跑偏模式"
```

### 8.3 Permission Council 误判率过高

```
1. 开启 PERMISSION_COUNCIL_DRY_RUN 模式（只记录，不拦截）
2. 收集 30 个样本的决策记录
3. 分析误判类型，调整对应 agent 的 prompt
4. 关闭 dry_run，再运行 50 个样本验证
```

---

## 9. 文档更新规则

| 文档 | 谁来更新 | 触发时机 |
|------|---------|---------|
| `Nexus_System_Design_v3.md` | 架构组 | 重大设计决策变更时 |
| `Nexus_Tasklist_v3.md` | PM | 每次里程碑完成 / 新任务加入 |
| 本文档 | 架构组 + 全员 | 发现新的跑偏模式时 |
| Contract 文档 | 模块负责人 | 接口变更时同步更新 |
| `CHANGELOG.md` | 开发者 | 每个 PR 合并后 |

**所有设计文档修改必须更新文件头的版本号和日期。**

---

## 10. 快速判断清单（每次开始写代码前 30 秒检查）

```
□ 我的改动在哪个模块边界内？（Brain / Orchestrator / Worker / shared）
□ 是否涉及核心保护文件？（需要 PR review）
□ 是否需要运行 canary？
□ 是否修改了任何接口？（需要更新 contract 文档）
□ 是否有 hardcode 绕过权限 / schema / 验证？
□ 是否有"临时方案"需要打 TEMP 标签？
```

六个问题都回答了，再动手。

---

*治理文档版本：v3.1 | 2026-04-01 | 发现新的跑偏模式请向架构组提 issue*
