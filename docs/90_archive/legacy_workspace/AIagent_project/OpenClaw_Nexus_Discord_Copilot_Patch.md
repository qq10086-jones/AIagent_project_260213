# OpenClaw Nexus（Discord 指令入口）Copilot/Planner 增强补丁（MD）
> 目的：让 OpenClaw 在“非执行任务时”也具备 **日常问答（LLM/多模态）+ 任务分析/规划** 能力，并避免默认把所有请求一股脑丢给预设的 Quant 工具链。  
> 适用入口：Discord 指令（Slash Command / 前缀命令均可）  
> 设计原则：**默认 CHAT，不自动 RUN；先分析再路由；高风险必 Guard + 人审**

---

## 1. 现状问题（Why）
当前 Worker-Quant 设计强调：Workflow Runner 调度 + Facts Store 溯源 + Guard 风控 + MinIO 资产 + Discord 通知/审批。  
因此 OpenClaw 的默认行为更像 **“任务编排与风控中枢（Orchestrator）”**：
- 以意图映射为主，快速路由到既有 workflow/skills
- 缺少“对话/分析/规划层”，导致：**看起来没有任务分析能力，只是把任务丢给 quant 工具**

---

## 2. 目标（What）
新增一个 **Nexus Copilot（对话 + 任务分析 + 规划 + 工具选择）** 层，支持 3 种模式：
1) **CHAT（默认）**：只回答/解释/总结/建议，不调用任何执行型 tools  
2) **PLAN**：输出可执行计划（DAG/步骤/输入输出/成本与风险），仍不执行  
3) **RUN**：在 PLAN 基础上执行（可能触发 Guard）

> 用户日常问答、需求澄清、任务拆解默认都走 CHAT/PLAN；只有明确执行或按钮确认才 RUN。

---

## 3. Discord 指令与模式切换（How: UX）

### 3.1 Slash Commands（推荐）
- `/ask <问题>` → **CHAT**
- `/plan <任务描述>` → **PLAN**（返回计划 + “Run this plan”按钮）
- `/run <task_id | plan_id>` → **RUN**（执行已生成计划）
- `/mode <chat|plan|run>` → 设置会话默认模式（可选）
- `/status <run_id>` → 查询运行状态
- `/cancel <run_id>` → 取消任务（如支持）

> 如果你不想新增太多指令，也可以只做 `/ask` + `/run`，其余通过按钮完成。

### 3.2 按钮（交互式确认）
在 PLAN 输出里附带按钮：
- ✅ **Run this plan**（进入 RUN）
- ✏️ **Edit plan**（回到 PLAN 修改参数）
- ❌ **Cancel**（终止）

### 3.3 默认策略（关键）
- 默认模式：**CHAT**
- 用户显式触发词（“执行/开始跑/回测/抓取今天/生成报告/写入数据库”）→ 允许从 CHAT 进入 PLAN，但仍**不直接 RUN**  
- 必须由：`/run` 或按钮 **Run this plan** 才进入 RUN

---

## 4. Tool Policy（防止“一上来就跑 quant”）

### 4.1 工具分级
- **Read-only / Low Risk**：检索、读取、解析、总结（不写库、不交易）
- **Write / Medium Risk**：写 DB、生成文件、批量拉取外部 API（可能限频/成本）
- **Trade / High Risk**：下单/改仓/连接券商接口/自动执行

### 4.2 规则
1) **CHAT：禁止调用任何执行型工具**（可允许纯本地推理与文本回答）  
2) **PLAN：允许调用只读工具做信息补全**（可选；默认也可以不调）  
3) **RUN：按计划调用工具**  
4) **High Risk：必须 Guard + 人审**（RUN 也只能生成 propose_orders，进入 waiting_approval）  
5) 每次 RUN 必须输出：Cost Ledger + Replay Unit（便于审计与复盘）

---

## 5. 新增组件：Nexus Copilot（架构层）

### 5.1 Copilot 职责
- 任务理解：把自然语言转成 TaskSpec（目标/约束/输入/输出/风险）
- 工具选择：决定是否需要 tools、需要哪些、顺序是什么
- 计划生成：输出可执行 DAG + 参数 + 验收条件
- 解释与可视化：用人类可读方式说明“为什么这样跑”
- 多模态入口：允许用户贴图/截图后先解释，再决定是否抽取为 facts

### 5.2 Copilot 输出结构（建议）
**TaskSpec（JSON）**
- `intent`: qa | planning | quant_research | backtest | proposal | ops
- `mode_suggested`: chat | plan | run
- `risk_level`: low | medium | high
- `requires_tools`: true/false
- `constraints`: e.g. 日期范围、标的列表、数据源
- `artifacts`: 需要产出哪些报告/图表
- `guard_required`: true/false

**PlanSpec（JSON/YAML）**
- `plan_id`
- `steps[]`: 每步的 skill、输入、输出、失败重试策略、预算
- `acceptance`: 成功判定标准
- `cost_budget`: tokens/page/api_calls/time

---

## 6. Skill Registry（新增 skills）
在现有 `worker-quant` 之外新增一个命名空间（建议）：`worker-copilot` 或 `nexus-copilot`

### 6.1 新增技能
- `skill_copilot_qa`（v1.0, Risk: Low）  
  - 纯问答/解释/总结/建议；禁止触发执行工具

- `skill_copilot_task_analyzer`（v1.0, Risk: Low）  
  - 生成 TaskSpec（意图、风险、是否需要工具、建议模式）

- `skill_copilot_plan_builder`（v1.0, Risk: Low）  
  - 生成 PlanSpec（可执行 DAG + 参数 + 预算）

- `skill_copilot_plan_refiner`（v1.0, Risk: Low）  
  - 根据用户反馈修改 plan（缩小范围、改标的、改频率）

### 6.2 与 worker-quant 的关系
- Copilot 不替代 quant skills；它只负责：
  - **先分析与规划**
  - **决定是否调用 quant pipeline**
  - **把 plan 转译为 Workflow Runner 可执行的 DAG**

---

## 7. Orchestrator 路由改造（入口从 “intent→workflow” 变为 “analyze→decide→plan/run”）

### 7.1 新路由流程（伪逻辑）
1) 收到 Discord 指令  
2) 调用 `skill_copilot_task_analyzer` → 得到 TaskSpec  
3) 根据 TaskSpec：
   - 如果 intent=qa 或 requires_tools=false → `skill_copilot_qa`（CHAT）
   - 否则 → `skill_copilot_plan_builder`（PLAN）并返回按钮
4) 只有当用户点击 **Run this plan** 或执行 `/run plan_id`：
   - Workflow Runner 执行 PlanSpec（RUN）
   - 若涉及交易/写入高风险 → Guard Gate → `waiting_approval`

---

## 8. Discord 输出模板（建议统一）

### 8.1 CHAT 输出（简洁）
- 结论（1-3 行）
- 关键依据（列表）
- 可选下一步（Plan / Run）提示

### 8.2 PLAN 输出（必须）
- 目标与范围（清晰写出）
- DAG 步骤（每步输入输出）
- 成本预算（tokens/API calls/预计耗时等级）
- 风险点（限频、数据缺失、写库、交易）
- 按钮：Run / Edit / Cancel

### 8.3 RUN 输出（必须）
- run_id、状态（running/failed/waiting_approval/succeeded）
- 关键产物链接（MinIO 图表、report.md）
- Cost Ledger 摘要
- 若 waiting_approval：展示 Approve/Reject 按钮

---

## 9. 最小落地路线（MVP）
1) 增加 `/ask`（CHAT）与 `/plan`（PLAN）两条指令  
2) 增加 `task_analyzer` 与 `plan_builder` 两个 skills  
3) Orchestrator 路由改成：先 analyzer，再决定走 qa 或 plan  
4) PLAN 输出里加 “Run this plan” 按钮，点击后才启动现有 quant DAG  
5) 对涉及交易/写库的 step 保持 Guard（propose_orders → waiting_approval）

---

## 10. 验收标准（Definition of Done）
- 用户在 Discord 输入普通问题时，系统 **不触发 quant workflow**，能直接回答（CHAT）
- 用户输入任务请求时，系统先返回计划（PLAN），并提供按钮确认
- 未经确认不执行（不 RUN）
- RUN 后仍保持：可溯源（facts/evidence/link）、可回放（Replay Unit）、可审计（Cost Ledger）
- 涉及交易的动作必须进入 `waiting_approval`

---

**End.**
