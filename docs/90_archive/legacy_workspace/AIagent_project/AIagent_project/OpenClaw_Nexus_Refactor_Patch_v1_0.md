# OpenClaw Nexus 交付体系修正文档（去重造轮子版）
版本：v1.0（Refactor Patch）  
日期：2026-03-01  
适用范围：基于你当前 **Nexus + OpenClaw + /coder + adapter** 的体系，**显式引入开源组件（Kimaki / OpenSwarm / Lobster）**，减少重复开发，把精力集中到“领域执行层（可验证规则）+ 治理层（审批/审计/权限）”。

---

## 0. 这份修正文档要解决什么问题

你当前方案的风险不是“做不出来”，而是：

- **重复造轮子**：Discord→本地会话桥接、多 Agent 协作流水线、工作流/审批可恢复，这三块在 GitHub 社区已经有高相似开源实现。
- **时间被框架层吞噬**：如果继续自研这些通用层，很容易把项目拖成“平台工程”，而不是“交付系统”。
- **真正的差异化被稀释**：你真正值钱的部分是“领域规则/可验证执行层（例如量化交易的离散手数、挂单定价、回放测试等）”，而不是又写一个 Discord bot。

---

## 1. 总体原则（Buy vs Build）

### 1.1 结论先行（强约束）
- **能直接采用的：优先 adopt（或包一层）**  
  A. Discord→本地代码库→Agent 会话桥接层：`Kimaki`  
  B. 多 Agent 协作流水线模板（Worker/Reviewer/Test/Doc）：`OpenSwarm`  
  C. 确定性工作流壳 + 审批门 + 可恢复：`Lobster`
- **必须自研沉淀的：保留并强化**  
  D. 治理控制平面：风险策略（policy-as-code）、审批与审计、权限与隔离  
  E. 领域执行层：可验证的规则、验收/回放测试、产物标准（artifacts）

> 关键心法：  
> **把 LLM 的“创造性”限制在内容生产，把“流程、门禁、恢复”交给确定性系统。**

---

## 2. 组件对齐与职责边界（修正版）

### 2.1 三块最容易重复造轮子的通用层（建议直接用开源）

#### A) Discord → 本地代码库 → Agent 会话桥接层：Kimaki（建议 Adopt）
- **目标**：Discord channel = project workspace；消息触发 OpenCode session；agent 可读写文件/跑命令/用工具。
- **你要做的不是重写 Kimaki**，而是：
  - 用它作为 /coder 的“前端入口实现”或“入口备选”
  - 只保留你 Nexus 的治理能力：审批、审计、权限、风险策略

#### B) 多 Agent 协作流水线模板：OpenSwarm（建议 Adopt / Borrow）
- **目标**：Worker/Reviewer/Test/Documenter pipeline；对接工单系统；Discord 汇报；长期记忆可选。
- **你要做的不是重写 OpenSwarm**，而是：
  - 参考其 agent role 拆分与流水线骨架
  - 把“治理/审批/风险策略”插在关键 side-effect 点（见 2.3）

#### C) 确定性工作流壳 + 审批门 + 可恢复：Lobster（建议 Adopt）
- **目标**：typed（JSON-first）pipeline；审批门；resume token；避免每一步都让模型重新规划。
- **你要做的不是重写 workflow/DSL**，而是：
  - 把你的任务执行都落在 Lobster pipeline 上
  - 把“高危动作”定义为 pipeline 的 approval gate

---

### 2.2 你真正要保留自研的两层（核心差异化）

#### D) Nexus 治理控制平面（必须自研/主导）
职责：
- **风险分级（policy-as-code）**：按文件路径、命令类型、网络访问、依赖变更、CI/infra 触碰、敏感信息触碰评分
- **审批与审计链路**：谁审批、为什么、批准了哪些 side-effect、产物可追溯（run_id/task_id/artifacts）
- **权限与隔离**：runtime sandbox、专用凭证、最小权限、敏感路径永不进入上下文/日志

#### E) 领域执行层（必须自研/主导）
职责：
- **可验证的领域规则**：例如量化执行的离散手数、挂单定价、资金 headroom、tick 取整等
- **验收与回放测试**：每次变更必须能用 replay/regression 验证
- **产物标准化（artifact pack）**：plan、diff、tests、risk_report、run_summary

---

### 2.3 修正版的体系结构（控制平面 vs 执行平面）

#### 控制平面（你主导）
- Nexus / OpenClaw Orchestrator
  - policy engine（风险评分）
  - approval service（等待/批准/恢复）
  - audit logger（全链路追溯）
  - artifact registry（产物索引/对比/回放入口）

#### 执行平面（尽量复用社区）
- Kimaki / OpenSwarm / Codex / OpenCode / SWE-agent（可插拔）
- Lobster（工作流壳，作为“执行序列的确定性容器”）

> 你需要做的是“插槽化”：  
> **入口可以换（Kimaki/Discord slash），执行引擎可以换（Codex/OpenCode），但治理与领域规则不换。**

---

## 3. 迁移路线图（止损式重构）

### 3.1 第 0 步：做一张组件替换表（必须先做）
把现有代码按三层拆：
- 入口与会话：Discord bot / slash / channel mapping / session manager
- 编排与治理：policy、approval、audit、artifact、RBAC
- 领域执行：项目模板、规则库、测试与回放

然后逐项打标签：
- **Replace**：可以被 Kimaki/OpenSwarm/Lobster 覆盖
- **Keep**：必须自研（治理/领域规则）
- **Wrap**：可保留但只做适配层

### 3.2 第 1 步：先引入 Lobster（优先级最高）
目标：让“执行序列”具备确定性与可恢复能力。
- 把你目前的多步执行改成 Lobster pipeline（JSON-first）
- 把“高危操作”统一变成 approval gate
- 把每一步的输入输出固化为 JSON schema（便于审计与回放）

### 3.3 第 2 步：入口层对齐 Kimaki（减少维护成本）
两种策略二选一（推荐 1）：
1) **Kimaki 作为默认入口**：Discord channel 直接映射 project；/coder 只是一个命令别名  
2) **Nexus 继续做入口**：但把“会话管理/线程/项目映射”委派给 Kimaki

### 3.4 第 3 步：借用 OpenSwarm 的流水线（而不是照抄）
- 把 Worker/Reviewer/Test/Doc 的 role 拆分引入你的“Skill 体系”
- 但**治理插口**必须由你控制：  
  - side-effect（依赖安装、网络下载、部署、DB migration、infra）前必须走 policy + approval
  - 产物必须回写 artifact registry

---

## 4. 你需要新增/修正的“硬规范”（不然又会漂）

### 4.1 Gate Matrix（门禁矩阵）
按风险等级定义最小门禁：
- L0（文档/低风险）：lint + markdown link check
- L1（业务代码）：lint + unit
- L2（影响鉴权/数据模型）：lint + unit + integration + schema validation
- L3（infra/部署/依赖/网络）：以上全部 + security scan + manual approval

### 4.2 Artifact Pack（产物包强制标准）
每次 run 必须输出：
- `plan.md`（拆解与验收）
- `diff.patch`
- `tests.json`（执行用例、耗时、结果）
- `risk_report.json`（命中规则与评分）
- `run_summary.md`（面向人类的摘要）
- 可选：`replay_report.md`（回放/回归结果）

### 4.3 Capability Registry（能力注册表）
把“能做什么”写成机器可读清单（避免 prompt 漂）：
- project_types（webapp_crm / quant_execution / data_pipeline / video_pipeline / ecommerce_listing / writing）
- skills（product/architect/backend/frontend/qa/devops/security + domain-specific skills）
- workflows（对应 Lobster pipeline 定义）
- policies（风险规则集）

---

## 5. 交付口径：CRM 只是 Reference Project，不是系统边界

把 CRM 文档改定位为：
- `project_types/webapp_crm/` 的模板工程
- 用它验证“交付 OS”是否能跑通：需求→设计→实现→测试→发布→回放
- 未来新增项目，只需要新增一个 Project Type + 对应 workflows/policies，不需要改系统主干

---

## 6. 风险与对策（修正版）

1) **开源项目演进快，接口变化导致适配成本**
- 对策：采用“Wrapper + Pin version + Contract tests”策略；只依赖最小接口面。

2) **OpenSwarm/Kimaki 侧的安全模型与你的审批/审计冲突**
- 对策：治理权在 Nexus；执行端必须受 Lobster gate + policy 控制；side-effect 必须经由你统一的工具层。

3) **系统复杂度上升（组件多）**
- 对策：以 Lobster 为中心把执行统一收口；入口层可替换但不叠加；保持“一个默认路径”。

---

## 7. 本修正文档的 Done Definition（DoD）

当满足以下条件时，视为“去重造轮子改造成功”：
1. 至少 1 条端到端 pipeline（Lobster）已上线：从 Discord 触发到产物包落地可追溯
2. 高危 side-effect 已统一走 approval gate，且支持 resume（不中断重复跑）
3. 入口层至少完成 Kimaki 对齐（Replace 或 Wrap 其一）
4. 能用同一套 OS 交付至少 2 类不同 Project Type（例如：CRM + 量化执行/数据管线）

---

## 8. 参考开源项目（用于对齐，不是照抄）
- Kimaki（Discord → OpenCode session）：https://github.com/remorses/kimaki
- OpenSwarm（多 Agent Worker/Reviewer pipeline + Discord/Linear）：https://github.com/Intrect-io/OpenSwarm
- Lobster（OpenClaw-native workflow shell / approval gates / resumable）：https://github.com/openclaw/lobster
- Lobster 文档（OpenClaw docs）：https://docs.openclaw.ai/tools/lobster
