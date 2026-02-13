# Project OpenClaw Nexus（项目蓝图 v1.1）

> **定位**：以 **OpenClaw** 作为“控制平面（Control Plane）”的中枢调度器，统一任务状态机、工具协议、审计与风险门禁；将具体业务执行（量化/电商/内容/绘图/剪辑等）下沉为可替换的 **Worker（执行平面 / Tentacles）**。  
> **目标**：跑通至少 1 条可验证的“数据→决策→执行→反馈→复盘”的商业闭环，并可在同一数据与审计体系下按插件方式接入更多触角。  
> **硬件**：AMD RX 7900 XTX（ROCm/DirectML，本地推理与 ComfyUI 绘图）。  
> **强制约束**：**全系统 Docker 隔离运行**；高风险动作默认 **不允许全自动**，必须经过风险门禁（Guard/Audit）。

---

## 0. 目录
1. 背景与愿景  
2. 关键原则（可靠性 / 可恢复 / 可审计）  
3. 系统架构总览（控制平面 / 执行平面 / 数据平面）  
4. Docker 隔离与部署拓扑  
5. 任务模型与状态机（Task FSM）  
6. Tool Contract（工具调用协议）  
7. 大脑池（Brain Pool）与路由策略  
8. 安全审计官（Guard / Audit）与权限体系  
9. 数据层：事件日志与指标口径（Event & Metrics）  
10. MVP 闭环（先跑通一条）  
11. 里程碑与验收标准  
12. 风险清单与备用方案（Fallback Plan）  
13. 待决策项（Open Questions）

---

## 1. 背景与愿景

本项目旨在构建一个本地优先（Local-first）、可扩展（Plugin-ready）、可审计（Audit-ready）的自主商业系统，将 LLM 的“建议能力”升级为在安全边界内可执行的“闭环能力”。

---

## 2. 关键原则（Reliability Principles）

### 2.1 薄中枢（Thin Orchestrator）
OpenClaw **不承载业务逻辑**与 GPU 重活，只做：
- 任务调度与队列（routing / scheduling）
- 任务状态机与恢复（FSM / recovery）
- 工具调用协议与失败收敛（tool contract / retries / rollback）
- 审计日志（append-only）
- 风险门禁与审批流（guard / approval）

### 2.2 可恢复优先（Recoverability > Perfection）
- 任何步骤失败，都必须能回到一致状态（可重试 / 可回滚 / 可人工接管）
- 中枢容器可随时重启，重启后从数据层恢复未完成任务

### 2.3 默认安全（Safe-by-default）
- 涉及资金/账号/不可逆的动作：默认进入 `waiting_approval`
- 允许自动化的范围必须显式白名单化（policy allowlist）

---

## 3. 系统架构总览（3 平面）

### 3.1 架构图（Mermaid）
```mermaid
graph TD
  User[用户指挥官] -->|自然语言/指令| Core[OpenClaw Orchestrator<br/>(Control Plane)]

  subgraph Brain[Brain Pool]
    L0[Local LLMs<br/>Ollama/ROCm] --> Core
    L1[Cloud LLMs<br/>API] --> Core
  end

  subgraph Exec[Execution Plane / Workers]
    W1[Quant Worker]
    W2[Ecom Worker]
    W3[Content Worker]
    W4[Media Worker<br/>ComfyUI/FFmpeg]
  end

  subgraph Guard[Guard & Audit]
    G1[Policy Engine]
    G2[Approval Gate]
    G3[Audit Logger]
  end

  subgraph Data[Data Plane]
    D1[(Postgres/SQLite)]
    D2[(Redis Queue)]
    D3[(Object Storage)]
  end

  Core -->|dispatch| W1
  Core -->|dispatch| W2
  Core -->|dispatch| W3
  Core -->|dispatch| W4

  Core --> Guard
  Guard -->|allow/deny/approve| Core

  Core --> Data
  Exec --> Data
  Guard --> Data
```

### 3.2 平面职责边界
- **控制平面（OpenClaw）**：决定“做什么、谁来做、何时做、失败怎么办”
- **执行平面（Workers / Tentacles）**：负责“具体怎么做”（脚本/模型/工具）
- **数据平面（DB/Queue/Storage）**：唯一事实源（SSOT），负责可追溯与重放

---

## 4. Docker 隔离与部署拓扑（Mandatory）

### 4.1 容器拆分建议
1) `orchestrator`：OpenClaw 中枢（无 GPU）  
2) `worker-quant`：量化触角（CPU优先，可选 GPU）  
3) `worker-ecom`：电商触角（账号操作必须走 guard）  
4) `worker-content`：社媒/文案（含合规检查）  
5) `worker-media`：ComfyUI/FFmpeg（GPU 重负载，独立容器）  
6) `guard`：策略引擎 + 审批流 + 审计写入  
7) `db`：Postgres（或前期 SQLite 过渡，但建议尽快 Postgres）  
8) `redis`：队列/缓存  
9) `minio`：对象存储（素材/报告/中间产物）

### 4.2 资源隔离要点（AMD 7900 XTX）
- GPU 容器与非 GPU 容器分离，避免互抢显存导致崩溃
- GPU 容器启用队列串行化或限并发
- 必须有降级路径：GPU 失败 → CPU / 云端模型

---

## 5. 任务模型与状态机（Task FSM）

### 5.1 统一 Task Schema（示例字段）
- `task_id`, `parent_task_id`
- `goal`：目标（业务结果）
- `constraints`：约束（成本/风险/时限/平台规则）
- `inputs`：数据引用（文件/表/链接/快照版本）
- `plan`：步骤列表（可更新）
- `actions`：可执行动作（必须可审计）
- `outputs`：产物引用（报告/图片/DB记录）
- `status`：`queued | running | waiting_approval | succeeded | failed | rolled_back`
- `risk_level`：`low | medium | high`
- `audit_ref`：审计链路指针（trace id）

### 5.2 状态转换规则（核心）
- 任何 `high risk` action → 强制 `waiting_approval`
- `failed` 必须写入失败原因与可恢复建议（retry / fallback / human takeover）
- `rolled_back` 必须写入回滚动作与结果

---

## 6. Tool Contract（工具调用协议）

每个工具（Python skill / API / CLI）必须声明：
- `name`, `version`
- `input_schema`（JSON Schema）
- `output_schema`
- `idempotent`（幂等性）
- `timeout_s`, `retry_policy`
- `side_effects`：是否有外部副作用（下单/改价/发帖）
- `risk_tag`：`read_only | write_draft | write_external | money_movement`
- `fallbacks`：失败时替代工具/降级策略

---

## 7. 大脑池（Brain Pool）与路由策略

### 7.1 分层策略（建议）
- **L0 本地优先**：日常任务、草稿、低风险动作（省钱 + 隐私）
- **L1 云端增强**：复杂推理、多模态、关键交付物
- **L2 双脑复核**：高风险/关键决策时，至少两模型一致或 guard 批准

### 7.2 自动评测（Eval）最小化
为常见任务建立小样本评测集：
- 交易复盘生成质量
- 电商上架文案与合规命中率
- 图片/视频流水线的成功率与耗时
结果用于路由与成本优化。

---

## 8. 安全审计官（Guard / Audit）

### 8.1 权限分级
- **Read**：仅读取数据
- **Suggest**：只产出建议/草稿
- **Execute-Soft**：执行但需人工确认（默认）
- **Execute-Hard**：全自动执行（仅白名单任务）

### 8.2 高风险动作清单（默认拦截）
- 资金/下单/提现/改价/投放预算变更
- 账号设置、批量删除、不可逆内容发布

### 8.3 审计日志（Append-only）
- 每个 action 记录：`who/when/what/why/evidence/result`
- 允许后期“重放”（replay）与“归因”（attribution）

---

## 9. 数据层：事件日志与指标口径（Event & Metrics）

### 9.1 Single Source of Truth（SSOT）
- `event_log`：系统中一切动作与结果的事实源（append-only）
- `task`：任务状态与引用
- `artifact`：产物索引（报告/图片/视频/模型输出）

### 9.2 指标（示例）
- 量化：PnL、胜率、最大回撤、滑点、执行偏差
- 电商：CTR、CVR、GMV、退款率、广告 ROI
- 内容：完播率、互动率、违规命中率、转化率

---

## 10. MVP：先跑通 1 条闭环（强烈建议）

### 10.1 MVP-A（推荐）：量化交易闭环
**数据采集 → 信号/风控 → 回测 → 建议 →（人工确认）→ 交易录入 → 盘后复盘**

交付物：
- `daily_report.md/html`
- `execution_report.csv`
- 可追溯到每一笔建议与执行差异的审计链路

### 10.2 MVP-B：电商单品闭环（备选）
**选品/定价 → 上架文案/主图 → 运营动作 → 数据回流 → 复盘**

---

## 11. 里程碑与验收标准

### P0（骨架可跑）
- 中枢 + guard + 数据层容器化
- 任务状态机可运行并可恢复（重启不丢任务）
- 事件日志可追溯

### P1（MVP 闭环跑通）
- 选择 MVP-A 或 MVP-B 跑通端到端
- 每个 action 都有审计记录
- 失败可降级/可人工接管

### P2（安全门禁完善）
- 高风险动作全部强制审批
- 双脑复核策略上线（L2）

### P3（第二触角插件化接入）
- 第二个触角接入同一 SSOT 与 guard，不另起炉灶

---

## 12. 风险清单与备用方案（Fallback Plan）

### 12.1 中枢故障
- 方案：无状态中枢 + 外置状态（DB）+ watchdog 重启

### 12.2 GPU 不稳定（ROCm/DirectML/显存）
- 方案：GPU Worker 限并发；失败降级到 CPU/云端；任务不中断

### 12.3 外部平台封禁/接口变化
- 方案：执行类动作走审批；提供“只产出建议/脚本”模式

### 12.4 数据损坏/误操作
- 方案：append-only event log + 定期备份 + 只读模式

---

## 13. 待决策项（Open Questions）
- 数据层：SQLite 过渡到 Postgres 的时间点与迁移脚本
- 队列：Redis Streams vs RabbitMQ（先 Redis 简化）
- Secrets 管理：docker secrets / vault
- 审批 UI：CLI → Web（Streamlit/Next.js）

---

## 附：下一步（推荐执行顺序）
1) 落地 Docker 拓扑与最小可运行骨架（P0）  
2) 选定 MVP-A（量化）并定义 Task/Tool Schema（P1）  
3) 上线 Guard 的审批流与高风险拦截（P2）  
4) 插件化接入第二触角（P3）
