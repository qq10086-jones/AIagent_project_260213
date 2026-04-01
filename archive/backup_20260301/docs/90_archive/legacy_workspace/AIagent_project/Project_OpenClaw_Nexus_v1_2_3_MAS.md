# OpenClaw Nexus v1.2.3 - Multi-Agent System (MAS) Engineering Blueprint

> **核心哲学**：事实即证据，决策必溯源，执行有预算。
> **演进方针**：建立强契约的事实仓、层级化的权威源仲裁、以及标准化的任务回放机制。

---

## 1. 核心存储契约：Facts Store Schema

为了确保 Writer Agent 严禁虚构事实，Facts Store 强制执行以下三层拓扑结构：

- **fact_item** (事实条目): `{fact_id, run_id, agent_name, kind, payload, created_at}`
- **evidence** (原始证据): `{evidence_id, url, captured_at, screenshot_ref, extracted_text, hash}`
- **link** (溯源关联): `{fact_id -> evidence_id}`

**强制约束**：Writer 只能读取 `fact_item`，且 Supervisor 必须在 `link` 存在并验证 `evidence.hash` 后，才允许 Writer 引用该条事实。

---

## 2. 执行治理与成本账本 (Execution & Cost Ledger)

### 2.1 成本账本 (Cost Ledger)
每次 `run` 结束时，Supervisor 必须输出标准化“账单”，用于性能评估与防跑飞：
- `tokens_total` / `tokens_by_agent`
- `browser_pages_visited` / `domain_failures`
- `retries_count` / `fallback_triggered`
- `wall_time_ms`

### 2.2 权威源优先级表 (Source Hierarchy)
在处理冲突或提取事实时，系统遵循以下优先级，低级来源不得覆盖高级来源：
1. **L1: 官方数据** (交易所公告 / 官方 IR / EDINET)
2. **L2: 主流媒体** (白名单内的财经通讯社，如日经、路透)
3. **L3: 聚合/三手信息** (一般性财经网站)
4. **L4: 社交舆情** (雪球、Yahoo评论 - **仅限作为情绪指标，严禁作为财务事实**)

---

## 3. 数据完整性：可验证证据 (Verifiable Evidence)

### 3.1 证据包结构
Browser Agent 提交的 `evidence_package` 必须包含可验证的引用：
- `screenshot_ref`: `{run_id, step_id, object_key, sha256}`
- **验证逻辑**：Supervisor 定期抽检 `extracted_text` 的哈希值是否与 `screenshot_ref` 对应页面的 OCR 或 DOM 快照一致。

---

## 4. 标准化任务回放包 (Replay Unit)

为了实现 **Phase 0** 的观测目标，每次任务执行后需生成一个标准化的可回放归档（Zip/Directory）：
- `plan.json`: Supervisor 生成的初始与修正后的 DAG 计划。
- `events.jsonl`: 包含所有的 Delegation, Tool Calls, 反思日志 (Reflection) 和错误记录。
- `facts/`: 本次任务生成的 Immutable Facts。
- `narrative/`: Writer 生成的所有草稿版本。
- `report.md`: 最终发往 Discord 的成品。

---

## 5. 接口与通信协议 (API Standards)

所有对外接口（Orchestrator -> Brain, Brain -> Workers）强制要求：
- `run_id`: 全链路追踪唯一标识。
- `idempotency_key`: 防止重试导致的重复抓取或重复下单。
- `client_msg_id`: Discord 端的原始消息追踪。

---

## 6. 实施路线图 (Updated Roadmap)

### Phase 0: 观测、回放与幂等化
- 实现 `run_id` 透传逻辑。
- 建立 `Facts Store` 最小 Schema。
- 在 Orchestrator 接口层实施 `idempotency_key` 检查。

### Phase 1: 脑部重构 (LangGraph Integration)
- 基于 LangGraph 编写 Supervisor。
- 实施 Facts vs Narrative 物理隔离。

### Phase 2: 调查员 Agent 闭环
- 实现证据包（Evidence Package）校验逻辑。
- 引入权威源优先级仲裁。
