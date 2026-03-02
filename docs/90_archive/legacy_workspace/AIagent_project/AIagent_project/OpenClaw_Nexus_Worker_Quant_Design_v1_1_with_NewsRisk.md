# OpenClaw Nexus: Worker-Quant（量化智能体）设计文档（更新版）
> 版本：v1.1（在原 5-Step Pipeline 基础上增强“可落地契约 + 新闻风险因子”）  
> 目标读者：Orchestrator / Worker-Quant 开发者、量化策略开发者、最终审批人（人类决策者）

---

## 1. 业务目标与架构定位

### 1.1 业务目标
实现从 **宏观/微观情报收集 → 数据清洗与证据溯源 → 因子构建与筛选 → 回测评估 → 投资建议生成** 的全链路自动化。系统定位为“超级研究员”，提供结构化数据与建议单，不直接执行交易。

### 1.2 风险控制核心（Human-in-the-Loop）
- 系统仅生成 **propose_orders（建议决策）**。
- 任何交易执行（approve_orders / record_fills）必须通过 Orchestrator 的 **Guard** 模块进入 `waiting_approval` 状态，**决策权 100% 交由人类决策者**（通过 Discord 交互）。

### 1.3 在 Nexus 架构中的定位
- **身份**：隶属于 `worker-quant` 命名空间的一系列组合 Skills（可版本化、可禁用、可插入审批 Gate）。  
- **数据来源约束**：受 Facts Store Schema 与 Source Hierarchy 约束，所有采集数据必须形成 `evidence -> fact_item -> link` 闭环，严禁模型“凭空生成”财务指标或事件事实。  
- **交互方式**：通过 Discord 接收自然语言指令（NLP 意图映射），并将最终 Markdown 报告与图表资产（MinIO）推送回 Discord 频道。

---

## 2. 核心原则（不可破）

1) **事实即证据，决策必溯源**  
所有可影响因子、策略或建议的输入，必须来自 `evidence -> fact_item -> link` 闭环。Writer/Strategy 不得直接“幻觉”出财务指标或事件结论。

2) **Source Hierarchy 仲裁**  
- 财务事实（财报、公告、交易所数据）优先采用 L1/L2。  
- 舆情/评论（L4）仅能进入“情绪/风险因子”，不得作为财务事实。

3) **人类拥有最终执行权**  
任何可能导致下单/资金动作的流程必须经过 Guard：`waiting_approval`。

4) **执行有预算（Cost Ledger）**  
每次 run 输出预算与消耗：tokens、页面访问数、失败域、重试次数、wall time 等。

5) **可回放（Replay Unit）**  
每次 run 产出可复现单元：plan/events/facts/narrative/report，支持事后审计与回放。

---

## 3. 数据面：Facts Store 与 Feature Store（新增）

### 3.1 Facts Store（不可变、可溯源）
- `evidence`：网页截图 hash、原文/提取文本、来源 URL、抓取时间等
- `fact_item`：从 evidence 抽取的结构化事实或中间结果（带 kind）
- `link`：fact_item 与 evidence 的强绑定（溯源链路）

> 备注：事实仓是“审计层”。任何后续计算都必须可追溯回事实仓。

### 3.2 Feature Store（可回测、日频对齐、面向策略）
为支持“每日因子快照”与回测复现，引入一个薄 Feature Store（MVP 可先用 SQLite 表实现）：

**建议新增表：`feature_daily`**
- `asof`：YYYY-MM-DD（日频对齐）
- `symbol`：例如 `9432.T`
- `feature_name`：例如 `value_pe`、`quality_roe`、`news_risk_z`
- `value`：float
- `source_fact_ids`：JSON 数组，引用产生该 feature 的 fact_item ids
- `created_at`：写入时间

> 解释：feature 是“策略可直接使用的数值”，但必须能追溯回 fact_item，避免“算出来就丢证据”。

---

## 4. 核心工作流设计（The 5-Step Pipeline → 可执行 DAG）

### Step 1：自动情报收集（Macro & Industry Intelligence）
**目标**：收集宏观经济数据、行业政策、权威媒体情绪与事件，为风险预算与行业偏好提供上下文。  
**输入**：白名单源（L1/L2 优先）。  
**输出**：
- `evidence`：网页截图/公告 PDF hash（可校验）
- `fact_item`：例如 `macro_indicator` / `policy_event` / `rates_snapshot`
- `link`：fact → evidence

**验收条件**：
- evidence.hash 可验证
- 每次 run 至少生成 N 条 fact_item（避免空跑）

---

### Step 2：股票情报收集（Micro & Ticker Intelligence）
**目标**：收集特定股票池的财务报表、公告事件、分析师评级变化等。  
**约束**：财务事实必须遵循 Source Hierarchy：EDINET/SEC/交易所等 L1 数据优先。  
**输出**：
- `fact_item`：例如 `fundamental_snapshot` / `earnings_event` / `rating_change`
- 相关 evidence + link 完整闭环

---

### Step 3：筛选与因子计算（Screening & Feature Computation）【重点升级】
该步骤拆解为三层：

#### Step 3A：基础因子生成（传统）
- **Value**：PE/PB/EV-EBITDA 等（以可获取数据为准）
- **Quality**：ROE、毛利率、利润率
- **Momentum**：1/3/6/12M return、volatility
- **Liquidity**：成交额、换手率

**输出**：写入 `feature_daily`（带 `source_fact_ids`）。

#### Step 3B：新闻风险因子（News Risk Factor）【新增】
目标：把“每天定时收集新闻”变成可回测、可审计的风险因子，并用于**修正仓位/风险预算**（从而影响策略曲线）。

**(1) News Ingest → fact_item（可溯源）**
- 每条新闻生成 `evidence`（URL + 截图/原文 text hash）
- 生成 `fact_item(kind="news_item")`：
  - `symbols`：命中标的/行业（可多对多）
  - `publisher`、`published_at`
  - `title`、`summary`（可由模型生成，但必须 link 到 evidence）
  - `tags`：监管/诉讼/财报/并购/事故/召回等

**(2) News Scoring → risk 子因子（结构化、可解释）**
新增 skill：`compute_news_risk_factor`，对每个 symbol 每日计算：
- `news_sentiment`：[-1, +1]（情绪）
- `news_uncertainty`：[0, +∞)（冲击强度/不确定性）
- `news_event_severity`：[0, 1]（事件严重度）
- `news_source_weight`：按 L1/L2/L3/L4 权重（L4 仅影响情绪，不影响财务事实）

汇总为主因子：
- `news_risk_raw = w1*max(0, -sentiment) + w2*uncertainty + w3*severity`
- `news_risk_z = zscore(news_risk_raw, rolling_window=252)`

写入：`feature_daily(asof, symbol, "news_risk_z")`。

**(3) 如何“每天修正曲线”（推荐实现顺序）**
> 工程定义：不是对 equity curve 图形硬改，而是用 news risk 改变仓位/风险预算，从而使回测与实盘曲线真实变化。

1. **仓位缩放（最推荐、最稳）**  
   - `position_scale = clamp(1 - k * sigmoid(news_risk_z), min=0.2, max=1.0)`  
   - 交易信号不变，仓位按风险收缩。
2. **风险预算/止损参数自适应（进阶）**  
   - `target_vol = base_vol / (1 + a*news_risk_z)`
3. **收益曲线后处理（不推荐）**  
   - 会破坏可解释性与审计性，仅用于压力测试可选。

#### Step 3C：筛选（Screening）
- **规则引擎（MVP）**：  
  - `quality_roe > 0.15` AND `value_pe < 20` AND `news_risk_z < threshold`
- **打分模型（后续）**：  
  - `score = α*value + β*quality + γ*momentum - λ*news_risk_z`

输出：`candidate_tickers.json`（MinIO + DB 引用）。

---

### Step 4：模型回测（Backtesting）
**目标**：对候选标的运行历史回测，验证胜率、盈亏比、最大回撤等，并产出图表与统计。  
**输入**：候选标的 + `feature_daily`（含 news risk）+ 历史价格数据  
**输出**：
- 图表资产：收益率曲线、回撤曲线等（MinIO object_key）
- `fact_item(kind="backtest_result")`：参数、统计、曲线摘要
- `backtest_report.md`：引用 facts，不可虚构

**关键要求**：回测必须记录当日使用的 feature snapshot（asof 对齐），确保复现。

---

### Step 5：投资建议生成（Proposal & Human-in-the-Loop）
**目标**：综合前四步，生成带推理过程的建议报告，阻断自动交易，等待人类审批。  
**输出**：
- `proposal_report.md`：包含溯源链接（Link to Evidence）
- `propose_orders`：建议单（不执行）
- 状态进入：`waiting_approval`

**审批与后续**：
- Discord 按钮 Approve/Reject  
- Approve：写入 event_log，可对接模拟盘/实盘的 record_fills  
- Reject：任务终止、资源释放

---

## 5. Workflow Runner（DAG 视角：像 n8n，但 SSOT 在 OpenClaw）

建议将量化闭环固化为 workflow spec（YAML/JSON），支持节点级重跑与审批 Gate。MVP 推荐 DAG：

1. `fetch_prices_daily`  
2. `fetch_news_daily`  
3. `compute_news_risk_factor`  
4. `compute_features_daily`（传统因子）  
5. `screen_candidates`  
6. `backtest_candidates`（可选开关）  
7. `generate_proposal`（Guard Gate）  
8. `publish_discord_report`

---

## 6. Worker-Quant 技能注册表（Skill Registry）

在原技能表基础上新增：

- `skill_quant_fetch_news`（v1.0, 风险: Low）  
  - 抓取新闻、生成 evidence + fact_item(news_item)

- `skill_quant_compute_news_risk`（v1.0, 风险: Medium）  
  - NLP 分类/打分 → 写 `feature_daily(news_risk_z)`  
  - 建议输出预算：tokens 上限、单日最大新闻条数、最大域名访问数（写入 Cost Ledger）

保留原有：
- `skill_quant_fetch_macro`（v1.0, 风险: Low）
- `skill_quant_fetch_fundamentals`（v1.0, 风险: Low）
- `skill_quant_multi_factor_screener`（v1.0, 风险: Low）
- `skill_quant_backtrader_engine`（v1.0, 风险: Medium）
- `skill_quant_generate_proposal`（v1.0, 风险: High，触发 Guard）

---

## 7. 差距分析与实施建议（Gap Analysis）

### 7.1 当前 Nexus 已实现/已具备
- 基础设施：任务队列（Redis Streams）、状态机（queued/running/failed/waiting_approval）、Guard 风控门禁、MinIO 资产存储概念。  
- 架构契约：防幻觉的 Facts Store 与证据溯源机制（evidence -> fact_item）。  
- 交互方式：Discord 作为 Headless UI 接收指令与发送报告。

### 7.2 最短闭环落地顺序（推荐）
1) **Feature Store（`feature_daily`）表结构 + 写入接口**（SQLite 即可）  
2) **新闻 → evidence/fact_item/link 闭环**（证明因子输入可审计）  
3) **compute_news_risk_factor**（先做可解释的简化版，逐步迭代）  
4) **在策略里用 news risk 做仓位缩放**（最稳的“曲线修正”实现）  
5) 接入 DAG Runner：产出 Replay Unit + Cost Ledger，形成可观测与可复盘闭环

---

## 8. 附录：新闻风险因子（NRF）接口契约（建议）

### 8.1 输入
- `news_item` fact_items（带 symbols、publisher、published_at、tags、summary）
- Source Hierarchy 权重配置：`L1/L2/L3/L4 -> weight`

### 8.2 输出（写入 feature_daily）
- `news_sentiment`
- `news_uncertainty`
- `news_event_severity`
- `news_risk_raw`
- `news_risk_z`

### 8.3 使用（策略侧）
- `position_scale(symbol, asof)`：由 `news_risk_z` 映射得到
- 策略执行：`effective_position = base_position * position_scale`

---

**End.**
