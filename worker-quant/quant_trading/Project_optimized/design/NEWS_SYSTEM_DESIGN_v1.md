# Quant News Overlay System - Architecture Design

## 1. 核心定位 (System Identity)

News System 在量化架构中不是一个"读报器"，而是一个**高频风险门控（Risk Gating Layer）**和**非结构化 Alpha 因子源**。它的主要职责是将市场噪音降维成确定性的数值（Sentiment Score, Urgency），交由 `ss7_sqlite_news_overlay.py` 进行头寸缩减或做多确认。

## 2. 架构痛点与重构方向 (From Toy to Prod)

### 现状 (Intern Version)
- 同步抓取：交易主进程等待网络 IO。
- 脆弱落地：临时生成 `news.csv`，极易丢失，缺乏回测所需的点对点（Point-in-Time）快照。
- 逻辑揉杂：爬虫、NLP 处理、Nexus 工具注册全部堆在 `worker.py` 中。

### 重构目标 (Senior Quant Version)
- **解耦提取**：剥离爬虫逻辑为独立的 `news_ingester` 模块。
- **SQLite 持久化**：废弃 `news.csv` 软连接，新闻直接入库 `japan_market.db`，建立 `news_feed` 与 `news_sentiment` 表。
- **异步守护**：由 Cron 或 orchestrator 异步触发新闻入库，量化交易进程（`run_pipeline.py`）只做只读查询（Read-Only）。
- **统一 NLP 引擎**：规范情感打分机制（LLM-based 或词典 based），输出标准化的 `[-1.0, 1.0]` 极性分。
- **[新增] 流控与背压**：在 ingester → analyzer 链路中引入批处理队列与速率控制，防止突发灌入导致 LLM token 预算耗尽。
- **[新增] 事件级去重**：在 analyzer 阶段对同一事件的重复报道进行聚类，避免信号被人为放大。

## 3. 数据库设计 (Schema Update)

需要在 `japan_market.db` 中新增以下结构：

```sql
-- ================================================================
-- 原始新闻表：保证数据随时可追溯、可重新跑 NLP
-- ================================================================
CREATE TABLE news_feed (
    news_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    published_ts TEXT NOT NULL,       -- 新闻原始发布时间
    source TEXT NOT NULL,             -- 来源标识 (tdnet / google_news / jquants ...)
    title TEXT NOT NULL,
    content_summary TEXT,
    url TEXT,
    ingested_ts TEXT NOT NULL,        -- 系统实际抓取时间
    event_cluster_id TEXT,            -- [新增] 事件级去重聚类 ID (同一事件的不同报道共享此 ID)
    raw_hash TEXT                     -- [新增] content_summary 的 SHA-256，用于精确去重
);

-- 建议索引（加速 PIT 查询和去重）
CREATE INDEX idx_news_feed_symbol_ts ON news_feed(symbol, published_ts);
CREATE INDEX idx_news_feed_ingested ON news_feed(ingested_ts);
CREATE INDEX idx_news_feed_cluster ON news_feed(event_cluster_id);

-- ================================================================
-- NLP 情感打分表：与原始新闻解耦，支持模型迭代
-- ================================================================
CREATE TABLE news_sentiment (
    news_id TEXT,
    model_version TEXT,                -- 打分 LLM 版本 (如 qwen-2.5, gpt-4o)
    sentiment_score REAL,              -- [-1.0 到 1.0]
    urgency REAL,                      -- [0.0 到 1.0]
    expected_impact_days REAL,         -- [新增] 模型预估的影响持续天数，用于差异化衰减
    reason TEXT,                       -- [新增] LLM 给出的判定理由摘要
    scored_ts TEXT,                    -- [新增] 打分完成时间戳
    PRIMARY KEY (news_id, model_version)
);

-- ================================================================
-- [新增] 模型评估记录表：每次 model_version 上线前的 Golden Set 测试结果
-- ================================================================
CREATE TABLE sentiment_model_eval (
    model_version TEXT PRIMARY KEY,
    eval_date TEXT NOT NULL,
    golden_set_size INTEGER,           -- 测试集条数
    macro_f1 REAL,                     -- Macro-F1 得分
    mean_abs_drift REAL,               -- 与前版本的平均绝对漂移
    passed BOOLEAN,                    -- 是否通过上线门槛
    eval_detail_json TEXT              -- 详细分类报告 (JSON 存储)
);

-- ================================================================
-- [新增] 数据生命周期日志表
-- ================================================================
CREATE TABLE data_lifecycle_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,              -- ARCHIVE / PURGE / RESTORE
    table_name TEXT NOT NULL,
    date_range_start TEXT,
    date_range_end TEXT,
    row_count INTEGER,
    executed_ts TEXT NOT NULL,
    operator TEXT                      -- 执行人或自动任务标识
);
```

## 4. Ingester → Analyzer 流控设计 [新增章节]

### 4.1 问题背景

当 `news_ingester` 在短时间内灌入大量新闻（如财报季单次抓取 500+ 条），若 `news_analyzer` 逐条同步调用 LLM，会导致：
- LLM API token 预算在数分钟内耗尽；
- 打分延迟飙升，影响盘前信号生成的时效性。

### 4.2 解决方案：批处理 + 令牌桶

```
[news_ingester] --(写入)--> news_feed 表
                                |
                         (Analyzer 轮询)
                                |
                    +-----------+-----------+
                    |  Batch Scheduler      |
                    |  - 每次拉取 ≤ 50 条   |
                    |  - 令牌桶: ≤ 30 req/min|
                    |  - 优先级: 按 symbol   |
                    |    持仓权重排序        |
                    +-----------+-----------+
                                |
                         (调用 LLM API)
                                |
                    [写入 news_sentiment]
```

- **批次大小**：每次从 `news_feed` 中拉取 `scored_ts IS NULL` 的未打分条目，上限 50 条。
- **速率控制**：令牌桶算法，默认 ≤ 30 requests/min（可通过 `config.yaml` 调整）。
- **优先级排序**：当前持仓股票的新闻优先打分（需从 `portfolio_holdings` 表读取权重）；非持仓股票排在后面。
- **超时保护**：单条新闻 LLM 打分超时 30 秒后自动跳过，标记 `scored_ts = 'TIMEOUT'`，下一轮重试。

## 5. 模块交互流 (Data Flow)

```
                   ┌──────────────────────────────┐
                   │    Cron / Orchestrator        │
                   │  (每 15 min 或盘前触发)       │
                   └──────────────┬───────────────┘
                                  │ 触发
                                  ▼
               ┌──────────────────────────────────┐
               │  news_ingester.py                 │
               │  - 抓取 TDnet / Google News       │
               │  - 写入 news_feed                 │
               │  - raw_hash 精确去重              │
               │  - 指数退避重试 (max 3 retries)   │
               └──────────────┬───────────────────┘
                              │ INSERT
                              ▼
               ┌──────────────────────────────────┐
               │  news_feed (SQLite)               │
               └──────────────┬───────────────────┘
                              │ Batch Poll
                              ▼
               ┌──────────────────────────────────┐
               │  news_analyzer.py                 │
               │  - 事件聚类 → event_cluster_id    │
               │  - LLM 打分 (批处理 + 令牌桶)     │
               │  - 结果写入 news_sentiment        │
               └──────────────┬───────────────────┘
                              │ SELECT (Read-Only)
                              ▼
               ┌──────────────────────────────────┐
               │  ss7_sqlite_news_overlay.py       │
               │  - PIT 过滤:                      │
               │    published_ts < order_time       │
               │    AND ingested_ts < order_time    │
               │  - 同一 cluster 只取一条得分       │
               │  - 差异化衰减 (expected_impact_days)│
               │  - 输出 sentiment_gate, vol_z      │
               └──────────────────────────────────┘
```

**关键改动说明**：
1. `ss7_sqlite_news_overlay.py` 完全废弃 `news.csv`，改为 SQLite 只读查询。
2. PIT 过滤同时约束 `published_ts` 和 `ingested_ts`，与 Governance 规则 1.2 / 1.3 对齐。
3. 同一 `event_cluster_id` 下仅取最早入库的一条得分（或中位数），防止重复报道放大信号。
4. 衰减算法使用每条新闻自身的 `expected_impact_days` 字段，而非全局固定半衰期。

## 6. 数据保留策略 [新增章节]

| 数据层级 | 保留期限 | 操作 |
|---------|---------|------|
| `news_feed` + `news_sentiment` 主表 | 最近 2 年 | 在线查询 |
| 归档表 (`*_archive`) | 2-5 年 | 低频回测查询 |
| 冷存储 (Parquet 导出) | 5 年以上 | 合规审计留存 |

- 归档任务由 `data_archiver.py` 按季度执行，操作记录写入 `data_lifecycle_log`。
- 归档前自动检查是否有活跃回测任务引用该时段数据。
