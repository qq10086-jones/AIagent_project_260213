# Quant News Overlay System - Architecture Design

## 1. 核心定位 (System Identity)
News System 在量化架构中不是一个“读报器”，而是一个**高频风险门控（Risk Gating Layer）**和**非结构化 Alpha 因子源**。它的主要职责是将市场噪音降维成确定性的数值（Sentiment Score, Urgency），交由 `ss7_sqlite_news_overlay.py` 进行头寸缩减或做多确认。

## 2. 架构痛点与重构方向 (From Toy to Prod)

### 现状 (Intern Version)
- 同步抓取：交易主进程等待网络 IO。
- 脆弱落地：临时生成 `news.csv`，极易丢失，缺乏回测所需的点对点（Point-in-Time）快照。
- 逻辑揉杂：爬虫、NLP 处理、Nexus 工具注册全部堆在 `worker.py` 中。

### 重构目标 (Senior Quant Version)
- **解耦提取**：剥离爬虫逻辑为独立的 `news_ingester` 模块。
- **SQLite 持久化**：废弃 `news.csv` 软连接，新闻直接入库 `japan_market.db`，建立 `news_feed` 与 `news_sentiment` 表。
- **异步守护**：由 Cron 或 orchestrator 异步触发新闻入库，量化交易进程（`run_pipeline.py`）只做只读查询（Read-Only）。
- **统一 NLP 引擎**：规范情感打分机制（LLM-based或词典based），输出标准化的 `[-1.0, 1.0]` 极性分。

## 3. 数据库设计 (Schema Update)

需要在 `japan_market.db` 中新增以下结构：

```sql
-- 原始新闻表：保证数据随时可追溯、可重新跑NLP
CREATE TABLE news_feed (
    news_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    published_ts TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    content_summary TEXT,
    url TEXT,
    ingested_ts TEXT NOT NULL
);

-- NLP 情感打分表：与原始新闻解耦，支持模型迭代
CREATE TABLE news_sentiment (
    news_id TEXT,
    model_version TEXT, -- 记录打分的LLM版本（如 qwen-2.5, GPT-4）
    sentiment_score REAL, -- [-1.0 到 1.0]
    urgency REAL,         -- [0.0 到 1.0]
    PRIMARY KEY (news_id, model_version)
);
```

## 4. 模块交互流 (Data Flow)

1. **Ingestion (Worker / Cron)**: 
   `worker-quant/news_ingester.py` 定时访问 TDnet / Google News，将原始内容写入 `news_feed`。
2. **Analysis (LLM pipeline)**:
   `worker-quant/news_analyzer.py` 监听 `news_feed` 新增条目，调用 LLM 进行结构化打分，结果写入 `news_sentiment`。
3. **Consumption (Quant Backtest/Live)**:
   `ss7_sqlite_news_overlay.py` 废弃读取 `news.csv`，改为直接 `SELECT` 对应交易日之前（避免未来函数）的新闻情感均值，生成 `vol_z` 和 `sentiment_gate`。