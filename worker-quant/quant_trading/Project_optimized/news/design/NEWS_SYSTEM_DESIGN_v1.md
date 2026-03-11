# News Worker — 系统设计文档 v1.1

更新日期：2026-03-11

---

## 1. 工具定位

News Worker 是一个**独立的自动化新闻收集与分析工具**，运行于量化交易系统的上游。

它的职责边界：
- **收集**：定期从外部数据源抓取日本股票相关新闻，持久化存储
- **分析**：对原始新闻进行事件去重和情感打分
- **提供**：向量化模型（ss7）提供只读的情感信号，ss7 决定如何使用

它**不负责**：
- 量化因子构建（由 ss7 内部完成）
- 交易决策（由 make_decision.py 完成）
- Nexus 系统集成（后续 P3 工作）

---

## 2. 模块结构

```
news/
├── news_ingester.py       收集层：抓取新闻 → 写入 news_feed 表
├── news_analyzer.py       分析层：事件聚类 + LLM 打分 → 写入 news_sentiment 表
├── gen_synthetic_news.py  测试工具：生成合成新闻 CSV 用于调试
└── design/
    ├── NEWS_SYSTEM_DESIGN_v1.md  (本文件)
    ├── NEWS_GOVERNANCE_v1.md
    └── NEWS_SYSTEM_TASKS_v1.md
```

---

## 3. 数据流

```
[外部数据源]
Google News RSS / TDnet RDF / J-Quants API
        │
        ▼
news_ingester.py
  - 按 symbol 列表抓取
  - raw_hash 精确去重（SHA-256）
  - 写入 news_feed 表
  - 指数退避重试（最大 3 次）
        │
        ▼
news_feed 表（japan_market.db）
  - news_id, symbol, title, url
  - published_ts, ingested_ts
  - raw_hash, event_cluster_id
        │
        ▼
news_analyzer.py
  - TF-IDF 事件聚类 → 写回 event_cluster_id
  - LLM 打分（批处理 ≤ 50 条，≤ 30 req/min）
  - 写入 news_sentiment 表
        │
        ▼
news_sentiment 表
  - sentiment_score [-1, 1]
  - urgency [0, 1]
  - expected_impact_days
  - reason（LLM 判定摘要）
        │
        ▼（只读查询，ss7 内部处理）
ss7_sqlite_news_overlay.py
  - PIT 过滤
  - cluster 去重
  - 情感门控信号输出
```

---

## 4. 数据库 Schema

```sql
CREATE TABLE news_feed (
    news_id          TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    published_ts     TEXT NOT NULL,   -- 新闻原始发布时间 (ISO-8601 UTC)
    source           TEXT NOT NULL,   -- google_news_rss / tdnet / ...
    title            TEXT NOT NULL,
    content_summary  TEXT,
    url              TEXT,
    ingested_ts      TEXT NOT NULL,   -- 系统实际抓取时间 (ISO-8601 UTC)
    event_cluster_id TEXT,            -- 事件聚类 ID
    raw_hash         TEXT             -- SHA-256(title+url)
);

CREATE TABLE news_sentiment (
    news_id              TEXT,
    model_version        TEXT,
    sentiment_score      REAL,        -- [-1.0, 1.0]
    urgency              REAL,        -- [0.0, 1.0]
    expected_impact_days REAL,
    reason               TEXT,
    scored_ts            TEXT,        -- 'TIMEOUT' 表示超时跳过
    PRIMARY KEY (news_id, model_version)
);

CREATE TABLE sentiment_model_eval (
    model_version    TEXT PRIMARY KEY,
    eval_date        TEXT NOT NULL,
    golden_set_size  INTEGER,
    macro_f1         REAL,
    mean_abs_drift   REAL,
    passed           BOOLEAN,
    eval_detail_json TEXT
);

CREATE TABLE data_lifecycle_log (
    log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    action           TEXT NOT NULL,
    table_name       TEXT NOT NULL,
    date_range_start TEXT,
    date_range_end   TEXT,
    row_count        INTEGER,
    executed_ts      TEXT NOT NULL,
    operator         TEXT
);
```

---

## 5. Ingester → Analyzer 流控

收集和打分解耦运行，通过数据库状态衔接：

- Batch 大小：每次拉取未打分条目上限 50 条
- 速率控制：令牌桶算法，≤ 30 req/min
- 优先级：当前持仓 symbol 优先打分
- 超时保护：单条超时 30s 自动标记 `TIMEOUT`，下轮重试

---

## 6. 运行方式

```bash
# 从 Project_optimized/ 根目录执行

# 收集新闻（所有 daily_prices 里的 symbol）
python news/news_ingester.py --db japan_market.db

# 收集指定 symbol
python news/news_ingester.py --db japan_market.db --symbols 9432.T,9433.T,7203.T

# LLM 打分（需要 ANTHROPIC_API_KEY）
ANTHROPIC_API_KEY=sk-... python news/news_analyzer.py --db japan_market.db

# 只做聚类不调 LLM（调试用）
python news/news_analyzer.py --db japan_market.db --dry-run
```

---

## 7. 数据保留策略

| 层级 | 数据 | 保留期 |
|------|------|--------|
| 主表在线 | news_feed + news_sentiment | 最近 2 年 |
| 归档表 | *_archive | 2–5 年 |
| 冷存储 | Parquet 导出 | 5 年以上 |

归档由 `news/data_archiver.py`（待建）按季度执行。
