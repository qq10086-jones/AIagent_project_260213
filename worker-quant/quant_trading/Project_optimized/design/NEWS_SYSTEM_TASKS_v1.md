# News System Optimization Task List

> **变更说明**：本版本根据 PM/Quant 评审意见进行了以下调整：
> - P0 新增去重与 Schema 补全任务
> - P1 修正 PIT 约束描述（与 Governance 对齐）、新增流控与 Dashboard 任务
> - P2 新增 Golden Set 质量基线、差异化衰减
> - P3 新增数据保留策略

## P0: 核心重构与持久化 (Foundation & Persistence)

- [x] **DB Schema 升级**: 在 `trade_schema.py` 和 `db_update.py` 中引入以下表和字段：
  - `news_feed` 表（含 `event_cluster_id`, `raw_hash` 新增字段）
  - `news_sentiment` 表（含 `expected_impact_days`, `reason`, `scored_ts` 新增字段）
  - `sentiment_model_eval` 表（模型上线评估记录）
  - `data_lifecycle_log` 表（归档操作审计）
  - 相关索引（`idx_news_feed_symbol_ts`, `idx_news_feed_ingested`, `idx_news_feed_cluster`）
- [ ] **拆分 Worker**: 从臃肿的 `worker.py` 中剥离新闻抓取代码，建立 `news_ingester.py` 作为独立脚本。
- [ ] **废弃 CSV 流程**: 修改 `ss7_sqlite_news_overlay.py` 中 `load_news_items()` 函数，使其直接从 SQLite 查询新闻，而不是读取易碎的 `.csv` 文件。
- [ ] **[新增] 精确去重**: 在 `news_ingester.py` 写入时，基于 `raw_hash`（content_summary 的 SHA-256）进行精确去重，跳过完全重复的内容。
- [ ] **[新增] 事件级去重**: 在 `news_analyzer.py` 中实现基于 TF-IDF 余弦相似度的事件聚类，生成 `event_cluster_id`，写回 `news_feed` 表。相似度阈值 > 0.85，时间窗口 ±4 小时，按 `symbol` 分组。

## P1: 健壮性、防未来函数与可观测性 (Robustness, PIT & Observability)

- [ ] **Point-in-Time 查询 [修正]**: ~~确保量化模型读取新闻时，严格过滤 `published_ts < market_open_ts`~~
  - **修正**：与 Governance 规则 1.2 / 1.3 对齐，改为同时过滤 `published_ts < order_time` **且** `ingested_ts < order_time`。截止时间使用 `order_time`（实际下单时刻），而非固定的 `market_open_ts`。
  - 上线前须由 QA 使用已知 look-ahead case 进行回归验证。
- [ ] **引入权威数据源**: 减少对 Google News 的依赖，集成日本市场官方的 TDnet RSS 或 J-Quants 资讯 API 接口。
- [ ] **异常重试机制**: 在 `news_ingester.py` 中加入指数退避（Exponential Backoff）重试机制，最大重试 3 次，防止网络抖动导致当天断流。
- [ ] **[新增] Ingester → Analyzer 流控**: 实现批处理调度器：
  - 每次拉取未打分条目上限 50 条
  - 令牌桶速率控制 ≤ 30 req/min
  - 当前持仓股票新闻优先打分
  - 单条超时 30s 自动跳过并标记 `TIMEOUT`
- [ ] **[新增] 动态断流报警阈值**: 将降级报警从静态"30日均值的20%"改为基于 90 个交易日分布的 P5 百分位动态阈值（冷启动期间回退至 30 日均值的 30%）。
- [ ] **[新增 — 优先级提升] News Monitor Dashboard**: 在 `app.py` (Streamlit 看板) 中增加 "News Monitor" 标签页。最小可用版本须包含：
  - 当日各 symbol 新闻抓取数量 & 情感得分分布
  - 断流/降级状态指示灯
  - 极端得分（|score| > 0.8）的新闻列表及 LLM 判定理由
  - **理由**：PM 在系统上线初期需要实时观测新闻信号状态，对建立信任和快速发现问题至关重要，因此从 P3 提升至 P1。

## P2: NLP 与情感引擎 (Sentiment Engine)

- [ ] **统一打分 Prompt**: 设计标准的 LLM Prompt，要求强制输出 JSON 格式：
  ```json
  {
    "sentiment": 0.8,
    "urgency": 0.9,
    "expected_impact_days": 3,
    "reason": "..."
  }
  ```
  新增 `expected_impact_days` 字段，让模型预估该新闻影响的持续时间，用于差异化衰减。
- [ ] **[新增] Golden Set 质量基线**: 
  - 建立不少于 200 条人工标注的日语财经新闻 Golden Set（正面/负面/中性均衡分布）。
  - 每次 `model_version` 上线前，在 Golden Set 上运行回归测试，要求 Macro-F1 ≥ 0.75 且 Mean Absolute Drift ≤ 0.10。
  - 测试结果写入 `sentiment_model_eval` 表；未通过的模型版本禁止写入生产 `news_sentiment`。
- [ ] **多模型回测支持**: 支持对同一条历史新闻使用不同版本的 LLM (`model_version`) 进行重复打分对比，寻找最稳定有效的分析模型。
- [ ] **[修改] 差异化衰减算法**: ~~实现新闻冲击力的时间衰减（半衰期机制，例如重大财报影响在 3 天内按指数衰减）~~
  - **修改**：废弃全局固定半衰期。改用每条新闻自身的 `expected_impact_days` 作为半衰期参数进行指数衰减：`decay = exp(-ln2 * Δt / expected_impact_days)`。
  - 若 `expected_impact_days` 缺失（旧数据兼容），回退至默认值 3 天。

## P3: Nexus 系统集成与数据治理 (Nexus & Data Lifecycle)

- [ ] **Worker Contract 对齐**: 在 `NEXUS_WORKER_CONTRACT.md` 中更新 `news.*` 系列工具的输入输出契约，包含新增的 `event_cluster_id`、`expected_impact_days` 等字段。
- [ ] **[新增] 数据保留策略实施**:
  - 开发 `data_archiver.py`，按季度将超过 2 年的 `news_feed` 和 `news_sentiment` 数据归档至 `*_archive` 表或导出为 Parquet。
  - 归档前自动检查是否有活跃回测任务引用目标时段。
  - 所有归档操作写入 `data_lifecycle_log`。
- [ ] **[新增] robots.txt 合规检查**: 在 `news_ingester.py` 中集成 `robotparser`，自动跳过禁止爬取的路径并记录日志。
