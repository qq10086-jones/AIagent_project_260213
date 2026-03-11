# News Data Governance & Compliance (新闻数据治理规范)

作为量化系统中的外部非结构化数据源，新闻数据极易引入系统性偏差和合规风险。所有针对 News Pipeline 的开发和维护必须遵守以下治理规范。

## 1. 消除未来函数 (Zero Look-ahead Bias)

**红线原则**：在任何历史回测（Backtesting）和特征工程中，严禁模型"看到"在交易时刻尚未发布的新闻。

* **规则 1.1**：所有写入 `news_feed` 的记录必须同时包含 `published_ts`（新闻实际发生时间）和 `ingested_ts`（系统抓取到它的时间）。
* **规则 1.2**：回测时，SQL 查询必须同时约束 `published_ts < order_time` **且** `ingested_ts < order_time`，以系统真实接收时间为准，防止利用事后修改/补发的新闻。
* **规则 1.3 [新增]**：截止时间点统一使用 `order_time`（实际下单时刻），**而非** `market_open_ts`。原因：若策略支持盘中交易，使用固定的开盘时间作为截止点会导致盘中新闻信号完全失效。仅在策略明确为"开盘一次性下单"模式时，方可用 `market_open_ts` 作为 `order_time` 的等价替代。
* **规则 1.4 [新增]**：所有涉及 PIT 约束的 Task（如 P1 任务）在实施前，须与本节规则逐条对齐。**禁止** Task 描述中出现与 Governance 不一致的截止时间定义。上线前由 QA 逐条交叉检查。

## 2. 缺失与降级处理 (Fail-closed & Degradation)

新闻网络 API 极易中断（Rate limit, 网站改版）。

* **规则 2.1 (静默降级)**：如果今天某只股票无法获取新闻，其当日情感得分应归一化为中性（`0.0`）或继承上一日衰减值，绝不能抛出异常阻断当天的量化选股（Screener）和下单逻辑。
* **规则 2.2 (大面积断流报警 — 动态阈值)**：~~如果某天全市场抓取到的新闻条数低于 30 日均值的 20%~~
  - **[修订]**：静态阈值（20%）缺乏统计依据，改为**动态百分位阈值**：取过去 90 个交易日抓取量的分布，若当日抓取量低于 **P5**（第 5 百分位），系统发送 `DATA_DEGRADED` 警告。
  - 首次上线时，若历史数据不足 90 天，可临时回退至"低于 30 日均值的 30%"作为冷启动阈值，并在日志中标注 `COLD_START_THRESHOLD`。
  - 报警发送至 Orchestrator，交由 PM 判断是否暂停当日实盘。

## 3. 合规与抓取道德 (Scraping Ethics)

对于外部网站的数据采集必须遵守商业道德与当地法律。

* **规则 3.1**：所有爬虫请求必须带有明确的 `User-Agent`，标明为内部量化机器人的识别名。
* **规则 3.2**：对于非官方 API 的网页抓取，请求频率（Rate Limit）必须限制在 `≤ 1 req/sec`，防止对目标服务器造成 DDoS。优先使用官方付费 API（如 Nikkei, J-Quants）替代网页爬虫。
* **规则 3.3 [新增]**：遵守日本《不正競争防止法》及各数据源 `robots.txt` 规定。对于明确禁止爬取的页面路径，系统须自动跳过并记录 `SKIP_ROBOTS_TXT` 日志。

## 4. 情感打分审计 (Sentiment Auditability)

* **规则 4.1**：所有用于实盘交易的 `sentiment_score` 必须是确定性的。如果在 `news_sentiment` 中发现极端得分（如 `1.0` 满分或 `-1.0` 极差），必须记录触发该得分的 LLM Prompt 快照或判定词汇。
* **规则 4.2**：不允许手工修改数据库中的 `sentiment_score`。如需纠错，必须插入新的 `model_version`（例如从 `v1.0` 升级为 `v1.1_hotfix`），并保留旧数据以供审计追踪。
* **规则 4.3 [新增 — 日语 NLP 质量基线]**：鉴于日语财经文本的特殊性（否定表达、敬语层级、术语歧义），每次 `model_version` 迭代上线前，必须通过 **Golden Set 回归测试**：
  - 维护一个不少于 200 条人工标注的日语财经新闻 Golden Set（含正面/负面/中性各类别）。
  - 新模型版本在 Golden Set 上的 **Macro-F1 不得低于 0.75**，且与前版本相比的 **得分漂移（Mean Absolute Drift）不得超过 0.10**。
  - 测试结果须存入 `sentiment_model_eval` 表（见 Design 文档 Schema 更新），未通过测试的模型版本禁止写入 `news_sentiment`。

## 5. 数据去重 (Deduplication) [新增章节]

同一事件会被多个来源重复报道。如果不做去重，情感得分会被同一事件反复加权，人为放大信号强度。

* **规则 5.1**：`news_id` 不得仅使用 URL hash。须引入 **事件级去重（Event-level Dedup）** 机制：对同一 `symbol` 在 ±4 小时窗口内、标题 TF-IDF 余弦相似度 > 0.85 的新闻，归入同一 `event_cluster_id`。
* **规则 5.2**：在 `ss7_sqlite_news_overlay.py` 中计算情感均值时，同一 `event_cluster_id` 下的多条新闻只取**最早入库的那条**的得分，或取该 cluster 的得分中位数，避免重复计数。
* **规则 5.3**：`event_cluster_id` 的生成逻辑须在 `news_analyzer.py` 中实现，并将结果写入 `news_feed` 表的 `event_cluster_id` 字段（见 Design 文档 Schema 更新）。

## 6. 数据保留与归档 (Retention Policy) [新增章节]

`news_feed` 表会随时间无限膨胀，需明确生命周期管理。

* **规则 6.1**：`news_feed` 和 `news_sentiment` 中超过 **2 年** 的数据，按季度归档至 `news_feed_archive` 和 `news_sentiment_archive` 表（或导出为 Parquet 文件存入冷存储）。
* **规则 6.2**：归档前须确认该时段数据未被任何活跃回测任务引用。归档操作记录写入 `data_lifecycle_log`。
* **规则 6.3**：主表建议按 `published_ts` 做月度分区（如使用 SQLite，可通过应用层逻辑模拟分区查询），保证近期数据的查询性能。
