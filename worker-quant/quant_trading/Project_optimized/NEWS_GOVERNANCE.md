# News Data Governance & Compliance (新闻数据治理规范)

作为量化系统中的外部非结构化数据源，新闻数据极易引入系统性偏差和合规风险。所有针对 News Pipeline 的开发和维护必须遵守以下治理规范。

## 1. 消除未来函数 (Zero Look-ahead Bias)
**红线原则**：在任何历史回测（Backtesting）和特征工程中，严禁模型“看到”在交易时刻尚未发布的新闻。
* **规则 1.1**：所有写入 `news_feed` 的记录必须同时包含 `published_ts`（新闻实际发生时间）和 `ingested_ts`（系统抓取到它的时间）。
* **规则 1.2**：回测时，SQL 查询必须同时约束 `published_ts < order_time` 且 `ingested_ts < order_time`，以系统真实接收时间为准，防止利用事后修改/补发的新闻。

## 2. 缺失与降级处理 (Fail-closed & Degradation)
新闻网络 API 极易中断（Rate limit, 网站改版）。
* **规则 2.1 (静默降级)**：如果今天某只股票无法获取新闻，其当日情感得分应归一化为中性（`0.0`）或继承上一日衰减值，绝不能抛出异常阻断当天的量化选股（Screener）和下单逻辑。
* **规则 2.2 (大面积断流报警)**：如果某天全市场抓取到的新闻条数低于 30 日均值的 20%，系统必须向 Orchestrator 发送 `DATA_DEGRADED` 警告，交由 PM 判断是否暂停当日实盘。

## 3. 合规与抓取道德 (Scraping Ethics)
对于外部网站的数据采集必须遵守商业道德与当地法律。
* **规则 3.1**：所有爬虫请求必须带有明确的 `User-Agent`，标明为内部量化机器人的识别名。
* **规则 3.2**：对于非官方 API 的网页抓取，请求频率（Rate Limit）必须限制在 `≤ 1 req/sec`，防止对目标服务器造成 DDoS。优先使用官方付费 API（如 Nikkei, J-Quants）替代网页爬虫。

## 4. 情感打分审计 (Sentiment Auditability)
* **规则 4.1**：所有用于实盘交易的 `sentiment_score` 必须是确定性的。如果在 `news_sentiment` 中发现极端得分（如 `1.0` 满分或 `-1.0` 极差），必须记录触发该得分的 LLM Prompt 快照或判定词汇。
* **规则 4.2**：不允许手工修改数据库中的 `sentiment_score`。如需纠错，必须插入新的 `model_version`（例如从 `v1.0` 升级为 `v1.1_hotfix`），并保留旧数据以供审计追踪。