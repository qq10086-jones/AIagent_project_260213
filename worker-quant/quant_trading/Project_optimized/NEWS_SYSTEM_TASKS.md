# News System Optimization Task List

## P0: 核心重构与持久化 (Foundation & Persistence)
- [ ] **DB Schema 升级**: 在 `trade_schema.py` 和 `db_update.py` 中引入 `news_feed` 和 `news_sentiment` 两张核心表。
- [ ] **拆分 Worker**: 从臃肿的 `worker.py` 中剥离新闻抓取代码，建立 `news_ingester.py` 作为独立脚本。
- [ ] **废弃 CSV 流程**: 修改 `ss7_sqlite_news_overlay.py` 中 `load_news_items()` 函数，使其直接从 SQLite 查询新闻，而不是读取易碎的 `.csv` 文件。

## P1: 健壮性与防未来函数 (Robustness & Point-in-Time)
- [ ] **Point-in-Time 查询**: 确保量化模型读取新闻时，严格过滤 `published_ts < market_open_ts`，绝不允许“收盘后的财报”影响“当天的开盘决策”（修复潜在的 Look-ahead Bias）。
- [ ] **引入权威数据源**: 减少对 Google News 的依赖，集成日本市场官方的 TDnet RSS 或 J-Quants 资讯 API 接口。
- [ ] **异常重试机制**: 在 `news_ingester.py` 中加入指数退避（Exponential Backoff）重试机制，防止网络抖动导致当天断流。

## P2: NLP 与情感引擎 (Sentiment Engine)
- [ ] **统一打分 Prompt**: 设计标准的 LLM Prompt，要求强制输出 JSON 格式 `{"sentiment": 0.8, "urgency": 0.9, "reason": "..."}`。
- [ ] **多模型回测支持**: 支持对同一条历史新闻使用不同版本的 LLM (`model_version`) 进行重复打分对比，寻找最稳定有效的分析模型。
- [ ] **极值衰减算法**: 实现新闻冲击力的时间衰减（半衰期机制，例如重大财报影响在 3 天内按指数衰减）。

## P3: Nexus 系统集成 (Nexus Compliance)
- [ ] **Worker Contract 对齐**: 在 `NEXUS_WORKER_CONTRACT.md` 中更新 `news.*` 系列工具的输入输出契约。
- [ ] **Dashboard 可视化**: 在 `app.py` (Streamlit 看板) 中增加一个 "News Monitor" 标签页，便于 PM 直接查看高风险舆情。