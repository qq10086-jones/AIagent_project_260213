# News Worker — 任务清单 v1.1

更新日期：2026-03-11

> **定位说明**：本任务清单聚焦于 News Worker 本身（收集、分析、存储），
> 不包含 ss7 量化模型内部的信号消费逻辑。

---

## P0：基础建设（已完成）

- [x] **DB Schema**：`trade_schema.py` 新增 `news_feed`、`news_sentiment`、`sentiment_model_eval`、`data_lifecycle_log` 四张表及索引
- [x] **news_ingester.py**：独立收集脚本，Google RSS 抓取，SHA-256 精确去重，写入 `news_feed`
- [x] **news_analyzer.py**：独立分析脚本，TF-IDF 事件聚类 + LLM 打分，写入 `news_sentiment`
- [x] **目录结构**：`news/` 包独立，与量化引擎代码分离；设计文档迁移至 `news/design/`

---

## P1：收集稳定性（下一阶段）

**目标：让 news_ingester 成为可靠的生产级收集工具**

- [x] **指数退避重试**：`_get()` 失败后自动重试最多 3 次，间隔 2s / 4s / 8s，全部失败返回 None，继续下一个 symbol（2026-03-11）
- [x] **收集结果摘要**：运行结束输出结构化报告：总 symbol 数、失败列表、新增/重复计数、Top 5 symbol、健康状态（2026-03-11）
- [x] **动态断流报警**：`check_degradation()` — ≥90 天历史用 P5 百分位，<90 天用 30d 均值 × 30% 冷启动阈值，触发输出 `DATA_DEGRADED`（2026-03-11）
- [x] **限速**：symbol 间间隔 1s（Governance Rule 3.2）（2026-03-11）
- [x] **robots.txt 合规**：集成 `robotparser`，自动跳过禁止爬取的路径并记录日志（Governance Rule 3.3）（2026-03-11）

---

## P2：数据来源扩展

**目标：减少对 Google News 单一来源的依赖**

- [x] **TDnet RSS 接入**：日本上市公司官方信息披露，最权威的日本市场新闻源；全量 RDF feed 进程内缓存，按4位代码过滤（2026-03-11）
- [x] **J-Quants API 接入**：日本取引所集团官方数据接口；需 `JQUANTS_REFRESH_TOKEN` 环境变量，缺失时静默跳过（2026-03-11）
- [x] **来源标记**：`news_feed.source` 字段区分 `google_news_rss` / `tdnet` / `jquants`，便于后续质量分析（2026-03-11）

---

## P3：NLP 质量提升

**目标：让情感打分更准、可审计**

- [x] **Golden Set 回归测试**：`eval_golden_set.py` — 加载 `golden_set/golden_set_v1.csv`（当前 30 条样本，需扩充至 ≥200），Macro-F1 ≥ 0.75 + 漂移 ≤ 0.10，结果写 `sentiment_model_eval`（2026-03-11）
  - [ ] **扩充标注集**：将 `golden_set_v1.csv` 从 30 条扩充至 ≥200 条（人工标注工作）
- [x] **多模型对比**：`news_analyzer.py --model <model_id>` — `(news_id, model_version)` 联合主键，同一新闻可用不同模型打分并横向对比（已内置）
- [ ] **差异化衰减**：ss7 消费端改用每条新闻的 `expected_impact_days` 做指数衰减，废弃全局固定半衰期（ss7 侧变更，待 News Worker 稳定后实施）

---

## P4：数据治理与生命周期

- [x] **data_archiver.py**：按季度归档超过 2 年的 `news_feed` 和 `news_sentiment` 至 `*_archive` 表，写入 `data_lifecycle_log`；支持 `--dry-run`（2026-03-11）
- [x] **QA 回归验证**：`qa_pit_regression.py` — 7 个 look-ahead case 全部通过，验证 PIT 双时间戳过滤和 cluster 去重的正确性（Governance Rule 1.4）（2026-03-11）

---

## 暂缓（不在当前 scope）

- Nexus Worker Contract 对齐（P3 工作，等工具稳定后再做）
- News Monitor Dashboard（等 analyzer 稳定、有真实数据后再做）
