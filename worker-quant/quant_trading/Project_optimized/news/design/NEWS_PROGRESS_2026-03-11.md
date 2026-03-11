# News Worker — 项目进展报告

**日期**：2026-03-11
**状态**：P0-P2 优化完成，生产就绪（pending 真实 API 验证）

---

## 1. 本轮完成内容

### 1.1 全阶段功能交付（P0 → P4）

| 阶段 | 核心内容 | 文件 |
|------|---------|------|
| P0 基础建设 | DB schema、news_ingester、news_analyzer、目录结构 | `trade_schema.py`, `news_ingester.py`, `news_analyzer.py` |
| P1 收集稳定性 | 指数退避重试、限速、断流报警、robots.txt 合规 | `news_ingester.py` |
| P2 数据来源扩展 | TDnet RDF、J-Quants API（可选）、source 标记 | `news_ingester.py` |
| P3 NLP 质量 | Golden Set 回归测试框架、多模型对比内置 | `eval_golden_set.py`, `golden_set/golden_set_v1.csv` |
| P4 数据治理 | 季度归档、PIT 回归验证 | `data_archiver.py`, `qa_pit_regression.py` |

### 1.2 QA 审视后的 Bug 修复与优化（P0-P2 优化轮次）

**P0 Bug 修复（5 项）**
- `data_archiver.py`：INSERT + DELETE 用 `with conn:` 包裹，实现原子事务
- `data_archiver.py`：移除 TIMEOUT 行过滤，TIMEOUT 行随 parent 一起归档，消除永久孤儿
- `news_ingester.py`：TDnet 代码匹配从子串搜索改为词边界正则 `(?<!\d)code(?!\d)`
- `news_ingester.py`：J-Quants 标题 em dash `—` 改为 ASCII `-`，消除 Windows cp932 编码错误
- `news_ingester.py`：移除 `__import__("os")` 反模式，改为顶部 `import os`

**P1 可靠性（4 项）**
- `_get()` 重构：`enumerate([(0,)] + ...)` 改为 `range(len(_RETRY_DELAYS) + 1)`，消除未使用变量
- 新增 `news/requirements.txt`：声明 `requests`、`anthropic`、`scikit-learn`、`numpy` 版本下限
- TDnet 解析加固：分步 fallback（namespaced RDF → findall → bare item），无匹配时打印 `[warn]`
- 新增 `validate_sources.py`：实网验证 TDnet/J-Quants 响应结构，CI 友好（exit 0/1）

**P2 数据质量（3 项）**
- `sentiment_model_eval` 表从单一 PK 改为 `(model_version, eval_date)` 复合主键，保留历史评估记录
- 配套 schema 自动迁移逻辑（存量 DB 自动 rename → recreate → copy → drop）
- `eval_golden_set.py`：`INSERT OR REPLACE` 改为 `INSERT OR IGNORE`；`_get_last_f1()` 取最新记录
- `qa_pit_regression.py`：7 → 8 个测试，新增 `null_ingested_ts_blocked_by_schema`，正确定性为 schema 层保护
- `golden_set_v1.csv`：从 30 条扩充至 100 条，覆盖极端事件、重组公告、模糊标题、日英混合

---

## 2. 当前测试状态

```
qa_pit_regression.py   8/8 PASS   (Governance Rule 1.4)
P0 归档测试            5/5 断言通过（原子性 + TIMEOUT 归档）
P1-P2 smoke tests      5/5 通过
```

---

## 3. 遗留问题

### 3.1 需要真实网络环境验证（阻塞 P2 生产上线）

| 问题 | 验证方式 | 负责人 |
|------|---------|--------|
| TDnet RDF 实际 XML 结构是否与解析逻辑匹配 | `python news/validate_sources.py` | 需运行环境 |
| J-Quants API 字段名是否与代码中假设一致 | `JQUANTS_REFRESH_TOKEN=xxx python news/validate_sources.py` | 需 API token |

验证通过前，TDnet 和 J-Quants 来源静默返回 `[]`，不影响 Google News RSS 正常收集。

### 3.2 Golden Set 标注量不足（阻塞 P3 统计可信度）

- 当前 100 条，P3 标准要求 ≥200 条
- 运行 `eval_golden_set.py` 会打印 `WARN: golden set has 100 rows`
- 建议重点补充：标题含转折语气、多公司联合公告、行业整体下行、数字反转（如"亏损收窄"）

### 3.3 暂缓项（设计范围外）

- **差异化衰减（ss7 侧）**：ss7 消费端改用每条新闻 `expected_impact_days` 做指数衰减，废弃全局固定半衰期。需与 quant 模型迭代协调，单独立项
- **Nexus Worker Contract 对齐**：等 News Worker 稳定后再接入 Nexus 调度
- **News Monitor Dashboard**：等 analyzer 有真实数据后再做
- **Parquet 冷存储导出**：`data_archiver.py` 已预留接口，归档超过 5 年数据时实施

---

## 4. 下一步计划

### 近期（生产准备）

1. **执行 `validate_sources.py`**（需网络）：确认 TDnet/J-Quants 解析正确
2. **首次真实运行**：
   ```bash
   # 收集
   python news/news_ingester.py --db japan_market.db --symbols 9432.T,7203.T,9984.T
   # 分析（需 ANTHROPIC_API_KEY）
   ANTHROPIC_API_KEY=sk-... python news/news_analyzer.py --db japan_market.db --dry-run
   # 确认无误后去掉 --dry-run 正式打分
   ```
3. **补充 Golden Set 至 ≥200 条**（人工标注）

### 中期（质量提升）

4. **差异化衰减**：与 quant 团队协调 ss7 消费逻辑改造
5. **定期评估计划**：每次模型版本升级后跑 `eval_golden_set.py`，追踪历史 F1 趋势
6. **归档调度**：设置季度 cron，参考命令：
   ```
   # 每季度第一个工作日 00:30
   30 0 1 1,4,7,10 * python .../news/data_archiver.py --db japan_market.db
   ```

### 长期（扩展）

7. **Nexus 接入**：完成 Worker Contract 对齐后，将 news 信号接入 Nexus 调度体系
8. **Dashboard**：基于真实数据，建立新闻收集趋势和情感分布监控看板

---

## 5. 文件清单

```
news/
├── __init__.py
├── requirements.txt               # P1: 依赖声明
├── news_ingester.py               # 收集层（P0-P2 全部特性）
├── news_analyzer.py               # 分析层（TF-IDF + LLM）
├── eval_golden_set.py             # P3: Golden Set 回归测试
├── data_archiver.py               # P4: 季度归档
├── qa_pit_regression.py           # P4: PIT 回归验证（8/8）
├── validate_sources.py            # P1: 数据源实网验证工具
├── gen_synthetic_news.py          # 测试辅助工具
├── golden_set/
│   └── golden_set_v1.csv          # 100 条标注（需扩充至 ≥200）
└── design/
    ├── NEWS_SYSTEM_DESIGN_v1.md
    ├── NEWS_GOVERNANCE_v1.md
    ├── NEWS_SYSTEM_TASKS_v1.md
    └── NEWS_PROGRESS_2026-03-11.md  (本文件)
```
