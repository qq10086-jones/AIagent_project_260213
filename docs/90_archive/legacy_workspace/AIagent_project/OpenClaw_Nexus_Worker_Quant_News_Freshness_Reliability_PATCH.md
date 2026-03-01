# News Freshness & Reliability（时效性与可靠性）补丁（MD）
> 适用：OpenClaw Nexus / Worker-Quant v1.3（含 News Risk Factor 与 Portfolio Loop）  
> 目标：在“GDELT + Google News RSS（宏观嗅探）+ Browser Agent（定点取证）”组合拳下，建立**可观测的时效性指标、自动降级机制、断流告警、深度取证触发器**，避免“过时新闻因子影响今日决策/曲线”。

---

## 1. 背景与设计结论（组合拳是对的，但必须可观测）

### 1.1 当前情报收集的分工
- **全市场大盘嗅探**：使用 **GDELT** 与 **Google News RSS**（HTTP 拉取结构化文本）  
  - 优点：快、稳、维护成本低、不会被网页结构/反爬频繁打断  
- **微观定点取证**：使用 **Browser Agent**（真实联网浏览器）  
  - 触发场景：进入 `quant.deep_analysis`，对特定标的（如 9432.T）抓详情页截图，作为“视觉证据”写入 evidence

### 1.2 为什么不用 Browser 抓全市场新闻首页
- Headless Browser 抓 Yahoo Finance / Bloomberg 首页成本高（渲染/图片/弹窗/反爬），且页面结构变动导致脆弱  
- 全市场新闻本质是“海量、低信噪比”，更适合结构化源做筛选，再对少数高价值事件做取证

### 1.3 真正的风险点：时效性（Freshness）
“用不用浏览器”不是时效性的决定因素。决定因素是：
- 数据源本身的更新延迟
- 你的抓取频率与查询窗口
- 去重/解析/写库流水线的延迟与失败
- 是否有**延迟遥测、断流检测、降级与回补**

---

## 2. Freshness Telemetry（新鲜度遥测：必须落库）

### 2.1 News Item 的必备时间字段（写入 fact_item 或其扩展字段）
对每条新闻（`fact_item(kind="news_item")`）必须记录：
- `published_at`：源侧发布时间（若缺失，需标记）
- `ingested_at`：系统抓取入库时间
- `source`：gdelt / google_rss / (future) official_ir / etc.
- `query_window`：本次抓取时间窗（例如 last_2h / last_24h）
- `dedup_key`：去重键（hash(title+url) 或 url canonical）
- `symbols`：命中标的（可多对多）

**派生指标（可写入或运行时计算）**
- `lag_seconds = ingested_at - published_at`
- `staleness_flag`：fresh / stale / expired（见 3.1）

### 2.2 Run-Level 汇总指标（写入 run report / metrics）
每次 `news.daily_report`（或 fetch_news_daily）运行必须输出：
- `items_raw`：拉取条数
- `items_deduped`：去重后条数
- `items_matched_watchlist`：命中持仓/关注标的条数
- `lag_p50 / lag_p90 / lag_max`
- `missing_published_at_ratio`
- `source_breakdown`：各 source 条数占比
- `error_count`：失败次数（HTTP/解析/写库）

> 建议在 Discord 日报顶部固定一行：  
> **Freshness：P50=xx min，P90=xx min，Max=xx h；Items=xxx（dedup=yyy）；断流=否/是**

---

## 3. Freshness Guard（时效性风控：自动降级/停用/告警）

### 3.1 时效阈值（建议默认值，可配置）
- **T1（Stale）**：`lag_seconds > 3600`（> 60 分钟）  
  - 允许计算 news_risk，但标记 `stale=true`，策略侧降权
- **T2（Expired）**：`lag_seconds > 21600`（> 6 小时）或出现断流信号  
  - news_risk 因子**停用或强降权**（见 3.2）并触发告警

> 阈值应做成配置：`freshness_t1_seconds / freshness_t2_seconds`。

### 3.2 因子降级策略（推荐：降权而非硬改曲线）
对 `feature_daily(news_risk_z)` 引入 `news_factor_weight`：
- fresh：`weight = 1.0`
- stale：`weight = 0.5`（或按 lag 线性/指数衰减）
- expired：`weight = 0.0`（停用）或 `0.2`（极弱信号）

策略侧使用：
- `effective_news_risk_z = news_risk_z * news_factor_weight`
- 仓位缩放：`position_scale = clamp(1 - k * sigmoid(effective_news_risk_z), min=..., max=...)`

### 3.3 断流检测（Coverage Guard）
触发条件示例（任一满足视为异常）：
- `items_raw` 或 `items_deduped` 显著低于过去 7 日均值（例如 < 20%）
- `items_matched_watchlist == 0` 且持仓不为空（异常）
- 连续 N 次 run 失败或超时
- `missing_published_at_ratio` 突然升高（例如 > 50%）

动作：
- 标记本次 run `coverage_flag=degraded`
- 将当天 news 因子降级（3.2）
- Discord 告警（3.4）

### 3.4 Discord 告警模板（建议固定）
- 标题：⚠️ News Feed Freshness Degraded
- 内容：
  - P50/P90/Max lag
  - items（raw/dedup/matched）
  - suspected cause（RSS timeout / GDELT empty / parse error）
  - action taken（weight reduced / factor disabled / deep_analysis triggered）

---

## 4. Browser Agent 的“深度取证触发器”（从被动到主动）
> 目标：Browser 不做全市场抓取，但作为 **fallback + 证据增强**，在关键事件出现时自动出手。

### 4.1 触发条件（推荐）
对持仓/关注标的（watchlist）满足任一：
1) `news_risk_z` 突增（例如日内增量 > Δthreshold 或 zscore 超阈值）
2) 出现高严重度 tag（监管/诉讼/财务造假/停产/事故/召回）
3) 价格异常波动（close-to-close 或 intraday > Nσ）
4) “价格在动但新闻很少”（coverage_flag=degraded 且价格波动异常）

### 4.2 触发后的动作（最小闭环）
- 调用 `browser.screenshot` 抓：公司 IR/公告页、交易所公告页、权威媒体原文页（按 Source Hierarchy 白名单）
- 生成 `evidence`（截图 hash）
- 抽取关键信息生成 `fact_item(kind="news_evidence" / "event_evidence")`
- 将 deep_analysis 结果回填到报告中（必须引用 evidence link）

---

## 5. 回补与重试（让“快”与“稳”兼得）

### 5.1 两阶段拉取（建议）
- **Phase A：快路径**（每小时/更频繁）
  - GDELT + RSS，小窗口（last_1h / last_2h）
  - 目标：快速发现事件，及时更新 risk 因子
- **Phase B：慢路径回补**（每日一次）
  - 更大窗口（last_24h / last_48h）
  - 目标：补齐 Phase A 漏掉的新闻，修复断流时的 coverage

### 5.2 去重与幂等
- 去重键：`dedup_key = hash(canonical_url or title+publisher+published_at)`
- 写库必须幂等：重复抓取不会产生重复 fact_item（或通过 unique constraint 防止）

---

## 6. 数据契约（推荐写入 Facts/Feature Schema）

### 6.1 fact_item(kind="news_item") 最小字段集合
- `published_at`（可空但需标记）
- `ingested_at`
- `source`
- `publisher`
- `title`
- `url`
- `symbols[]`
- `tags[]`
- `dedup_key`
- `query_window`

### 6.2 feature_daily 相关字段建议（扩展）
- `feature_name = "news_risk_z"`
- `value = news_risk_z`
- `source_fact_ids = [news_item_fact_ids...]`
- 可选：`meta_json` 包含 `lag_p50/lag_p90/coverage_flag/news_factor_weight`

---

## 7. 验收标准（Definition of Done）
- 日报固定输出 Freshness 指标（P50/P90/Max + 条数）
- 当延迟/断流异常时：
  - 自动降级/停用 news 因子（影响仓位/风险预算，而非硬改曲线）
  - Discord 发出告警说明“发生了什么 + 做了什么”
- 触发条件满足时可自动进入 deep_analysis 并产出 Browser 视觉证据
- 所有结论可溯源：news_item → evidence → link

---

**End.**
