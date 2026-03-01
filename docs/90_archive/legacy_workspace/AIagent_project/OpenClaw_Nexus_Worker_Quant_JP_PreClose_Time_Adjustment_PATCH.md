# 日本市场收盘时间与“收盘前 15 分钟”调度修订补丁（MD）
> 适用：Pre-Close News Brief / `news.preclose_brief_jp`  
> 目的：修正“收盘前 15 分钟”的默认时间点，并补充日本市场收盘机制与 TDnet 公告时点的落地建议。

---

## 1. 背景：TSE 现金股票收盘时间已延长
日本交易所集团（JPX）信息显示，TSE 现金股票交易时段下午场为 **12:30–15:30（JST）**。  
因此，若你要在“收盘前 15 分钟”推送简报，默认时间应为：

- **15:15 JST（= 15:30 收盘前 15 分钟）**

> 修订结论：把文档中 `14:45 JST` 的默认值统一替换为 `15:15 JST`。

---

## 2. 修订项：`news.preclose_brief_jp` 的 Schedule

### 2.1 原默认（需废止）
- 14:45 JST（基于旧 15:00 收盘假设）

### 2.2 新默认（推荐）
- **15:15 JST（JST）**  
- 工作日（Mon–Fri）运行；可选增强：交易日判断（排除休市日）

### 2.3 可配置参数（建议加入 workflow config）
- `jp_market_close_time = "15:30"`
- `preclose_offset_minutes = 15`
- `preclose_run_time = close_time - offset = 15:15`
- `timezone = "Asia/Tokyo"`

---

## 3. 与收盘机制/大引け的关系（为什么 15:15 更贴合）
- 收盘前最后一段时间通常流动性更集中、价格对信息更敏感  
- 将简报时间设在 **15:15**，可以最大化“当日可操作窗口”，便于你的“是否操作模型”使用当日新闻因子做反馈分析

> 说明：本简报仍不输出最终操作结论，仅输出新闻/因子/可靠性信息与可追溯证据，供你的 action_model 使用。

---

## 4. TDnet 公告时点：建议引入“二段式”推送（可选增强）
你先前担心“关键公告在特定时点发布”是合理的。由于公告可能贴近收盘或收盘后发布，建议引入第二条轻量推送：

### 4.1 15:15 Pre-Close Brief（主简报）
- 覆盖：持仓 + 今日推荐（Top N）  
- 内容：last_6h 快路径 + last_24h 回补；freshness/coverage；news_risk_z；必要时触发 Browser 取证  
- 用途：收盘前决策支持（因子输入/feedback）

### 4.2 15:35 TDnet Close Flash（公告回补，轻量）
- 覆盖：仅持仓（优先）+ 风险事件候选  
- 内容：15:00–15:35 窗口的 TDnet/公告类事件（高严重度优先）  
- 用途：盘后风险提醒与次日计划输入（同样供 action_model 使用）

> 若你暂时只想保留一条推送：优先保留 15:15 主简报；TDnet Flash 可作为 vNext。

---

## 5. 设计文档/补丁同步点（建议）
需要同步修改的条目：
1) Pre-Close Brief 补丁中的默认时间：`14:45 → 15:15 JST`
2) `news.preclose_brief_jp` workflow 的 schedule 示例
3) Discord 文案：标题与时间戳显示（YYYY-MM-DD 15:15 JST）
4) Freshness Telemetry 的 `ingested_at` 采用 JST 或 UTC（建议统一 UTC 存储，展示 JST）

---

## 6. 验收标准（Definition of Done）
- `news.preclose_brief_jp` 在 **15:15 JST** 稳定触发并推送
- 简报标题/正文时间一致（不再出现 14:45）
- 可选：添加 `TDnet Close Flash` 的占位 workflow（可先不启用）

---

**End.**
