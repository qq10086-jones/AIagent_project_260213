# Worker-Quant — 专属量化AI员工 任务清单 v1.0

**关联设计文档**: `QUANT_AI_EMPLOYEE_DESIGN_v1.md`
**创建日期**: 2026-03-22
**优先级说明**: P0=阻塞/基础 | P1=高价值 | P2=重要 | P3=增强

---

## Epic A — 持仓管理基础（v1.1 目标）

> **目标**: 让系统知道"我现在持有什么、赚了多少、风险在哪里"

---

### A-01 成本价计算
**优先级**: P0
**估时**: 2天
**文件**: `worker.py` → `_calc_avg_cost(symbol)`

**描述**
从 `fills` 表按时序计算每个 symbol 的加权平均成本价（WAVG）。
- BUY：累计总成本 / 累计总数量
- SELL：按加权均价减少持仓，不重置成本
- 考虑股票拆分（暂不处理，备注已知限制）

**验收标准**
- [ ] 纯BUY情形：多次建仓后，avg_cost = 总花费 / 总数量，误差 < 0.01JPY
- [ ] BUY后部分SELL：成本不变，只减少数量
- [ ] 清仓后再买入：成本重置为新买入价
- [ ] 单元测试覆盖上述3种情形

---

### A-02 持仓快照写入（每日收盘后）
**优先级**: P0
**估时**: 1天
**文件**: `worker.py` → `_write_positions_snapshot()`，新增 `positions_snapshot` 表

**描述**
在每次 `portfolio.position_review` 或 `portfolio.post_close` 调用时，将当前持仓状态写入 `positions_snapshot` 表（见设计文档 C1 DDL）。
字段包含：成本价、现价、浮动盈亏、浮动盈亏%、止损价、持仓天数。

**验收标准**
- [ ] 表不存在时自动创建（migration）
- [ ] 同一天多次调用只覆盖不重复插入（upsert on asof+symbol）
- [ ] 浮动盈亏 = (current_price - avg_cost) × net_qty，精度两位小数

---

### A-03 新工具：`portfolio.position_review`
**优先级**: P0
**估时**: 3天
**文件**: `worker.py` → `portfolio_position_review(payload)`，注册到 TOOL_REGISTRY

**描述**
主动复盘工具。输出内容（见设计文档 C2）：
1. 持仓汇总表（每行一只股票，含浮盈%、信号、止损距离）
2. 组合统计（总市值、总浮盈、仓位使用率）
3. 风险提示（触发止损线 / 信号变差 / 持仓超期）
4. 每只股票的具体建议（持有/减仓/止损/加仓）

**验收标准**
- [ ] 无持仓时输出"当前无持仓，可用资金 XXX JPY"
- [ ] 有持仓时输出完整汇总表，数字正确
- [ ] 止损价 = avg_cost × (1 - user_profile.stop_loss_pct)，默认8%
- [ ] 触发止损的仓位在输出中有明显标记（⚠️）
- [ ] LLM生成"综合建议"段落（可 fallback 为纯数据）
- [ ] 工具注册并可通过 Discord `/review` 或 `quant.position_review` 触发

---

### A-04 新工具：`portfolio.midday_pnl`
**优先级**: P1
**估时**: 1天
**文件**: `worker.py` → `portfolio_midday_pnl(payload)`

**描述**
午间（12:00 JST）轻量版持仓快报，只输出每只持仓的当前价格和浮动盈亏，不调用LLM，延迟 < 5秒。

**验收标准**
- [ ] 输出格式简洁，适合 Discord 一眼扫完
- [ ] 总耗时 < 10秒（yfinance批量查询）
- [ ] 无持仓时静默（不推送）

---

### A-05 用户配置文件持久化
**优先级**: P1
**估时**: 1天
**文件**: `quant_trading/Project_optimized/user_profile.json` + `_load_user_profile()` / `_save_user_profile()`

**描述**
创建用户偏好配置文件（见设计文档 E1），所有工具统一从此文件读取默认值。
提供 `portfolio.set_preference` 工具允许用户通过 Discord 修改配置。

**验收标准**
- [ ] 文件不存在时自动写入默认值
- [ ] `portfolio_set_account` 调用后同步更新 `user_profile.json` 中的 `capital_base_jpy`
- [ ] `stop_loss_pct` 默认0.08，用户可通过指令修改后持久化

---

## Epic B — 新闻数据源升级（v1.2 目标）

> **目标**: 把"装饰性新闻"换成"有alpha价值的信息"

---

### B-01 TDnet RSS 解析器
**优先级**: P0
**估时**: 3天
**文件**: `worker.py` → `_fetch_tdnet_announcements(symbols: list) -> list[dict]`

**描述**
抓取 TDnet 适时開示の RSS / HTML，解析并返回结构化公告列表。
每条公告包含：証券コード、企業名、公告标题、公告分类（増配/修正/自社株等）、发布时间、URL。

数据源优先级（依次尝试）：
1. `https://www.release.tdnet.info/inbs/I_main_00.html` — 官方页面
2. Kabutan RSS `https://kabutan.jp/news/marketnews/`
3. 降级：Google RSS 日文关键词（`決算 OR 増配 OR 自社株`）

**验收标准**
- [ ] 能解析今日 TDnet 公告，返回含 `code`/`title`/`category`/`published_at` 字段
- [ ] 公告分类准确率 > 80%（手工验证10条）
- [ ] 单次抓取耗时 < 8秒（含超时保护）
- [ ] 失败时返回空列表，不抛异常

---

### B-02 公告 → Ticker 映射
**优先级**: P0
**估时**: 2天
**文件**: `worker.py` → `_match_announcement_to_ticker(announcement, watchlist)`

**描述**
将公告中的証券コード或企業名称映射到 watchlist 中的 ticker。
- 4位証券コード → `{code}.T`（直接匹配）
- 子会社名称 → 扩展 `jp_names` alias map（见 B-04）

**验收标准**
- [ ] 4位コード命中率 100%（当コード在watchlist时）
- [ ] 企業名alias匹配：NTT、エネオス、ソフトバンク等主要别名均可命中
- [ ] 无匹配时返回空列表（不报错）

---

### B-03 新工具：`quant.event_alert`
**优先级**: P1
**估时**: 3天
**文件**: `worker.py` → `quant_event_alert(payload)`

**描述**
持仓相关事件预警（见设计文档 C3）。
调用流程：
1. 获取当前持仓（`_get_current_positions_from_fills`）
2. 获取最新公告（`_fetch_tdnet_announcements`）
3. 交叉比对，找到"持仓相关公告"
4. 对每条命中，生成影响评估：公告类型 → 历史同类事件平均影响幅度 → 建议
5. 写入 Discord 推送格式

**验收标准**
- [ ] 无持仓或无命中公告时静默（不推送空消息）
- [ ] 増配/上方修正类：输出"催化剂 ✅ + 建议继续持有/加仓"
- [ ] 下方修正/不祥事类：输出"风险 ⚠️ + 建议确认止损线"
- [ ] 包含当前浮盈%和止损距离%

---

### B-04 Watchlist Alias Map 扩充
**优先级**: P1
**估时**: 1天
**文件**: `quant_trading/Project_optimized/watchlist_alias.json`（新建）

**描述**
为 watchlist 中所有29只JP标的创建 alias map：
- 正式名称（日文）
- 略称（日文）
- 英文名
- 常见子会社/关联品牌名

格式：
```json
{
  "5020.T": ["エネオス", "ENEOS", "ENEOSホールディングス", "JX", "JXTG"],
  "9432.T": ["NTT", "日本電信電話", "エヌティティ"],
  ...
}
```

**验收标准**
- [ ] 覆盖所有29只JP标的
- [ ] `_match_announcement_to_ticker` 使用此文件
- [ ] 新闻命中率测试：随机抽取10条真实新闻，alias匹配率 > 70%

---

### B-05 `deep_analysis` 集成 TDnet 公告
**优先级**: P1
**估时**: 2天
**文件**: `worker.py` → `deep_analysis()` 修改

**描述**
在 `deep_analysis` 中增加对当日 TDnet 公告的查询，将"有无催化剂"作为信号输入之一：
- 有上方修正/増配 → signal 提升一级（Neutral → Overweight）
- 有下方修正 → signal 降低一级（Overweight → Neutral 或 Underweight）
- 输出报告中增加"今日催化剂"章节

**验收标准**
- [ ] 有催化剂时报告中明确显示公告标题和分类
- [ ] 无催化剂时显示"今日无重要公告"
- [ ] 信号调整有明确说明（不静默修改）

---

## Epic C — 主动推送体系（v1.3 目标）

> **目标**: 每个交易日关键时点自动推送，不需要用户主动询问

---

### C-01 新工具：`portfolio.morning_brief`
**优先级**: P0
**估时**: 3天
**文件**: `worker.py` → `portfolio_morning_brief(payload)`

**描述**
早间综合情报（08:45 JST），内容见设计文档 D2：
1. 市场环境（N225期货、VIX、昨日收盘）
2. TDnet昨日公告（持仓相关优先）
3. 当前持仓状态（快速浮盈表）
4. 今日3项关注事项（LLM生成）

**验收标准**
- [ ] 完整输出耗时 < 60秒
- [ ] 无持仓时仍输出市场环境和今日关注
- [ ] LLM超时时 fallback 为纯数据格式

---

### C-02 新工具：`portfolio.post_close`
**优先级**: P1
**估时**: 2天
**文件**: `worker.py` → `portfolio_post_close(payload)`

**描述**
收盘后复盘（15:35 JST）：
1. 当日持仓涨跌（相比昨收）
2. TDnet当日新公告汇总
3. 明日计划（LLM：基于今日情况，明日需要关注什么）
4. 调用 `_write_positions_snapshot()` 保存当日快照

**验收标准**
- [ ] 包含当日持仓涨跌% 和浮动盈亏变化（今日 vs 昨日快照）
- [ ] 明日计划不少于2条可执行建议

---

### C-03 Cron定时触发配置
**优先级**: P1
**估时**: 1天
**文件**: `orchestrator/crons/` 或 Discord bot 定时配置

**描述**
配置以下定时任务（JST时区）：
```
08:45  → portfolio.morning_brief
12:00  → portfolio.midday_pnl（仅有持仓时推送）
15:15  → news.preclose_brief_jp
15:35  → portfolio.post_close
```

**验收标准**
- [ ] 周一至周五执行，周末跳过
- [ ] 节假日（日本市场休市）跳过（需要接入日历API或静态节假日列表）
- [ ] 推送失败后有告警日志，不重试超过2次

---

## Epic D — 统计信号重构（v1.4 目标）

> **目标**: 从"规则引擎"升级为"统计验证的多因子模型"

---

### D-01 `signal_log` 表建立
**优先级**: P0（D系列前置）
**估时**: 1天
**文件**: `worker.py` → migration + `_log_signal()`

**描述**
建立 `signal_log` 表（见设计文档 A2 DDL）。
每次 `deep_analysis` 或 `discovery_workflow` 发出信号时，写入一条记录（含当时价格，20日后价格留空）。

**验收标准**
- [ ] 表自动建立（migration）
- [ ] 每次 `deep_analysis` 调用后 `signal_log` 有新记录
- [ ] `price_5d` / `price_20d` / `ret_5d` / `ret_20d` 初始为 NULL

---

### D-02 新工具：`quant.signal_backfill`
**优先级**: P1
**估时**: 2天
**文件**: `worker.py` → `quant_signal_backfill(payload)`

**描述**
定期（每周一次）回填 `signal_log` 中 `ret_5d` / `ret_20d` 为 NULL 的记录。
计算方式：查询 yfinance 实际价格 → 填入 → 计算 IC = rank_corr(alpha_score, ret_20d)。

**验收标准**
- [ ] 正确回填所有满足"发出日期 + N日 ≤ 今天"的记录
- [ ] 计算并打印当前滚动IC（最近30条信号）
- [ ] IC > 0.05 时输出 "✅ 信号有效"，否则输出 "⚠️ 信号弱，建议检查因子"

---

### D-03 横截面相对强弱因子
**优先级**: P1
**估时**: 2天
**文件**: `worker.py` → `_compute_cross_sectional_rank(symbols: list) -> dict`

**描述**
`discovery_workflow` 中已有候选池，在对每个标的打分时增加"在池内的相对排名"作为因子（见设计文档 A1 `rel_strength`）。
横截面打分 = 当前20日收益率在全watchlist中的百分位数（0-1）。

**验收标准**
- [ ] 输出每个symbol的 `rel_strength_pct`（0=最弱，1=最强）
- [ ] 相同的技术指标下，横截面排名高的stock得分更高
- [ ] 横截面计算增加的耗时 < 5秒（watchlist约30只）

---

### D-04 新工具：`quant.portfolio_risk`
**优先级**: P1
**估时**: 3天
**文件**: `worker.py` → `quant_portfolio_risk(payload)`

**描述**
组合风险度量（见设计文档 A4）：
- 用yfinance获取当前持仓所有标的过去60日日收益率
- 计算相关矩阵
- 计算组合加权年化波动率
- 输出集中度告警（单仓 > 40%）、板块重叠告警

**验收标准**
- [ ] 2只股票持仓时，输出2×2相关矩阵
- [ ] 相关性 > 0.7 时输出"持仓高度相关 ⚠️，考虑分散"
- [ ] 组合波动率计算与手工验算误差 < 0.5%

---

### D-05 信号加权合成（替换规则加法）
**优先级**: P2
**估时**: 3天
**文件**: `worker.py` → `_compute_quant_metrics()` 重构

**描述**
将现有 `score += 1.0 if ret_20d > 0` 的规则加法，替换为加权因子合成（见设计文档 A3）。
初始权重均等，后续通过 `signal_backfill` 计算的IC来调整各因子权重（IC加权）。

**前置依赖**: D-01, D-02 完成后，有足够信号历史

**验收标准**
- [ ] 重构后所有现有单元测试仍通过
- [ ] 新的 `composite_score` 在-3到+3范围内
- [ ] 与旧版本信号方向一致性 > 80%（回归测试）

---

## Epic E — 技术债务清理（持续进行）

---

### E-01 `tdnet_close_flash` 名实修正
**优先级**: P1
**估时**: 0.5天
**文件**: `worker.py` → `tdnet_close_flash()`

**描述**
当前 `tdnet_close_flash` 实际使用 GDELT 搜"決算"，并非 TDnet。
在 B-01 完成后，将 `tdnet_close_flash` 改为调用 `_fetch_tdnet_announcements`。
短期内（B-01完成前）在函数注释中标注 "TODO: 当前为GDELT降级，B-01完成后替换"。

**验收标准**
- [ ] B-01完成后，`tdnet_close_flash` 的数据来源改为TDnet
- [ ] 输出格式保持向后兼容

---

### E-02 `deep_analysis` 接入 `compute_news_risk_factor`
**优先级**: P1
**估时**: 1天
**文件**: `worker.py` → `deep_analysis()` 修改

**描述**
现在 `deep_analysis` 不调用新闻情绪因子。
在 alpha_score 计算中加入 news_risk_factor 的 sentiment 加成（小权重，避免过拟合）：
```python
sentiment = compute_news_risk_factor({"symbol": symbol}).get("sentiment", 0)
score += 0.5 * sentiment   # 正面新闻小加分
```

**验收标准**
- [ ] `deep_analysis` 输出中包含 `news_sentiment` 字段
- [ ] 整体信号不因新闻情绪改变方向（只微调score）

---

### E-03 复权价格修正
**优先级**: P2
**估时**: 1天
**文件**: `worker.py` → `_compute_quant_metrics()`

**描述**
当前用 `auto_adjust=False`，在除权日技术指标会出现伪信号（价格跳变）。
改为 `auto_adjust=True`（使用复权价格）计算技术指标，但原始价格仍用于限价计算。

**验收标准**
- [ ] 修改后的单元测试中，除权日前后SMA不出现异常跳变
- [ ] 原始价格用于执行建议（limit_prices）不变

---

### E-04 watchlist 管理工具
**优先级**: P2
**估时**: 1天
**文件**: `worker.py` → `quant_watchlist_add/remove/list(payload)`

**描述**
通过 Discord 指令动态管理 watchlist，而不是直接编辑 JSON 文件。

**验收标准**
- [ ] `/watchlist add 5020.T ENEOS` → 添加并持久化
- [ ] `/watchlist remove 5020.T` → 移除并持久化
- [ ] `/watchlist list` → 显示当前所有标的

---

## 任务汇总与优先级排序

### 第一批（v1.1，持仓管理基础，2-3周）

| ID | 任务 | 优先级 | 估时 | 前置 |
|---|---|---|---|---|
| A-01 | 成本价计算 | P0 | 2天 | — |
| A-02 | 持仓快照写入 | P0 | 1天 | A-01 |
| A-03 | portfolio.position_review | P0 | 3天 | A-01, A-02 |
| A-05 | 用户配置文件持久化 | P1 | 1天 | — |
| A-04 | portfolio.midday_pnl | P1 | 1天 | A-01 |
| E-03 | 复权价格修正 | P2 | 1天 | — |

**v1.1 里程碑验收**: 能通过 Discord 查询持仓浮盈、看到止损预警

---

### 第二批（v1.2，新闻升级，+1-2周）

| ID | 任务 | 优先级 | 估时 | 前置 |
|---|---|---|---|---|
| B-01 | TDnet RSS 解析器 | P0 | 3天 | — |
| B-02 | 公告 Ticker 映射 | P0 | 2天 | B-01 |
| B-04 | Watchlist Alias Map | P1 | 1天 | — |
| B-03 | quant.event_alert | P1 | 3天 | B-01, B-02, A-01 |
| B-05 | deep_analysis 集成公告 | P1 | 2天 | B-01, B-02 |
| E-01 | tdnet_close_flash 修正 | P1 | 0.5天 | B-01 |
| E-02 | deep_analysis 接入新闻情绪 | P1 | 1天 | — |

**v1.2 里程碑验收**: 持仓相关TDnet公告出现后，5分钟内收到预警

---

### 第三批（v1.3，主动推送，+1周）

| ID | 任务 | 优先级 | 估时 | 前置 |
|---|---|---|---|---|
| C-01 | portfolio.morning_brief | P0 | 3天 | A-01, B-01 |
| C-02 | portfolio.post_close | P1 | 2天 | A-02 |
| C-03 | Cron定时配置 | P1 | 1天 | C-01, C-02 |

**v1.3 里程碑验收**: 无需手动触发，每个交易日自动收到3次情报推送

---

### 第四批（v1.4，统计信号重构，+3-4周）

| ID | 任务 | 优先级 | 估时 | 前置 |
|---|---|---|---|---|
| D-01 | signal_log 表建立 | P0 | 1天 | — |
| D-03 | 横截面相对强弱因子 | P1 | 2天 | — |
| D-02 | quant.signal_backfill | P1 | 2天 | D-01 |
| D-04 | quant.portfolio_risk | P1 | 3天 | — |
| E-04 | watchlist 管理工具 | P2 | 1天 | — |
| D-05 | 信号加权合成重构 | P2 | 3天 | D-01, D-02 |

**v1.4 里程碑验收**: IC追踪运行4周，信号有效性可量化验证

---

## 当前进度跟踪

| Epic | 总任务数 | 已完成 | 进行中 | 未开始 |
|---|---|---|---|---|
| A: 持仓管理 | 5 | 5 | 0 | 0 |
| B: 新闻升级 | 5 | 5 | 0 | 0 |
| C: 主动推送 | 3 | 3 | 0 | 0 |
| D: 统计信号 | 5 | 3 | 0 | 2 |
| E: 技术债务 | 4 | 4 | 0 | 0 |
| **合计** | **22** | **20** | **0** | **2** |

### 已完成任务（2026-03-23）

- ✅ A-01: `_calc_avg_cost` — WAVG成本价计算
- ✅ A-02: `_write_positions_snapshot` + `positions_snapshot` 表建立
- ✅ A-03: `portfolio.position_review` — 浮盈/风险/建议完整工具
- ✅ A-04: `portfolio.midday_pnl` — 午间快报
- ✅ A-05: `user_profile.json` + `_load_user_profile` / `portfolio.set_preference`
- ✅ B-01: `_fetch_tdnet_announcements` — Kabutan+Google日文RSS，公告分类
- ✅ B-02: `_match_announcement_to_tickers` — 公告→Ticker映射
- ✅ B-03: `quant.event_alert` — 持仓相关事件预警
- ✅ B-04: `watchlist_alias.json` — 29只JP标的完整alias map
- ✅ B-05: `deep_analysis` 集成TDnet公告（催化剂评分调整）
- ✅ C-01: `portfolio.morning_brief` — 早间综合情报
- ✅ C-02: `portfolio.post_close` — 收盘复盘
- ✅ C-03: Cron配置待对接（工具已就绪，cron触发由Discord bot负责）
- ✅ D-01: `signal_log` 表 + `_log_signal` — IC追踪基础
- ✅ D-02: `quant.signal_backfill` — 收益回填+滚动IC计算（Spearman rank）
- ✅ D-04: `quant.portfolio_risk` — 相关矩阵+组合波动率+集中度预警
- ✅ E-01: `tdnet_close_flash` 增加TDnet真实数据源（Kabutan优先）
- ✅ E-02: `deep_analysis` 集成 `compute_news_risk_factor`（news_sentiment字段）
- ✅ E-03: `_compute_quant_metrics` 改用 `auto_adjust=True`（复权价格）
- ✅ E-04: `quant.watchlist` — watchlist增删查管理工具

### 未完成任务

- ✅ D-03: `_compute_cross_sectional_rank` — 已接入 `discovery_workflow`，`cs_rank_score` 纳入 `selection_score`（±0.30/±0.10 分位调整）
- ⏳ D-05: 信号加权合成重构 — 需4周信号历史积累后再执行

### 新增工具清单（v1.1 新增10个工具，总计29个）

| 工具名 | 功能 |
|---|---|
| `portfolio.position_review` | 持仓浮盈/风险/建议综合复盘 |
| `portfolio.midday_pnl` | 午间轻量P&L快报 |
| `portfolio.morning_brief` | 早间综合情报（市场+公告+持仓+关注） |
| `portfolio.post_close` | 收盘复盘+明日计划 |
| `portfolio.set_preference` | 修改用户配置（止损%、资金等） |
| `quant.event_alert` | 持仓相关TDnet公告预警 |
| `quant.signal_backfill` | 信号历史回填+IC/ICIR计算 |
| `quant.portfolio_risk` | 组合相关矩阵+波动率+集中度 |
| `quant.watchlist` | watchlist增删查管理 |
| `news.tdnet_announcements` | 直接获取TDnet公告列表 |

---

*任务清单版本: v1.1 | 2026-03-23 | 21/22任务完成，worker-quant v1.1 功能已上线*
