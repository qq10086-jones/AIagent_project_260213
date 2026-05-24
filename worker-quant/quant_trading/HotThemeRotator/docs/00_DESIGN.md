# HotThemeRotator Design

## 1. 背景

用户当前策略是基于市场温度、新闻催化、热点板块和龙头股强度进行短线轮动。交易目标通常是 2%-5% 利润，达到目标后清仓或换到新的热点标的。主操作市场是日股，但会观察 A股、美股和宏观事件带来的跨市场风险偏好。

旧项目 `Project_optimized` 已经具备日股数据库、候选股排序、新闻 overlay、盘中建议、SBI 实盘镜像信号、风控和报告能力。`Project_v5` 已经沉淀了 V7 News-Driven Hot Theme Hunter 的设计。新项目不重写所有能力，而是建立一个更清晰、更适合当前策略的专用工作台。

## 2. 产品目标

HotThemeRotator 的目标是每天或盘中回答四个问题：

1. 当前市场温度是否适合出手。
2. 哪些主题正在升温，哪些主题已经退潮。
3. 每个热点主题里真正的龙头是谁。
4. 当前持仓应该止盈、止损、继续持有还是换仓。

## 3. 非目标

- 不做毫秒级或微秒级高频交易。
- 不做自动实盘下单。
- 不把 A股和美股第一阶段纳入直接交易执行。
- 不把 LLM 判断作为唯一买入依据。
- 不绕过回测、paper 和治理门槛直接扩大仓位。

## 4. 核心用户流

### 4.1 开盘前

系统生成市场温度简报，包括日股整体温度、外部市场温度、热点主题、候选龙头、风险提示和今日可观察清单。

### 4.2 盘中

系统更新价格、成交额、主题热度和持仓状态，输出人工可执行的建议：

- BUY candidate
- HOLD
- TAKE_PROFIT
- STOP_LOSS
- ROTATE
- NO_TRADE

### 4.3 收盘后

系统记录当天信号、实际成交、退出原因、策略偏差和复盘要点，用于后续回测与参数冻结。

## 5. 系统架构

```text
data adapters
  -> normalized store
  -> market temperature engine
  -> theme detector
  -> leader ranking
  -> signal engine
  -> portfolio/risk layer
  -> execution advice
  -> daily report and project status

decision flow (parallel sink, P9 automation gates):
  opportunity scanner ─┐
                       ├─> decision log         (reports/predictions/, P9-01)
  attribution baseline ┘            │
                                    v
                              outcome join      (reports/outcomes/,    P9-02)
                                    │
                                    v
                              calibration       (calibration.reporter, P9-03)
                                    │
                                    v
                              human alerts      (alerts.human_alerts, P9-04)
                                    │
                                    v
                              paper trading     (P9-05, future)

user-facing surface (P8-09 onward):
  Python data layer (src/hot_theme_rotator/**)
        │
        v
  api/ (FastAPI, read-only JSON over decision_log + calibration + scanner)
        │
        ├─> frontend/ (Vite + React V3 dashboard, served at /)
        │
        └─> tools/streamlit_opportunity_app.py (legacy fallback at :8501)

upstream live-data inflow (P8-10..P8-15 onward; ADR-0005):
  Project_optimized/japan_market.db (positions / daily_prices / news_feed /
    cross_asset_snapshots / factor_signals / decision_journal / ...)
  Project_optimized/reports/paper_trading_account.json
  Project_optimized/reports/selected_tickers.json
  Project_optimized/universe.json
        │ (read-only)
        v
  src/hot_theme_rotator/data/{position_adapter, kline_adapter, market_temp_adapter,
                              theme_heat_adapter, news_adapter, universe_adapter}.py
        │
        v
  api/serializers.py fills V3 JSON shape (positions / markets / themes / news /
    kline / candidates) with real data; mock fallback only when adapter raises.
```

## 6. 模块职责

### 6.1 Data

统一读取日股行情、新闻、旧项目 SQLite 数据、外部指数、汇率和跨市场板块数据。第一阶段优先读取本地已有数据，不强依赖新付费 API。

**P8-10 起（ADR-0005）扩展为对 `Project_optimized` 全面只读消费**：

- `position_adapter.py` — 读 `Project_optimized/japan_market.db` 的 `positions` + `account_snapshots` 表（按 `strategy_id` 过滤，默认 `etf_buyhold` = 用户 Path A live），返回 `PortfolioState`（cash / NAV / 当前每只 symbol 的 qty / avg_cost / market_price / unrealized_pnl / asof / positions_asof）。**不使用** `paper_trading_account.json`（该 JSON 仅记录已下线 `sprint` 策略的 3041.T 旧快照）。
- `kline_adapter.py` — 读 `japan_market.db.daily_prices` 表，按 symbol + window 返回 `PriceBar[]`；同时实现 P9-02 的 `PriceFetcher` Protocol（即 `LegacyDailyPriceFetcher`）。
- `market_temp_adapter.py` — 读 `japan_market.db.cross_asset_snapshots` 表合成 6 市场温度（日经 / TOPIX / SOX / S&P / USDJPY / 上证）含 sparkline 尾部。
- `theme_heat_adapter.py` — 读 `factor_signals` + `signals` 表，按主题聚合 heat + 动量、排序输出 top N。
- `news_adapter.py` — 读 `news_feed` / `news_items` / `news_sentiment` 表，按时间倒序 + weight 标记输出时间线。
- `universe_adapter.py` — 读 `Project_optimized/universe.json`（951 标的）+ `selected_tickers.json`（当日 top N 短名单），驱动真实候选 scanner（P8-15）。

**规则约束**：HotThemeRotator 永不写回 Project_optimized；adapter docstring 写明依赖的列；schema 漂移由集成测试守门。

### 6.2 Market Temperature

计算市场是否适合交易。评分由以下部分组成：

- 日经/TOPIX 强度。
- 成交额扩张。
- 涨幅扩散。
- 热点主题数量。
- 外部风险偏好，包括美股、A股、VIX、USDJPY。
- 退潮信号，包括冲高回落、热点断层、指数放量下跌。

输出字段：

```text
market_temperature_score: 0-100
regime: HOT | WARM | NEUTRAL | COLD | RISK_OFF
trade_permission: ALLOW | REDUCE | BLOCK
```

### 6.3 Theme Detection

识别近期正在升温的主题，例如 AI 半导体、机器人、汽车出口、防卫、药品审批、TOB、回购增配、特朗普访华相关中美缓和链条等。

第一阶段采用关键词和规则，第二阶段再引入 embedding 或 LLM 聚类。

### 6.4 Leader Ranking

每个主题只挑 1-3 个候选龙头。评分包括：

- 主题相关性。
- 当日和多日相对强度。
- 成交额放大。
- 流动性。
- 新闻催化强度。
- 是否已经过热。
- 是否更早启动而不是末端跟风。

### 6.5 Signal Engine

把市场温度、主题热度和龙头评分合成为信号。第一阶段只输出建议，不写入真实订单。

信号字段：

```text
symbol
theme_id
action
entry_score
target_profit_pct
take_profit_price_2pct
take_profit_price_3pct
take_profit_price_5pct
stop_loss_price
max_hold_days
reason_codes
```

### 6.6 Portfolio And Risk

控制仓位、换仓、止盈和止损。默认规则：

- 单票初始建议仓位不超过 NAV 15%。
- 单主题总暴露不超过 NAV 40%。
- 目标利润区间 2%-5%。
- 固定止损初始为 -3% 到 -5%，以后由回测校准。
- 市场温度降为 COLD 或 RISK_OFF 时禁止新增买入。

### 6.7 Execution Advice

复用 `Project_optimized/intraday_decision.py` 的安全思想：系统只输出建议单，人工在 SBI/IBKR 执行。建议单必须写明有效期、价格参考、止盈止损和原因。

### 6.8 Reporting

输出三类报告：

- daily briefing：开盘前和盘中简报。
- signal sheet：人工执行建议。
- review report：收盘后复盘。

### 6.9 Decision Log

承载 §8.6 mandatory feedback log 与 §10 gate 3 (Decision Logging)。

职责：

- 定义唯一 `PredictionRecord` schema，覆盖 attribution 与 opportunity 两类预测产出。
- 提供 fail-closed 的 JSONL 追加写入接口（缺少必填字段、`available_ts > decision_cutoff`、重复 `prediction_id` 都直接抛错）。
- 持久化路径固定为 `reports/predictions/`，每个交易日一个 JSONL 文件。
- 生成稳定可重现的 `prediction_id`：`sha256(input_snapshot_id || model_version || decision_cutoff || symbol)[:16]`。

不做：

- 校准计算（属 P9-03）。
- 实时报警（属 P9-04）。
- 任何执行动作（违反 Rule 3 advice-only）。

P9-02 起，本子系统同时承担 §10 gate 4 outcome join：

- `decision_log.outcome_join.compute_outcome(prediction, fetcher, evaluated_as_of)` 把每条 PredictionRecord 与 cutoff 之后的历史 OHLC 关联，输出 `OutcomeRecord`。
- `PriceFetcher` 是 Protocol 抽象，生产用 `LegacyProjectAdapter` 或 yfinance 客户端，测试用 in-memory stub。
- 状态分支显式：`complete` / `insufficient_data` / `symbol_not_found` / `future_cutoff` / `malformed_data`（P0-04 加入）。
- 七档 ladder 触达事件按 below-tier (low ≤ tier) 与 above-tier (high ≥ tier) 分别检测；缺一档即 `malformed_data`。
- 写入 `reports/outcomes/{trade_date}.jsonl`，与 predictions/ 平行。

### 6.10 Calibration

承载 §10 gate 5（Calibration）。在 §8.6 decision log 和 §10 gate 4 outcome join 之上，把研究分映射为校准胜率证据。

职责：

- `calibration.calibrator` 提供数学原语：`compute_brier_score`、`compute_log_loss`、`compute_calibration_bins`（等宽 10 bin，最后一档含右闭区间）、`derive_opportunity_ground_truth`（bullish-only：horizon 收益 > 0 → 1，否则 0）。
- `calibration.reporter.build_calibration_report` 用 `prediction_id` 配对 PredictionRecord + OutcomeRecord，跳过未匹配或非 `complete` 的样本，按 horizon (1D/3D/5D) 评估，输出 `CalibrationReport`。
- `CalibrationReport` 含 `source`（opportunity / attribution）、`horizon_days`、`sample_count`、`status`、`brier_score`、`log_loss`、`bins`。
- `status='calibrated'` 必须 `sample_count >= min_samples_required`（默认 100）且 `brier_score` 与 `log_loss` 均存在；`status='insufficient_calibration'` 不允许携带任何数值（§9.4：不达样本不能贴胜率）。
- P8-05 adds `calibration.ladder_feedback` as an opportunity-specific evaluator on top of P9-01/P9-02/P9-03. It pairs `PredictionRecord` and `OutcomeRecord` by `prediction_id`, keeps only complete opportunity outcomes with all seven ladder tiers, and reports per-tier sample counts / touched counts. Numeric per-tier touch rates remain hidden until the tier reaches `min_samples`; tier touch rate is level-touch evidence, not a win rate, and does not change score labels or execution gates.

不做：

- 实时报警（属 P9-04）。
- paper 交易（属 P9-05）。
- 不静默回写底层模型参数 — 校准结论用于决策展示，参数变更仍走 Rule 4 流程。

### 6.10a Human Alerts

P9-04 adds `alerts.human_alerts` as a research-only alert record builder. It compares a current price with the seven Rule 9.3 ladder levels and emits local `AlertRecord` objects for crossed watched levels. Entry and stop levels trigger when current price is at or below the level; exit levels trigger when current price is at or above the level.

Alert records carry `research_only=True`, `data_ts`, reason, risk warning, deterministic `alert_id`, and no broker/order fields. Duplicate suppression is handled before any user-facing channel consumes the records. This module does not send external notifications, create paper trades, or place broker orders.

### 6.11 API

承载用户面 UI 与 Python 数据层之间的只读 JSON 契约（P8-09 起；ADR-0004）。

职责：

- 由 FastAPI 实现，端口 `8000`。
- 主端点 `/api/dashboard` 返回 V3 仪表盘所需完整 JSON 形状（`meta` / `gates` / `markets` / `themes` / `candidates` / `newsTimeline` / `decisionLog`）。
- 派生数据 100% 来自 `decision_log/` / `calibration/` / `opportunity/` / `ui/opportunity_dashboard._GATE_DEFINITIONS`，不重算评分、不重派生 ground truth、不重做校准。
- **只读**：API 不接受任何 POST/PUT/DELETE，不暴露执行通路 — Rule 3 在 API 层显式锁死。
- 生产模式：FastAPI 同时挂载 `frontend/dist/` 静态资产作为 `/`；开发模式仅 API，前端走 Vite dev server。

### 6.12 Frontend

承载 V3 市场温度仪表盘（P8-09 起；ADR-0004，Phase 1 = zero-build）。

职责：

- React 18 单页应用，**Phase 1 不引入 npm/Vite**：React + Babel-standalone 通过 CDN 加载，JSX 浏览器内编译。源文件 `frontend/index.html`、`frontend/shared.jsx`、`frontend/v3.jsx`、`frontend/data.js`（从 `quant.zip` 原样拷贝 + 一个新 `index.html` 完成 fetch+mount）。
- 数据源主路径：`fetch("/api/dashboard")` 合并到 `window.HTR_DATA` 之上后 mount `<V3MarketDashboard />`。`data.js` 中的 mock 作为 markets/themes/newsTimeline/kline 等 Python 层尚未供给字段的 fallback。
- 渲染 §10 八阶 gate flow、6 市场温度 mosaic、主题热力 treemap、SVG 七档纵向阶梯、新闻时间线、决策日志、Top 候选 hero。
- 校准 badge 渲染 `meta.calibration.text`（由后端按 §9.4 已强制 sample < threshold 时为 `insufficient_calibration` 标签，前端不重写）。
- 不引入任何"下单"按钮或表单提交 — Rule 3 在前端层重申。
- Phase 2（未来，P8-10 候选）：当页面加载延迟、HMR 或共享组件库重要时迁移到 Vite + ES 模块。

**P8-09 Cycle 6 起 — app shell + 变体导航**：`frontend/index.html` 中 `App()` 是 sticky 顶 nav + `<main>` 全屏渲染当前 active 变体的产品级 shell；用户选 (`localStorage.htr_variant`, 默认 V3) 切换 V1/V2/V3/V4；`<DesignCanvas>` / `<DCSection>` / `<Rationale>` 仅作设计师离线工具保留，不进产品。

**P8-16 起 — 顶 nav 子组件**：
- **§10 gate chip + modal**：app nav 右侧渲染 `<button class="gates-chip">` 显示真实 done/blocked 计数（如 "5/8 ✓ ⛔1"），点击弹 `.gates-modal` 含完整 `<GateFlow>`，ESC 或 backdrop 关闭。4 变体内部不再各自渲染 GateFlow。
- **`<Term k="key">` tooltip 子系统**：`shared.jsx` 含 `Term` 组件 + `GLOSSARY` 字典（40+ 术语：七档阶梯/未校准研究分/Brier/12 alpha factor/markets/positions/§10/§8.6/§9.4/screener/dashboard sections）。`<Term>` 渲染下划线虚线点提示，鼠标悬停/键盘 focus 显示深色 tooltip 含**术语名 + 中文白话定义**。**§9.4 红线坚守**：所有定义都不说"高分=高胜率"。变体里术语字串包 `<Term>` 即获得 tooltip。
- **V1 KLineChart 强化**（`shared.jsx`）：默认开启 `withVolume`（底 25% 成交量柱图，up=绿/down=红 半透明）+ `withMA`（MA20 蓝实线 + MA60 橙虚线，内联 rolling mean）+ `with52wLines`（252 session 高/低水平参考线）。配合 `api/serializers._serialize_kline(sessions=252)` 提供 1 年窗口让 MA60 + 52w 有意义。
- **P8-17 V1 价格卡片空间规则**：V1 的 `价格走势 · 七档阶梯` 卡片不再把七档文字标签嵌入 KLineChart 的 SVG 右侧 padding。卡片内部采用左侧 K 线主图 + 右侧七档侧栏布局；KLineChart 可继续渲染 OHLC、成交量、MA20/MA60、52w 线，但七档的可读标签由 V1 专用侧栏负责，避免卡片放大后信息仍挤在右侧小区域。
- **P8-17 V2 paper-surface 规则**：V2 研究备忘录的下半区内容块必须有明确白色 paper surface、边框和空状态。`新闻催化与决策日志` 区域中，新闻时间线和 §8.6 决策日志分别放入同规格 pane；当 `decisionLog` 为空时显示研究日志尚未生成的空状态，不留出看似漏渲染的裸背景。V2 的外层 flex 容器必须 `alignItems: "flex-start"`，避免白色 paper 容器被默认 stretch 成一屏高，滚动到 Section C/D 后露出 body 背景色。

## 7. 数据流

1. 从旧项目或外部数据源拉取价格、新闻、指数和持仓。
2. 标准化为统一 schema。
3. 计算市场温度。
4. 识别主题并计算主题热度。
5. 对主题内股票进行龙头排序。
6. 生成建议和风控动作。
6b. 把每条候选预测（opportunity scanner 与 attribution baseline 两路）写入 decision log（`reports/predictions/`），附 `prediction_id`、`input_snapshot_id`、`model_version`、`decision_cutoff` —— §8.6 / §10 gate 3 强制要求。
7. 写入报告和复盘记录。

## 8. 验收标准

第一阶段完成后，系统必须能在不触碰实盘下单的情况下生成：

- 当日市场温度。
- 最热 5 个主题。
- 每个主题 1-3 个候选龙头。
- 每个候选的入场理由、止盈价、止损价。
- 当前持仓的继续持有、止盈、止损或换仓建议。

## 9. 迁移原则

- 只迁移稳定能力，不迁移旧项目中的混乱状态。
- 对旧代码先写适配层，不直接复制大型脚本。
- 每次迁移必须有测试或样例输入输出。
- 旧项目继续保持不变，直到新项目 paper 结果证明可替代。
