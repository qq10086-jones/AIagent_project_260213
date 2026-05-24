# Governance

## 1. Absolute Rules

### Rule 1: One Status File

`PROJECT_STATUS.md` 是唯一项目更新文件。任何进度、阻塞、风险、下一步、阶段完成声明，只能写入该文件。

禁止新增：

- `PROGRESS_*.md`
- `STATUS_*.md`
- `daily_notes.md`
- 临时总结文件
- 只存在聊天记录中的项目状态

### Rule 2: One Task Source

所有工作必须先出现在 `docs/01_TASKS.md`。如果一个改动无法挂到已有任务，先新增任务，再改文件。

### Rule 3: Advice-Only Until Gates Pass

未通过 paper 验证前，本项目不得自动下单，不得调用真实券商下单接口。输出只能是人工可读的建议。

### Rule 4: No Silent Parameter Changes

任何策略参数修改必须记录：

- 修改字段。
- 原值。
- 新值。
- 修改原因。
- 预期影响。
- 验证方式。

记录位置是 `PROJECT_STATUS.md` 的 Change Log。

### Rule 5: No Backtest Without Costs

任何回测必须包含手续费、滑点、最小交易单位、买卖价差和换仓成本。没有成本模型的结果只能标为 research draft。

## 2. Change Workflow

每次修改必须按以下顺序：

1. Read `PROJECT_STATUS.md`.
2. Find or add task in `docs/01_TASKS.md`.
3. Confirm the target folder in `docs/03_FOLDER_MAP.md`.
4. Make the smallest coherent change.
5. Add or update tests when code behavior changes.
6. Run verification command.
7. Update `PROJECT_STATUS.md`.

## 3. Design Change Policy

设计文档不是流水日志。只有以下情况才修改 `docs/00_DESIGN.md`：

- 策略边界变化。
- 模块职责变化。
- 数据流变化。
- 风控原则变化。
- 实盘权限变化。

普通进度不写入设计文档。

## 4. Folder Ownership

- `src/hot_theme_rotator/data`: 数据接入和标准化。
- `src/hot_theme_rotator/market_temperature`: 市场温度。
- `src/hot_theme_rotator/theme_detection`: 主题发现。
- `src/hot_theme_rotator/leader_ranking`: 龙头排序。
- `src/hot_theme_rotator/signal_engine`: 信号生成。
- `src/hot_theme_rotator/risk`: 风险控制。
- `src/hot_theme_rotator/execution_advice`: 人工执行建议。
- `src/hot_theme_rotator/reporting`: 报告和复盘。

模块之间通过 schema 传递数据，不允许跨模块读取彼此内部状态。

## 5. Verification Gates

### Research Gate

允许生成研究报告，但不得生成实盘建议。

要求：

- schema 测试通过。
- 数据样例可复现。
- 无未来函数。

### Paper Gate

允许生成 paper 建议。

要求：

- 最近 60 个交易日回放可运行。
- 输出包含交易成本。
- 最大单笔亏损和最大回撤可统计。

### Live Advice Gate

允许生成用户手动执行建议。

要求：

- 至少 4 周 paper 记录。
- 胜率、盈亏比、平均持仓时间、最大亏损均可解释。
- 用户明确确认仍然保持 advice-only。

### Auto Execution Gate

当前禁止。未来如要启用，必须新建设计文档、任务清单、风控 ACK 和人工审批。

## 6. Documentation Rules

- `PROJECT_STATUS.md`：动态状态。
- `docs/00_DESIGN.md`：稳定设计。
- `docs/01_TASKS.md`：任务定义。
- `docs/02_GOVERNANCE.md`：规则。
- `docs/03_FOLDER_MAP.md`：目录职责。
- `docs/04_DATA_AND_OPEN_SOURCE.md`：数据和外部项目选型。
- `docs/adr/*.md`：重大架构决策，只记录不可逆或高影响决策。

## 7. Definition Of Done

A task is only done when the target files are updated, verification has been run, failures or blockers are recorded in `PROJECT_STATUS.md`, and the advice-only constraint remains intact.

## 8. Universal Attribution And Probability Rules

`1306.T` is only an example instrument. These rules govern any report that explains or scores representative instruments, themes, sectors, ETFs, or single stocks.

The goal is not a one-symbol explainer. The goal is a reusable attribution engine that compares multiple representative instruments, learns which factors matter across regimes, and integrates the symbol-level evidence into one cross-symbol decision view.

### Rule 8.0: Representative Universe Comes Before Symbol Analysis

Every attribution run must define a representative universe before scoring any one symbol.

The first Japan universe must include at least these roles:

- broad beta proxy
- export cyclical
- rate-sensitive financial
- AI or semiconductor growth
- domestic defensive
- commodity or energy sensitivity
- external-risk growth

`1306.T` may be one member of the universe, but it must not be treated as the whole framework.

### Rule 8.1: Separate Ex-Ante From Ex-Post

Every instrument-level daily analysis must split two views:

- `ex_ante`: information available at the decision cutoff only. This may feed buy, sell, or hold probability estimates.
- `ex_post`: information known after the close. This may explain the actual move, but must not be used as that day's decision input.

Same-day close price, post-close news, revised factors, or future returns are forbidden in `ex_ante` inputs.

### Rule 8.2: Point-In-Time Data Is Mandatory

Every news item, price factor, macro factor, and external-market factor used for `ex_ante` scoring must carry an `available_ts`.

The system must fail closed when an input lacks `available_ts` or when `available_ts` is later than the decision cutoff. The report must show the cutoff used for each trading day.

### Rule 8.3: LLMs Cannot Invent Win Probabilities

`buy_win_prob`, `sell_win_prob`, and `hold_win_prob` may only be shown as probabilities when produced by a calibrated historical model with an explicit `model_version`.

If calibration is not available, the output must use one of these labels instead:

- `uncalibrated_research_score`
- `insufficient_calibration`

The default evaluation horizon is 3 trading days. Auxiliary horizons may include 1 trading day and 5 trading days.

### Rule 8.4: Daily Move Definition Is Required

Each trading day in a weekly attribution report must include a plain movement label, for example:

- `risk_off_drop`
- `broad_rebound`
- `rate_pressure_selloff`
- `topix_beta_follow`
- `fx_tailwind`
- `external_tech_drag`
- `low_signal_noise`

The label must be backed by factor evidence. A report cannot rely on narrative text alone.

### Rule 8.5: Attribution Buckets Are Fixed

The first version of universal attribution must score and report these buckets for every representative instrument:

- Japan equity beta: TOPIX proxy, Nikkei proxy, breadth, and broad index direction.
- Rates: JGB yield or rate-pressure proxy when available.
- FX: USDJPY direction and magnitude.
- External risk: US equities, global rates, technology or AI shock, and geopolitical risk when available.
- News: point-in-time news only.
- Own trading behavior: open, high, low, close, volume, gap, intraday range, and realized return.

Missing buckets must be reported as missing, not silently ignored.

### Rule 8.6: Feedback Log Is Mandatory

Every instrument-level prediction must be logged with:

- `prediction_id`
- `symbol`
- `trade_date`
- `decision_cutoff`
- `input_snapshot_id`
- `model_version`
- probability or score status
- buy, sell, and hold outputs

After outcomes are known, the system must attach realized 1D, 3D, and 5D returns. Calibration metrics such as Brier score, log loss, and calibration bins may only be reported when the sample size is sufficient.

### Rule 8.7: Cross-Symbol Integration Is Required

Final output must mathematically integrate symbol-level outputs across the representative universe.

The initial integration method is a role-weighted average over buy, sell, and hold outputs. A final output may be called `calibrated_probability` only when all contributing symbol-level outputs are calibrated probabilities. Otherwise, it must remain `uncalibrated_research_score` or `insufficient_calibration`.

### Rule 8.8: Advice-Only Remains In Force

Universal attribution and probability reports are research outputs. They must not trigger live orders or auto execution. Moving from research output to live advice requires the existing gates plus explicit human approval.

## 9. Realtime Opportunity And Price Ladder Rules

The user's target system is a real-time opportunity engine, not only a post-trade attribution report. `1306.T` is a worked example for the analysis style; the engine must search across candidate stocks and ETFs.

### Rule 9.1: Search Before Scoring

The system must first discover potential stocks from a broad candidate universe using point-in-time inputs:

- theme/news catalysts
- price momentum
- volume expansion
- relative strength
- liquidity
- market, sector, FX, rates, and external-risk context

Fixed single-symbol analysis is not sufficient for P8 outputs.

### Rule 9.2: Real-Time Inputs Must Be Point-In-Time

Every live or intraday candidate input must carry `available_ts`. Inputs later than the decision cutoff are forbidden. The system must fail closed rather than silently using stale, missing, or future data.

Default refresh intervals for Japan equities:

- pre-open 08:00-09:00 JST: 10 minutes
- morning session 09:00-11:30 JST: 3 minutes
- lunch break 11:30-12:30 JST: 15 minutes
- afternoon session 12:30-15:30 JST: 3 minutes
- post-close 15:30-23:00 JST: 3 hours
- overnight 23:00-08:00 JST: 6 hours
- material news event: immediate recalculation

### Rule 9.3: Price Ladders Are Required

A candidate recommendation must include staged price levels, not only BUY or SELL:

- aggressive entry
- balanced entry
- conservative entry
- stop price
- first exit
- second exit
- stretch exit

Each level must include the formula or source factors used to calculate it.

### Rule 9.4: Win Rate Requires Calibration

Before feedback calibration exists, the system may only output `uncalibrated_research_score` or `insufficient_calibration`. It must not label these numbers as true win rates.

Calibrated win rates require logged predictions, realized 1D/3D/5D outcomes, and enough sample size for calibration metrics.

### Rule 9.5: Candidate Panel Is The First P8 Product

The first runnable P8 product is a ranked candidate panel. It must show:

- rank
- symbol
- trigger theme
- opportunity score
- score status
- staged buy prices
- stop price
- staged sell prices
- reason codes
- data gaps

The panel is research-only and must not create orders.

### Rule 9.6: UI Must Preserve Ladder And Log Legibility

The user-facing dashboard may change visual layout, but it must not hide the required decision context.

For candidate price views:

- the seven ladder levels from Rule 9.3 must remain visible as seven distinct levels;
- layout changes may move ladder labels outside the chart, but they must not drop any level;
- chart whitespace fixes must not imply a new score, new probability, or execution readiness.

For decision-log views:

- an empty §8.6 decision log must show an explicit empty state;
- an empty log must not look like a broken or missing UI region;
- the empty state must not say calibration or paper trading is ready before the relevant §10 gates pass.

### Rule 9.7: Ladder Feedback Is Evidence, Not A Win Rate

P8-05 ladder feedback must consume the shared P9-01 prediction log and P9-02 outcome records. It must not create a second storage format for ladder outcomes.

For each Rule 9.3 ladder tier, feedback may report sample counts and touched counts. A numeric touch rate may only be exposed when that tier has enough complete matched samples. Before the threshold is reached, the tier remains `insufficient_calibration`.

Tier touch rate is not a win rate. It must not be labeled as probability of profit, execution readiness, or trading advice. Ladder feedback must not trigger alerts, paper orders, broker orders, or any other execution path.

## 10. Automation Gate Rules

Automation is a staged product path, not a shortcut to live trading. The system may only advance through these gates in order:

1. Candidate discovery from a broad, point-in-time symbol universe.
2. Ranked opportunity panel with staged buy, stop, and sell ladders.
3. Decision logging for every candidate row, including model version, input snapshot, reasons, data gaps, and generated ladder.
4. Feedback joining with realized 1D, 3D, and 5D outcomes.
5. Calibration with sufficient sample size before any output is called a win rate.
6. Human-readable alerts for watched levels.
7. Paper trading with position limits, kill-switches, and audit logs.
8. Broker execution only after explicit human approval and a passed paper gate.

Until gates 3-5 are implemented and verified, all interface output must remain `uncalibrated_research_score` or `insufficient_calibration`.

Alerts may notify a human, but they must not place orders. Paper trading may simulate orders, but it must not call live broker APIs. Live broker execution is forbidden unless a future governance update explicitly enables it and records the approval in `PROJECT_STATUS.md`.

### Rule 10.1: Human Alerts Are Not Orders

P9-04 alerts may describe that a watched level was crossed, but the alert payload must not contain broker, account, route, quantity, notional, order type, or submit fields.

Every alert must carry `research_only=True`, a data timestamp, a reason, and a risk warning. Duplicate alerts for the same symbol, level, and trade date must be throttled before any user-facing channel consumes them.

## 11. Read-Only Interactivity Rules

Rule 3 forbids execution endpoints. It does not forbid all interaction. Most dashboard interactions — switching symbols, browsing K-line history, recomputing a ladder against a chosen reference price, building a personal watchlist — are purely exploratory and never touch the execution path. They must be allowed, because without them the dashboard is a static report, not a research tool.

### Rule 11.1: Allowed Interactions

The user-facing surface MAY:

- Switch the displayed symbol (K-line, news, ladder, profile follow the active symbol).
- Adjust the time window or session count on any read-only chart.
- Hover, focus, or click for tooltips, drill-downs, and on-demand detail.
- Maintain a local watchlist persisted in `localStorage`.
- Add private notes on any candidate or holding (`localStorage` only — never written to `decision_log/`).
- Sort, filter, or re-rank the candidate panel by any visible column.
- Recompute the seven-tier ladder against a user-supplied reference price (the recomputation is a deterministic function of inputs; nothing is persisted).
- Compare two or more candidates side by side.

### Rule 11.2: Forbidden Interactions

The user-facing surface MUST NOT:

- Send any POST / PUT / DELETE / PATCH request — Rule 3 stays absolute.
- Persist user input into `reports/predictions/`, `reports/outcomes/`, `japan_market.db`, or any other system-of-record store.
- Override an algorithmic `score`, `score_status`, `calibration` label, or `gate` state with a user-supplied value.
- Trigger any alert, paper trade, or broker call from a UI click.
- Imply that the user's local view ("I am watching this") is the system's view ("this is a recommended position").

### Rule 11.3: User-State vs System-State Separation

Two stores, never mixed:

- **User state** — preferences, watchlist, notes, last-viewed symbol, theme tweaks. Lives in `localStorage` / cookies. Disposable; clearing it never changes a system fact.
- **System state** — predictions, outcomes, calibration reports, positions, candidates, gate status. Lives in `reports/`, `japan_market.db`, and Python data layer. Authoritative; UI reads but never writes.

If a future feature needs to persist user choices server-side (e.g., a shared watchlist), it must land as a separate `user_state/` store with its own ADR and explicit non-execution scope, and it must not share schema with `decision_log/`.

### Rule 11.4: Interaction Does Not Lift Calibration Status

A user clicking, hovering, or starring a candidate does not change its `score_status`. An uncalibrated score remains `uncalibrated_research_score` whether the user has watched it for 1 second or 1 month. Rule 9.4 still applies; UI interactions cannot launder a research label into a hidden recommendation.

一个任务只有同时满足以下条件才能标记 done：

- 对应文件已创建或修改。
- 验证命令已运行。
- 输出或失败原因已写入 `PROJECT_STATUS.md`。
- 没有绕过 advice-only 约束。
