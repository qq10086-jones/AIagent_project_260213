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

### 2.1 External Adapter Fixture Provenance (post Codex review 2026-05-25)

When adding a new external data adapter (scraper / API client / RSS reader / etc.):

- Fixtures MUST be captured from a real source response, not invented from assumptions about the shape.
- Captured fixtures live at `tests/fixtures/captured/{source_name}/{date}_{description}.{ext}` with a comment noting the capture date + URL.
- If the source's ToS prohibits storing responses (or the response contains personal data), instead document the shape in a `tests/fixtures/{source}/SCHEMA.md` and only the shape signature may be invented.
- Acceptance for any external-adapter task MUST include "live smoke test passed against real endpoint" before the task is marked `done` (not just `Cycle 1 done`).

Rationale: P10-14 TDnet parser had a shape mismatch caught only in live smoke after Cycle 1 was nominally "done" — fixtures had been invented based on documentation, not real captures. Real Yanoshin wraps items in `{"Tdnet": {...}}` with space-separated `pubdate`; my fixture used flat dict with ISO `pubdate`. This was avoidable via a one-time live capture.

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

### Rule 8.2.1: Backdated Calibration Exception (one-time bootstrap)

Per `docs/adr/ADR-0006-backdated-calibration.md`, the system MAY generate `PredictionRecord` instances from historical snapshots without enforcing strict per-input `available_ts > decision_cutoff` at the operator level, IF AND ONLY IF every condition below holds:

- `PredictionRecord.extra["backdated"] = True`.
- `PredictionRecord.extra["live"] = False`.
- `model_version` carries the suffix `-backdated`.
- The generation tool is `tools/backdated_calibration_bootstrap.py` and no other module.
- For each synthesized prediction with `decision_cutoff = D`, every input feature's `available_ts` strictly predates `D`, and the matched outcome window uses only `daily_prices` bars whose `asof > D` by at least 1 trading day.
- The scanner config hash used by the bootstrap tool corresponds to a `configs/scanner.yaml` commit that exists in `git log` on or before the bootstrap window start date.
- The bootstrap tool emits a `bootstrap_provenance.json` recording window start/end, total snapshots considered, snapshots excluded with reason, model_version, and scanner config hash.

Bootstrap predictions are research evidence only. They MUST NOT inform any new forward trade, paper trade, alert, or live order.

Calibration reports derived from bootstrap predictions carry `evidence_origin="bootstrap"`. UI surfaces MUST visually distinguish bootstrap-origin calibration from live-origin calibration.

Sunset: the first day live evidence_origin reaches `min_samples_required`, dashboard calibration display MUST switch to live-only. Bootstrap reports are retained on disk for audit but stop influencing the UI badge.

Rule 8.3 and Rule 9.4 are NOT relaxed. Bootstrap evidence must still satisfy `sample_count >= min_samples_required` and present `brier_score` / `log_loss` before any number may be labeled a win rate.

### Rule 8.3: LLMs Cannot Invent Win Probabilities

`buy_win_prob`, `sell_win_prob`, and `hold_win_prob` may only be shown as probabilities when produced by a calibrated historical model with an explicit `model_version`.

If calibration is not available, the output must use one of these labels instead:

- `uncalibrated_research_score`
- `insufficient_calibration`

The default evaluation horizon is 3 trading days. Auxiliary horizons may include 1 trading day and 5 trading days.

### Rule 8.3.1: LLM Narrative Synthesis Carve-out

Rule 8.3 forbids LLMs from inventing probabilities. It does NOT forbid LLMs from producing **narrative synthesis** — combining news, factors, fundamentals, and technicals into a Chinese research note for a single ticker (P10-06).

When LLM is used for narrative synthesis (e.g., `llm.per_ticker_brief`):

- Output schema MUST contain only `narrative` (markdown), `factual_grounding` (input citations), `model_version`, `generation_ts`. No probability / win-rate / score field.
- Default local model is `gemma4:e4b`. Cloud models are forbidden per existing user preference.
- Post-generation regex MUST reject `\d+%` or any of "胜率" / "概率" / "win rate" / "probability" / "likelihood" appearing in the narrative. First violation triggers one regeneration; second violation fails closed.
- Ollama / model unreachable → 503 + reason; never fabricate a brief.
- 24h cache per (ticker, input_hash) — required to respect single-GPU/VRAM constraint (one user, one card).

Anyone adding a new LLM call must satisfy Rule 8.3.1 in addition to Rule 8.3.

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

### Rule 8.6.1: Sample Frequency Display Carve-out

Rule 8.6 / 9.4 forbids displaying any number as a calibrated win rate before `sample_count >= min_samples_required`. This does NOT mean evidence must be hidden entirely. Raw sample frequency may be displayed in advisory UI under strict conditions:

- Minimum n: at least 10 paired (prediction, outcome) samples in the bucket being displayed. Below 10 — display nothing.
- Form: `n=23, %touched=56.5% — 未校准, 非胜率` (or English equivalent). Always show both n and percentage; never percentage alone.
- Label: every numeric frequency MUST carry an explicit `uncalibrated_evidence` pill. The words "胜率" / "win rate" / "probability" / "概率" / "likelihood" MUST NOT appear next to the number.
- No confidence intervals, no standard errors, no Bayesian posteriors — these imply statistical inference beyond what raw frequency supports.
- Sample bucket MUST be machine-defined (e.g., "all symbols where mom_20 z-score > 1.5, count of 3D positive returns"), not narratively described, so the bucket definition is auditable.

This carve-out is for evidence transparency in advisory UI. The win-rate labeling rule (8.3 / 9.4) remains absolute and unchanged. The carve-out admits the operational reality: a personal advisory user is making decisions every day; hiding all evidence below the calibration threshold serves discipline but leaves the user in the dark; showing labeled raw frequency serves discipline AND honesty.

Sample purity: post cutover (ADR-0008), journal entries with `source ∈ {paper, migration, correction}` MUST be excluded from the raw-frequency sample bucket (see Rule 14.6).

### Rule 8.7: Cross-Symbol Integration Is Required

Final output must mathematically integrate symbol-level outputs across the representative universe.

The initial integration method is a role-weighted average over buy, sell, and hold outputs. A final output may be called `calibrated_probability` only when all contributing symbol-level outputs are calibrated probabilities. Otherwise, it must remain `uncalibrated_research_score` or `insufficient_calibration`.

### Rule 8.8: Advice-Only Remains In Force

Universal attribution and probability reports are research outputs. They must not trigger live orders or auto execution. Moving from research output to live advice requires the existing gates plus explicit human approval.

### Rule 8.9: Cross-Strategy Advisory Discipline

When advisory recommends an action that touches a position belonging to a different strategy (e.g., HotThemeRotator suggests acting on a position held under `etf_buyhold`), the recommendation MUST:

- Name both strategies explicitly (source strategy + target strategy).
- Show the rebalancing math: which symbol moves, what quantity / value, what NAV % delta on each strategy.
- Acknowledge the strategy-isolation default and explain why this specific recommendation justifies crossing it.
- Be recorded in `reports/meta_strategy_journal.jsonl` with timestamp, source strategy, target strategy, symbol, proposed delta, reason, and the user's decision (accept / reject / defer) when known.
- NOT propose meta-rebalancing more than once per `(symbol, week)` — anti-churn guard.

The cross-strategy journal is research evidence only. It does NOT authorize execution. Live cross-strategy moves still require user manual action through the normal broker interface, and may not bypass §10 gates.

Default position: strategies stay isolated. Cross-strategy advisory is the exception, not the norm. UI may surface the option but MUST NOT lead the user to it (e.g., no "rotate from Path A to HTR" CTA button on dashboard).

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

Sample purity: post cutover (ADR-0008), only journal entries with `source ∈ {manual, import}` feed the calibration sample; `paper`, `migration`, and `correction` entries MUST be excluded per Rule 14.6.

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

## 12. Push Mode Discipline (Anti-FOMO Layer)

When the system proactively notifies the user (P9-04 alerts + P10-10 notification channels + P10-11 scheduled scan + P10-17 watchlist intelligence), the push channel itself becomes a risk surface. Without discipline, push amplifies FOMO ("踏空") trading, increases impulse decisions, and trains the user to react instead of decide.

Section 12 is enforced by P10-18 Anti-FOMO Guard Layer. All six sub-rules below MUST be satisfied before any alert reaches a user-facing notification channel.

### Rule 12.0: Activation Stages for Time-to-First-Value

The system may become useful before the full push / reflection stack is complete, but activation MUST be staged. Fast delivery is allowed only when it reduces interaction risk rather than bypassing discipline.

- **Stage 0 - Pull-only daily cockpit**: the user explicitly opens the dashboard or runs a briefing. It may show holdings, watchlist state, delayed quotes, TDnet disclosures, ladders, data gaps, and research-only warnings. It MUST NOT send notifications or imply calibrated win rates.
- **Stage 1 - Silent alerts**: scheduled scans may generate `AlertRecord` rows and write a dashboard-visible silent queue. No desktop, email, telegram, mobile, or audible notification is allowed in this stage.
- **Stage 2 - Guarded push**: user-facing notification channels may be enabled only after P10-18 enforces Rule 12.1-12.6. Any alert failing the discipline layer stays silent or is downgraded.
- **Stage 3 - Reflection / calibration**: P11 reflection, policy replay, and backdated calibration improve the system but do not block Stage 0 or Stage 1 use. They also do not relax Rule 3, Rule 8.2, Rule 8.3, Rule 9.4, or Section 10.

Implementation tasks MUST state which activation stage they enable. A task may advance the user experience to a later stage only if the acceptance criteria prove the preceding stage remains advice-only and non-executing.

### Rule 12.1: Alert Budget

Per-user daily push budget defaults to 10 alerts. Alerts beyond budget go to a silent queue (visible in dashboard but produce no desktop / email / telegram notification). Budget resets at 06:00 JST. The budget is a configuration value but MUST default conservatively.

### Rule 12.2: Stale Data Fail-Closed

If any upstream data source feeding an alert is more than 2 hours stale relative to wall-clock JST (where the source's natural cadence allows fresher), the alert MUST NOT be pushed. Stale alerts may still be logged but never reach the user. This applies to news, quotes, factor signals, and disclosures. The system MUST NOT fabricate freshness.

### Rule 12.3: Chase Filter

When a candidate's intraday move exceeds +10% (or the JP daily price limit for that price band, whichever is smaller), any BUY alert for that candidate MUST be downgraded to `study_only`. The downgraded alert still surfaces in the watchlist intelligence panel for review, but its severity is reduced and its action verbiage changes to "研究关注, 不建议追涨" / "study only, do not chase".

### Rule 12.4: Cooling-Off for New Watchlist Entries

A symbol newly added to the user's watchlist enters a 24-hour cooling-off window during which BUY alerts for that symbol are silently suppressed (still logged). The intent is to prevent same-day add-then-chase impulse. STOP / TAKE_PROFIT / TAKE_LOSS alerts for existing positions are NOT subject to cooling-off — those are reactive to existing exposure.

### Rule 12.5: Concentration Guard

Before any BUY alert that would push single-name concentration above the configured threshold (default 20% NAV) is pushed, the system MUST first push a separate "concentration warning" describing the projected post-action concentration. The BUY alert is suppressed until either: user dismisses the concentration warning, or 1 hour passes (at which point the BUY alert is silently dropped).

### Rule 12.6: Cross-Strategy Trigger Requires Cross-Strategy Journal

Any alert whose action would touch a position in a strategy different from the source strategy MUST also write a row to `reports/meta_strategy_journal.jsonl` per Rule 8.9, before the alert is pushed. If the journal write fails, the alert is suppressed.

Push discipline applies to all proactive channels — desktop notifications, email, telegram, in-dashboard banner alerts, mobile push. It does NOT apply to user-pulled queries (Pull mode), which the user explicitly initiates.

## 13. Reflection Discipline (P11 enforcement)

P11 reflection system inverts the normal advisory flow: instead of system → user, this layer is system → self. Without strict discipline, reflection becomes a hindsight-bias amplifier or auto-overfitting machine. The rules below MUST be enforced by P11-06 (Human Decision Gate).

Per ADR-0007 (post Codex review): this section does NOT permit Pearl-style causal identification, Shapley-style attribution, or any other terminology that overclaims epistemic authority. Reflection is policy replay under partial observability.

### Rule 13.1: Reflection Outputs Are Proposals

The reflection system's output is always a PROPOSAL, never auto-applied. Any parameter / threshold / config update requires Rule 4 flow + user explicit confirm.

### Rule 13.2: Counterfactual Must Use Real History

P11-03 policy replay MUST use the `P9-02 LegacyDailyPriceFetcher` (or equivalent PIT-safe fetcher) for all historical OHLC. LLM is FORBIDDEN from generating "hypothetical past prices" or "what could have happened" speculation that isn't directly traceable to historical bars.

### Rule 13.3: Sample Size Tiered Triggers

P11-02 trigger requires cumulative samples by purpose:

- `n >= 30` — "investigate" trigger allowed
- `n >= 100` — directional language allowed in proposals ("trend suggests X")
- `n >= 300` OR bootstrap CI established — parameter change proposals allowed

Single-event panic-trigger (e.g., one missed limit-up event) forbidden.

**Statistical basis** (per Codex review 2026-05-25): for binomial outcomes
(hit/miss) at `p ≈ 0.5`, the normal approximation gives sample size ≈ `(z * σ / E)²`
where `σ = √(p(1-p))` ≈ 0.5 and `z = 1.96` for 95% CI:

- `n = 30` → margin ≈ ±18% — sufficient only to flag for investigation
- `n = 100` → margin ≈ ±10% — sufficient for directional language
- `n = 300` → margin ≈ ±6% — sufficient for parameter-change proposals
- `n = 400` → margin ≈ ±5% — sufficient for high-confidence proposals

These tiers are governance discipline, not absolute statistical validity. For
returns (continuous outcomes), use block bootstrap CI instead of normal approx.

### Rule 13.4: Reflection Cannot Launder Calibration

P11-05 LLM report inherits Rule 8.3.1 — no numeric probability output. P11-04 RCA outputs are attribution shares / marginal recovery metrics under explicit intervention semantics, NOT win rates / probabilities / "likelihood".

### Rule 13.5: Proposal Expiry

P11-06 proposal 7 days without accept/reject auto-expires to `reports/reflections/proposals/expired/`. Expired proposals are NOT silently re-issued — they require Rule 13.3 fresh trigger.

### Rule 13.6: Proposal Metadata Required

Every reflection proposal MUST carry:

- `evidence_class` (e.g., `funnel_loss`, `ablation`, `freshness_attribution`, `chase_filter_overshoot`)
- `sample_size`
- `confidence_interval` or `uncertainty_metric` (block bootstrap CI for returns, beta-binomial for hit/miss)
- `counterfactual_validity` ∈ {`exact_replay`, `partial_replay`, `universe_reconstructed`, `price_only_replay`, `invalid`}

Proposals lacking any required field MUST be rejected at P11-06 intake.

### Rule 13.7: Parameter Proposals Require Pre/Post Backtest

Any proposal that suggests changing a parameter / threshold / config MUST include pre/post evaluation on a holdout window OR rolling-origin backtest. The backtest result must be attached to the proposal as evidence. Proposals without backtest evidence MUST be rejected at P11-06 intake.

### Rule 13.8: No Single-Event Proposals

Proposals MUST NOT be based on a single missed limit-up event, single losing trade, or single profitable trade. Rule 13.3 sample size tier minimum (`n >= 30`) applies even for "investigate" triggers.

### Rule 13.9: Rejected Proposals Are Logged

P11-06 MUST log all rejected proposals + rejection reason to `reports/reflections/rejected/{date}/{id}.json`. This log feeds Rule 13.10 meta-reflection.

### Rule 13.10: Meta-Reflection Triggers

Meta-reflection (P11-07) triggers when:

- 3+ consecutive proposals from same `evidence_class` are rejected
- 3+ proposals expire without review
- Post-change monitoring shows post-acceptance proposals failed to deliver predicted improvement

Meta-reflection output: "the proposal generator itself needs adjustment in area X". Proposals from misaligned generators MUST be paused until generator scrutiny passes.

The five hard limits Section 13 does NOT relax: Rule 3 advice-only / Rule 8.2 PIT mandatory / Rule 8.3 LLM no probability / Rule 9.4 calibrated win rate threshold / §10 gate progression.

## 14. Portfolio Ledger Rules (post-cutover, ADR-0008)

These rules become binding on cutover day T defined in ADR-0008, on which HotThemeRotator assumes ownership of portfolio state from Project_optimized. Before T, ADR-0005 read-only consumption remains in force. After T, all rules in this section are mandatory.

### Rule 14.0: Single Source of Truth

After cutover day T:

- HotThemeRotator's portfolio journal is the sole source of truth for live positions, cash, and realized P&L.
- `Project_optimized/japan_market.db` is frozen as historical archive — no further writes, no consumption by HotThemeRotator runtime code.
- `src/hot_theme_rotator/data/position_adapter.py` is removed from runtime use after migration completes; it survives only as a one-shot reference for the migration script and may be deleted in W3.
- ADR-0005 is superseded by ADR-0008.

Before T: ADR-0005 read-only consumption continues; HotThemeRotator MUST NOT create local positions state that diverges from `japan_market.db`.

### Rule 14.1: Append-Only Event Journal

All portfolio-affecting events MUST be recorded as append-only entries in the portfolio journal (recommended path `reports/portfolio/journal/{trade_date}.jsonl`; implementation defined in P10-21).

- UPDATE, DELETE, and any non-append mutation of journal entries are forbidden.
- `positions` and `cash_balance` are derived views over the journal — never persisted, never patched.
- Re-deriving positions from the same journal MUST be deterministic. Tests MUST guard this.

### Rule 14.2: Manual Entry Validation Gates (fail-closed)

A manual fill entry is accepted only if all of the following pass. Any failure rejects the entry with explicit reason; no partial accept, no silent coercion.

- `symbol` matches the known Japan-equity universe (`.T` suffix, member of accepted ticker list).
- `side` ∈ {BUY, SELL}.
- `qty` is a positive integer.
- `price` is a positive number in JPY (¥).
- `ts` ≤ now (JST). Future-dated entries are forbidden (PIT discipline per Rule 8.6).
- For SELL: derived holdings before this entry ≥ requested `qty`. No short positions.
- For BUY: derived cash − qty × price − fee ≥ 0 produces a warning visible to the user, who must either confirm or enter a corresponding `cash_event` deposit first.

### Rule 14.3: Source Attribution Is Mandatory

Every journal entry MUST declare its `source` ∈ {`manual`, `import`, `paper`, `migration`, `correction`}.

- `manual`: user-typed entry via cockpit UI or CLI.
- `import`: bulk import from broker statement (CSV / XLSX).
- `paper`: simulated paper-trading fill (when P9-05 lands).
- `migration`: one-shot opening-balance entry from Project_optimized cutover (one or more entries on day T).
- `correction`: a reversing or adjusting entry that references a prior `entry_id` via `corrects` field.

Source attribution is consumed by Rule 14.6.

### Rule 14.4: Corrections Via Invalidating Entries

A wrong entry cannot be deleted, edited, or hidden.

To correct an error:

- Append a new entry with `source='correction'` and `corrects=<original_entry_id>`.
- The correction is itself a journal entry with full fields (ts, symbol, side, qty, price, etc.) so it satisfies schema constraints; **its side / qty / price are NOT used for derivation math** — they are documentation that mirrors the original for human readability.
- Append a separate fresh entry with the correct values (e.g., the actual SELL qty). This is the entry that drives the derived view going forward.
- **Derivation semantic**: both the corrected entry and the correction entry are skipped during `derive_positions` / `derive_cash_balance` — they cancel out as if neither had ever happened. The fresh replacement entry is what produces the right state.
- Why not "reversal trade" math: treating the correction as a real BUY-reversal-of-SELL re-prices the un-traded shares (weighted-average cost basis drifts to the wrong-entry's price). Skip-both preserves cost-basis integrity for shares that never actually moved.
- The original wrong entry remains visible in the journal forever (Rule 14.1 append-only); the audit history MUST show both entries side by side.

**Forward-targeting is forbidden (clarified 2026-05-26)**: a correction's `corrects` field MUST reference an `entry_id` that already exists in the journal at the moment the correction is appended. The writer MUST reject corrections whose `corrects` points to a not-yet-appended entry. Without this guard, a malicious or buggy correction could pre-poison a future fill's deterministic `entry_id` (the id is `sha256(ts|symbol|side|qty|price|source|note)` so it is predictable from inputs), causing the legitimate future fill to be silently dropped by skip-both. The manual-entry commit path MUST also reject when the current journal already contains a correction targeting `preview.fill.entry_id` — otherwise the same forward-targeting attack succeeds by ordering correction-before-fill.

### Rule 14.5: Preview Before Commit

Manual entry UI MUST display a preview before the user commits, showing at minimum:

- New holdings (qty + avg cost) after this entry.
- New cash after this entry.
- Realized P&L for SELL entries (mark-to-cost minus fees).
- Any validation warning (Rule 14.2 BUY warning; near-NAV magnitude alarm).

A magnitude sanity check MUST fire when `qty × price > 10% of current NAV`, requiring explicit secondary confirmation — guard against fat-finger errors (e.g., 4000 typed instead of 400).

**Commit-time re-validation (clarified 2026-05-26)**: ``commit_fill`` and ``commit_cash_event`` MUST re-run Rule 14.2 hard gates against the current journal state immediately before appending. A preview is an *advisory* snapshot — between preview and commit the journal may have changed (concurrent append, correction, or clock tick that pushes ``ts`` into the future). If re-validation fails at commit time, the entry MUST NOT be appended; the caller MUST surface a fresh preview to the user. Soft warnings already acknowledged at preview time are NOT re-checked at commit (acknowledgment is sticky).

### Rule 14.6: Calibration Sample Exclusion

Calibration sample purity (Rule 8.6.1 raw-frequency display + Rule 9.4 calibrated win-rate + §10 gate 5) MUST exclude journal entries whose `source` ∈ {`paper`, `migration`, `correction`}.

- `paper`: not a real outcome of a live decision.
- `migration`: synthetic opening-balance, not the outcome of a HotThemeRotator prediction.
- `correction`: tied to a prior entry that itself is the calibration sample; double-counting is forbidden.

Only `source ∈ {'manual', 'import'}` (where `import` represents real broker fills) feeds calibration outcome verification.

### Rule 14.7: Cash Events Are Distinct From Fills

Non-trade cash flow events (deposits, withdrawals, dividends, interest, non-trade broker fees, currency conversion adjustments) MUST be recorded as `cash_event` entries in the journal, not as fills with synthetic symbols.

Each `cash_event` carries: `ts`, `amount` (signed JPY), `reason` ∈ {`deposit`, `withdrawal`, `dividend`, `interest`, `fee_non_trade`, `fx_adjustment`, `other`}, `note`, and an optional `symbol` field when the event is associated with a held instrument (e.g., 1306.T distributions are `cash_event` with `reason='dividend'` and `symbol='1306.T'`).

The seven hard limits Section 14 does NOT relax: Rule 3 advice-only / Rule 4 no silent parameter changes / Rule 8.2 PIT mandatory / Rule 8.3 LLM no probability / Rule 8.6 audit trail / Rule 9.4 calibrated win rate threshold / §10 gate progression.

一个任务只有同时满足以下条件才能标记 done：

- 对应文件已创建或修改。
- 验证命令已运行。
- 输出或失败原因已写入 `PROJECT_STATUS.md`。
- 没有绕过 advice-only 约束。
