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

Manual portfolio recording carve-out (clarified 2026-05-27): HTTP POST endpoints may exist only to record a trade or cash event the user has ALREADY completed in an external broker, per Section 14. These endpoints MUST NOT contain broker route, account, order type, order submission, or live execution fields. They are state-recording endpoints, not execution endpoints, and they do not relax Section 10 gate 8.

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

### Rule 5.1: Execution / Tradability Gate (added 2026-06-17, ADR-0010)

Any candidate or signal presented as potentially *actionable* (not just descriptive) MUST pass a deterministic, fail-closed execution gate that reflects the real constraints of the live account, BEFORE its expected return is taken seriously. The gate (`candidate_engine.tradability`) checks, at minimum:

1. **Lot affordability + diversification.** JPX trades in 100-share lots; one lot's yen cost MUST NOT exceed a position-size cap of the account (default 34%), or the name is flagged non-diversifiable for that account. A ~¥400k account cannot meaningfully hold names much above ~¥1,300.
2. **Round-trip cost from the JPX tick ladder.** Estimated round-trip cost (tick-implied spread × crossings + slippage) MUST clear a configured cap (default 60 bps). The cheapest names carry the worst tick cost — this gate makes that explicit, not silent.
3. **Liquidity floor.** When ADV is available, it MUST clear a minimum (default ¥50M) so a retail order is not a material fraction of volume.
4. **Net-of-cost + 2× stress.** When an expected gross move is supplied, the gate reports the return net of round-trip cost and requires the gross move to survive **2×** the round-trip cost. A signal whose edge does not survive doubled costs is **not actionable**.

The gate is descriptive arithmetic (no prediction). A name failing any sub-gate is surfaced with its `reasons` (never silently dropped), and the Event Desk priced-in read carries the verdict so "looks attractive" is always shown next to "can you actually trade it net of cost." Cross-references: Rule 3 (advice-only), Rule 5 (costs), Rule 11.13 (Event Desk), ADR-0010.

### Rule 5.2: S株 (Fractional / 単元未満株) Execution Mode (added 2026-06-24)

Rule 5.1 is **lot-mode by default (100-share lots)**, which structurally excludes expensive names for a small account (a ¥77,600 name needs ¥7.76M per lot ≫ ¥400k) — so the system was blind to such holdings (e.g. the user's 8035.T). **S株 mode** (SBI 単元未満株) is a first-class execution mode (`tradability.s_kabu_tradability`):

1. **Affordability**: min unit = **1 share**, so any name with `price ≤ max_position_frac × account` is tradable, and share-level sizing improves diversification. (`s_kabu_affordable`.)
2. **Cost**: under SBI ゼロ革命, S株 buy AND sell are **commission-free AND spread-free** (verified 2026-06-24); the only modelled cost is a small **session-reference timing buffer** (default 5 bps), because S株 fills at fixed times (始値 / 後場始値 / 大引け), not on demand. (`s_kabu_round_trip_cost_bps`.)
3. **Caveat (binds horizon)**: S株 executes **~3×/day at session reference prices, no intraday timing, no limit orders** → it suits horizons **≥5D (Rule 16.6)**, not intraday/short signals.
4. **Consequence**: S株 widens the tradable universe to large, liquid, low-spread names (the opposite end of ADR-0010's small-cap drift region) and **collapses the cost hurdle** `IC > τ·c_rt/σ_r` to ≈0. It does **not** create edge — with cost removed, the binding constraint shifts to **signal** (the existing screener score's live Rank-IC is negative, Phase −1 / §16). S株 actionability still requires every Rule 5.1 gate except lot affordability (ADV floor, net-of-cost + 2× stress) plus the Rule 12.5 20%-NAV concentration warning. Cross-refs: Rule 5.1, Rule 12.5, §16, ADR-0010, ADR-0011.

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
- `docs/05_USER_GUIDE.md`：使用说明书（稳定参考文档，非流水进度）。
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

### Rule 8.2.2: Validation-Gate Integrity & Forward-Sample Primacy (added 2026-05-30)

The calibration "ship" gate is the user's trust boundary. Under deployment / "上线" pressure there is a standing temptation to make a failing model pass by moving the goalposts; this rule forecloses it. (Reviewed adversarially via Codex 2026-05-30.)

1. **Locked criteria — no goalpost-moving.** The ship-criteria (the calibration verdict's pass condition — e.g. OOS Brier below the proper baseline) are fixed once set. They MUST NOT be relaxed, nor the validation protocol changed *after seeing results*, to force a failing model to "ship". Any change to the gate requires explicit governance review + a recorded justification in `PROJECT_STATUS.md`, and the gate stays `downgrade` until honestly met. Changing the bar after reading the score is search / p-hacking, not validation.
2. **Forward-sample primacy.** Backdated / replayed / "accelerated-clock" data is R&D diagnostic evidence ONLY — it may guide whether to keep investing effort, but it MUST NOT *alone* satisfy the ship gate. Surfacing a calibrated probability requires confirmation by genuine forward samples, or at minimum a single **locked out-of-time holdout** never touched during audit or tuning. Setting the system clock back and replaying days faster produces *backdated* samples, not forward evidence — forward's entire validity is that the outcome does not yet exist at prediction time, which replay cannot reproduce.
3. **Leakage-audit precondition.** Extending the backdated window (more history) requires a point-in-time **leakage audit FIRST**. Look-ahead — survivorship bias, retroactively-adjusted prices (the confirmed `yfinance auto_adjust` split case affects labels, momentum/MA/vol features, volume, and universe-inclusion filters), and model-selection on future data — *inflates* backdated skill, so a leaky "pass" is a false positive worse than no test. The leakage concern is empirically confirmed, not hypothetical.
4. **Live shadow deployment is allowed and honest.** The system MAY run live daily (automated forward sampling + shadow recalibration) as an *explicitly-uncalibrated research instrument* (Rule 9.4 labeling stays). "Going live" never means surfacing unvalidated signals as validated.
5. **Honest acceptance of the null.** If a leak-audited, methodologically-sound validation (Rule 9.4.1) still fails, the project records "no demonstrated edge" and the score remains an uncalibrated ranking signal — the honest Path-A-passive outcome — rather than searching for a protocol that happens to pass.

### Rule 8.2.3: Locked Forward-Calibration Ship-Criteria (LOCKED 2026-06-11)

Concretizes Rule 8.2.2.1: the forward-calibration ship-criteria are fixed below **as of 2026-06-11, while blind to validation results**. Any later change requires governance review + a recorded justification in `PROJECT_STATUS.md`; until then the gate stays `downgrade`. Owner-set on 2026-06-11 after the ≥100 paired-complete sunset count was reached (Rule 8.2.1) but only **5 independent trading-day clusters** existed — i.e. the raw-count gate was met while the effective-sample gate was not.

1. **Effective-sample floor — ≥ 20 independent trading-day clusters.** A verdict MAY be run only once ≥ 20 distinct live trade-dates have a paired-`complete` outcome. Same-day candidates are correlated (one market move ≈ one observation), so the unit is the **date cluster**, never the raw row count (Rule 9.4.1). 20 is the owner-chosen minimum; it is the floor to *attempt* a verdict, not a guarantee one will pass.
2. **Protocol (locked).** Purged + embargoed walk-forward (`tools/validate_calibration_walk_forward.py`, P12-02): purge training samples whose outcome window overlaps the test window, embargo a gap after each test fold, time-ordered folds, one out-of-time holdout run **once** (8.2.2.1).
3. **Pass condition — ALL must hold:** (a) model out-of-sample proper score (Brier/log-loss) beats **every** baseline — random, climatology/base-rate, **and** stratified; (b) the **date-cluster bootstrap CI on the skill metric excludes zero / no-skill**; (c) the forward leakage audit verdict is `clean` (Rule 9.4.2; `inconclusive` → `contaminated` → no-ship).
4. **Why ≥20 is safe despite being thin.** Pass condition 3(b) does the load-bearing work: at 20 clusters a weak edge will not tighten the cluster-bootstrap CI enough to exclude zero, so a lucky-thin sample stays `inconclusive` rather than passing. The floor lets the verdict be *attempted* earlier; the CI requirement still prevents a false positive.
5. **Outcome.** Pass (all of 3) → score may surface as `calibrated_probability` and activation may be considered (Rule 12.0). Any other result, including `inconclusive` → "no demonstrated edge", score stays `uncalibrated_research_score` (Rule 9.4), Path A passive. A `pass` at 20–29 clusters SHOULD be re-confirmed at the next ≥30-cluster window before it gates real action (advisory, not a moved goalpost).

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
- **Multi-leg synthesis carve-out (added 2026-05-28)**: when a brief is composed of multiple LLM legs (e.g. Bull / Bear / Judge, TradingAgents-style), each leg's output MUST be scanned by `forbidden_pattern` independently. No leg may be marked `speculative_bypass` or otherwise exempted. A multi-leg brief is rejected if any single leg fails the regex twice (per-leg fail-closed). This applies to `llm.per_ticker_brief`, `llm.reflection_brief`, and any future multi-leg variant.
- **User-provided context bidirectional gate (added 2026-05-28)**: when an LLM endpoint accepts user input (e.g. `user_context`, free-text `note` fields submitted from the frontend), that user input MUST also pass `forbidden_pattern` regex before being concatenated into the prompt. Rationale: a user-supplied phrase like "胜率多少？" or "give me a probability" would otherwise launder a probability claim back through the model, defeating post-generation regex. Violation → `422 invalid_input` with explicit reason; never silently strip and proceed.

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

### Rule 9.4.1: Calibration Methodology — Leak-Resistant Validation (added 2026-05-30)

A calibration result that gates the UI MUST be produced by a leak-resistant protocol, not just "more samples". (Surfaced by the Codex 2026-05-30 review of the current block-temporal K-fold.)

1. **Purged + embargoed walk-forward.** Because 3D/5D outcome labels overlap in time, plain random — or even block-temporal — K-fold can leak between train and test label windows. Validation MUST (a) **purge** training samples whose outcome-label window overlaps the test window, and (b) **embargo** a gap after each test fold. Time-ordered walk-forward is preferred over random folds. The existing `kfold_validate_isotonic` is a first cut and is itself suspect until purge/embargo are added.
2. **Proper baselines.** "Beats random (0.5 → Brier 0.25)" is insufficient. The calibrated model MUST also beat the **climatology / event-rate baseline** (constant = historical hit rate, Brier ≈ `p(1-p)`) and any obvious stratified baseline (per sector / ticker). Report all baselines alongside the model.
3. **Effective sample size.** Multi-ticker cross-sections are NOT independent samples — common market factors correlate same-day predictions. Sample counts and confidence intervals MUST use date-block / cluster-robust methods; the raw row count overstates information (a day of 100 tickers is far less than 100 independent observations).
4. **One locked holdout.** Tuning, leakage audit, and threshold selection happen on an earlier window; the most recent locked out-of-time holdout is run **once** and not re-used. A protocol changed after seeing the holdout result voids it (Rule 8.2.2.1).

### Rule 9.4.2: Leakage Audit — Deliverable, Verdict, and Consequence (added 2026-05-31)

Rule 8.2.2.3 + Rule 9.4.1 require a point-in-time leakage audit before any calibration result may gate the UI or the backdated window may be extended. This rule fixes the audit's deliverable, verdict, and what a failing verdict does to existing evidence — so a discovered leak cannot be quietly ignored.

1. **Recorded, auditable deliverable.** The audit MUST emit a verdict artifact (e.g. `reports/calibration/leakage_audit_{date}.json`) enumerating each checked leakage vector with pass/fail + evidence. At minimum it MUST check: (a) corporate-action / `auto_adjust` retroactive price adjustment leaking into labels, momentum/MA/vol features, volume, and universe-inclusion filters (the confirmed `yfinance` case, Rule 11.9.6); (b) survivorship bias in the candidate universe; (c) any `available_ts` later than the decision cutoff feeding an ex-ante feature (Rule 8.2); (d) model / threshold selection performed on future data.
2. **Three-valued verdict, fail-closed.** Verdict ∈ {`clean`, `contaminated`, `inconclusive`}. `inconclusive` is treated as `contaminated` for all gating purposes.
3. **Consequence of a non-clean verdict.** If not `clean`, the affected backdated / bootstrap calibration evidence is QUARANTINED: `evidence_origin="bootstrap"` reports derived from it stop counting toward the Rule 8.2.1 sunset or any validation, the UI calibration display stays downgraded (Rule 9.4), and the bootstrap MUST NOT be re-blessed without a clean re-run from leak-free inputs. The quarantine is recorded in `PROJECT_STATUS.md`.
4. **One-shot lock (anti-p-hacking).** The audit's checklist and pass criteria are fixed BEFORE it runs. Relaxing a check or dropping a vector after seeing a fail, to flip the verdict to `clean`, is search/p-hacking and voids the verdict (parallel to Rule 8.2.2.1). Any change to the audit spec requires a recorded governance justification, and the verdict stays non-clean until honestly met.
5. **Audit is not edge.** A `clean` verdict means "no detected leakage", NOT "demonstrated edge". It only removes a disqualifier; the model must still beat the Rule 9.4.1 baselines on a leak-free protocol before any number is labeled a probability.

Cross-references: Rule 8.2 (PIT), Rule 8.2.2 (validation-gate integrity), Rule 9.4.1 (leak-resistant methodology), Rule 11.9.6 (corporate-action adjustment).

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
- Adjust the time window or session count on any read-only chart, **including mouse-wheel zoom and drag-pan** (added 2026-05-30). Zoom/pan only re-windows EXISTING bars — it MUST NOT fabricate or interpolate bars, nor imply finer precision than the underlying data; the crosshair and any indicator (MA20/MA60, 52w) stay accurate to the real full series (Rule 11.9), and the seven ladder levels (Rule 9.6) are never dropped.
- Hover, focus, or click for tooltips, drill-downs, and on-demand detail.
- Maintain a local watchlist persisted in `localStorage`.
- Add private notes on any candidate or holding (`localStorage` only — never written to `decision_log/`).
- Sort, filter, or re-rank the candidate panel by any visible column.
- Recompute the seven-tier ladder against a user-supplied reference price (the recomputation is a deterministic function of inputs; nothing is persisted).
- Compare two or more candidates side by side.

### Rule 11.2: Forbidden Interactions

The user-facing surface MUST NOT:

- Send any POST / PUT / DELETE / PATCH request, except the explicit Section 14 manual portfolio recording endpoints that record already-executed external broker activity. Rule 3 still forbids broker/order execution endpoints.
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

**Clarification (chart tooltips)**: read-only hover tooltips on price charts displaying realized historical OHLCV and realized change % (e.g., "+1.32%" from prev close) are **structural data display**, not LLM narrative. Rule 8.3.1 / 13.4 forbidden-pattern regex guards **LLM-generated narrative output** (reflection_brief, per_ticker_brief), not structural UI fields. A K-line crosshair tooltip showing a bar's realized percent change is a historical fact, not a probability claim, and is permitted under Rule 11.1. A tooltip MUST NOT display any forward-looking probability or win-rate text unless the recalibrator is active and the value comes from a calibrated_prob structural field — and even then, the label must remain factual ("calibrated 3D probability"), never "win rate".

### Rule 11.5: Frontend Write Path Whitelist (added 2026-05-28)

Frontend-initiated POST / PUT / DELETE / PATCH HTTP requests MUST target only paths listed below. Any new write endpoint requires its own ADR. The whitelist is enforced by contract test (an OpenAPI route audit asserts no non-listed POST routes exist on the FastAPI app).

Allowed write paths:

- `POST /api/portfolio/fill` — record an already-executed trade (Rule 3 carve-out, §14.2 manual entry gates)
- `POST /api/portfolio/cash_event` — record a cash deposit / withdrawal / dividend
- `POST /api/proposals/{id}/accept` — L6 Decision Gate acceptance (Rule 13.18 metadata required)
- `POST /api/proposals/{id}/reject` — L6 Decision Gate rejection (Rule 13.9 ALLOWED_REJECTION_REASONS only)
- `POST /api/watchlist/add` — add a ticker to server-side watchlist (§14.9 user_state)
- `POST /api/watchlist/remove` — remove a ticker from watchlist
- `POST /api/notifier/toggle` — enable / disable a notifier channel (Rule 12.7 double-confirm + audit log)

Forbidden under any name or wrapper (this list is illustrative, not exhaustive):

- Any `/api/broker/*`, `/api/order/*`, `/api/trade/execute*`, `/api/paper/*`, `/api/live_signal/*`
- Any path that submits, modifies, cancels, or routes an order to any execution venue
- Any path that toggles the §10 gate 8 broker hard-block
- Any path that writes directly to `decision_log/predictions/*.jsonl` from the frontend (predictions originate from the scanner / emit CLI, not user clicks)

### Rule 11.6: Strategy / Action Card Output Discipline (added 2026-05-28)

Any frontend component that synthesizes ladder + risk + execution_advice into an actionable strategy card (e.g., `V3StrategyCard`, future V5 deep-dive page) MUST satisfy ALL of the following:

1. **Banner**: a visible `advice-only` banner at the top of the card; non-collapsible; explicit text including `Rule 3` and "由用户在外部券商手动执行".
2. **Forbidden vocabulary**: the card MUST NOT render the words "下单" / "执行" / "自动交易" / "place order" / "submit" / "auto-trade" / "买入" used as imperative verbs against the system. Suggested price levels use the noun form ("建议价位" / "ladder 保守档"), never the imperative.
3. **Manual execution disclaimer**: the card MUST contain the literal string `Rule 3 — manual execution outside HTR` somewhere in its body or footer.
4. **Risk warning section**: the card MUST display a `risk_warnings` block listing applicable warnings from Rule 12 (chase / cooldown / concentration / stale-data). Absence of warnings is itself surfaced as "no active warnings" — never silently omitted.
5. **Uncalibrated language**: when `score_status` is `uncalibrated_research_score`, the card MUST NOT use the phrase "建议买入" or any phrase that implies a probabilistic recommendation. It may use "达 ladder 保守档" / "进入 ladder 激进区间" — factual ladder language, not directional advice.
6. **Price target labels**: any forward price reference is labeled "建议价位" / "目标参考" / "止损参考"; never bare "目标价" (target price) or "目标" alone, which read as recommendations.
7. **Catalyst calendar disclaimer**: if a catalyst block (earnings / ex-dividend) is shown, it MUST carry a "结构化日历数据，非建议持有至该日" disclaimer.
8. **Contract test**: a unit test MUST assert these strings (banner, Rule 3, manual-execution disclaimer, risk_warnings key) appear in the rendered output for at least one sample ticker.
9. **Default-collapse permitted, safety content stays visible (added 2026-06-06, P13-04; clarifies the Codex-flagged gate).** A Strategy Card MAY render its *detail* (the 7-tier ladder, catalyst calendar, `score_status`) inside a default-collapsed body to reduce density on the shipping variant — BUT the **safety triad MUST remain visible in the collapsed state, outside the collapse guard**: (a) the advice-only banner (11.6.1), (b) the Rule 3 manual-execution disclaimer (11.6.3), and (c) the `risk_warnings` block (11.6.4, including the "no active warnings" surfacing when empty). Collapsing the ladder is permitted because the seven tiers (Rule 9.6) are *decision detail*, not a safety red-line, and remain reachable via in-flow expansion (Rule 11.7.2 — expanding pushes siblings, never occludes). The 11.6.8 contract test MUST additionally assert the `risk_warnings` block renders *before* the collapse guard (i.e. is non-collapsible), exactly as the banner+disclaimer already are.

A Strategy Card that fails any of these checks is governance non-compliant and MUST be fixed before merge. The card endpoint (`/api/symbol/{T}/strategy`) is read-only — `risk` and `execution_advice` modules feed it but the endpoint never returns broker-facing fields (`account`, `order_id`, `submit_endpoint`, etc.).

### Rule 11.7: UI Layout & Expansion Invariants (added 2026-05-29)

Content contracts (Rules 11.1-11.6, 9.6) govern *what* a card shows. They do not govern *whether the user can read it*. A variant that satisfies every content contract but crushes cards into illegible slivers, clips expanded content, or overlaps text lines is non-compliant. The following layout invariants apply per the **Scope** clause below:

1. **Document-flow scrolling, no destructive viewport lock.** Total dashboard content MUST NOT be hard-clipped to viewport height such that overflow is hidden or lost. When content exceeds the viewport it scrolls — the document, or a single designated scroll container (e.g. the variant `<main>` pane). A fixed `height: 100vh` / `height: 100%` on a content container is permitted only when paired with `overflow: auto` so nothing is destroyed.
2. **In-flow expansion, never occlusion.** A collapsible card's expanded state MUST reflow siblings (push them down) and grow the scroll container. Expanded content MUST NOT (a) overlap neighboring cards, (b) be clipped by an `overflow: hidden` ancestor, or (c) be squeezed into a height-constrained row that makes text lines overlap. Modals/popovers are the only permitted overlay pattern and use the fixed-overlay convention (`position: fixed; inset: 0`) with a backdrop — they are not "expansion".
3. **No primary-content scroll traps.** Inner fixed-`maxHeight` scroll regions are permitted only for genuinely unbounded feeds (candidate list, news timeline, **macro-news list**, decision log, historical-outcome tables). An expand-on-demand list (show N, ⋯, expand to a capped M with inner scroll) is the canonical pattern here. A card's primary body (factor breakdown, strategy ladder, LLM / debate brief, calibration bars) MUST take its natural height and MUST NOT be locked behind a `maxHeight` so small that its own content overlaps or truncates silently.
4. **Legibility floor.** On the default density, body and label text MUST render at ≥ 11px effective size; numeric / mono blocks ≥ 10.5px. Decorative eyebrows / captions may go smaller. This floor is suspended only inside an explicit user-opted "compact" density toggle, and even then invariants 1-3 still hold.
5. **Per-variant density is opt-in, not a content sacrifice.** A variant MAY pursue a dense single-screen aesthetic, but density MUST come from prioritizing / collapsing / tabbing content — never from clipping, overlapping, or shrinking below the legibility floor. The default variant (currently V3) prioritizes readability over single-screen density.
6. **Regression evidence.** A layout / interaction change to any variant MUST be verified by rendering the running app (headless screenshot or equivalent) in at least the collapsed state and one expanded state, confirming no clipping or overlap. The verification method (or evidence) is recorded in `PROJECT_STATUS.md`.
7. **Mobile priority order (added 2026-07-06, P24, after the design review).** On narrow (single-column) viewports the source/stack order MUST lead with what the operator acts on: market temperature → the Position Exit Board (current holdings, Rule 11.17) → the Daily Action Board (today's plans, Rule 11.16) → then research / feeds / detail. A phone layout MUST NOT bury the two decision surfaces below theme-heat, K-line, factor tables, or the variant switcher. The V1–V4 variant switcher is a comparison affordance, not daily-use chrome; it MUST NOT occupy prime mobile header space ahead of search / positions. Verified by a headless mobile-width screenshot (Rule 11.7.6) plus a source-order contract test.
8. **Vertical balance — no dead void (added 2026-07-06, P24).** A multi-rail layout MUST NOT leave one column empty while a sibling rail runs several times longer, producing a large blank region below the fold. Balance is achieved by *redistributing* content across columns (or reflowing to fewer columns), NOT by capping a rail with an inner scroll region — that would violate 11.7.3 (no primary-content scroll-trap). "Balanced" means the columns end within a bounded delta on the default variant at desktop and tablet widths; verified per 11.7.6.

**Scope (added 2026-05-29 after P10-24 governance audit; invariant list extended 2026-07-06).** Invariants 1-8 are MANDATORY for the **default / shipping variant** (currently V3). Non-default *comparison / experimental* variants (currently V1, V2, V4) exist to trial alternate aesthetics; they are OUT of enforced layout scope and MAY retain alternate layouts — including the pre-P10-24 dense viewport-lock — provided any Rule 11.7 gap is tracked as known-deferred in `PROJECT_STATUS.md`. Such gaps are not merge-blocking. A comparison variant promoted to default MUST first pass invariants 1-8 (verified per 11.7.6). This carve-out covers **layout / legibility only**; the content red-lines (Rule 3 advice-only, Rule 8.3 no-probability, Rule 9.4 uncalibrated, Rule 11.5 POST whitelist, Rule 11.6 strategy-card contracts, **Rule 11.9 display-data honesty, Rule 11.10 interactive write-path wiring**, §14 SSoT) bind **every** variant unconditionally, with no comparison-variant exemption. (Clarified 2026-06-05, P12-04: Rule 11.9 already self-declares "the following bind every variant" — fabricated/hardcoded mock data, faked session state, and unlabeled staleness are non-compliant in comparison variants V1/V2/V4 exactly as in the default V3. The comparison-variant carve-out NEVER licenses dishonest data; it licenses only an alternate *layout*.)

Rule 11.7 complements Rule 9.6: Rule 9.6 preserves the required decision *content*; Rule 11.7 preserves the *presentation* of that content. A change that fixes one MUST NOT break the other.

### Rule 11.8: Position-Sizing & Risk Calculator (added 2026-05-30)

A variant MAY offer a position-sizing / risk calculator (e.g. V1's `仓位与风险测算`, introduced by the 2026-05-30 designer integration) that synthesizes the user's portfolio (NAV / cash) with a candidate's 7-tier ladder into per-tier share counts and risk:reward math. This is a permitted Rule 11.1 exploratory calculation, bounded by ALL of the following:

1. **Deterministic, not directive.** Output MUST be a deterministic function of explicit inputs (a user-chosen risk-budget % of NAV + the ladder entry/stop prices). It may compute "at R% risk to the stop, this tier sizes to N shares"; it MUST NOT rank, recommend, time, or rate which tier or candidate to act on.
2. **Advice-only, never an order.** No broker / order / execution control or imperative verb. The calculator carries advice-only framing; sizes are research references the user acts on manually at an external broker (Rule 3).
3. **Honors the concentration cap.** Any suggested size MUST be capped at the Rule 12.5 concentration limit (default 20% NAV) and visibly flag a tier that would breach it.
4. **No probability / win-rate.** Risk-per-share and risk:reward ratios are arithmetic, not probabilities. The calculator MUST NOT emit any %-probability, win-rate, or expected-profit figure (Rule 8.3 / 9.4).
5. **Read-only.** It reads portfolio + ladder and writes nothing (no POST), consistent with Rule 11.5.

A calculator that fails any of these is governance non-compliant. The contract is asserted by `tests/unit/test_frontend_ui_contracts.py::test_v1_risk_sizing_is_advice_only_research_calc`.

### Rule 11.9: Display Data Honesty — No Simulated Data, Mandatory Freshness Labeling (added 2026-05-30)

The pull-mode dashboard is a research surface; it MUST present what the backend actually observed, never a fabricated or animated approximation. Discovered when a user noticed prices "jumping" on a Saturday (closed market). The following bind **every** variant:

1. **No simulated data or motion.** The UI MUST NOT generate price / quote / metric values or movement from client-side timers, random walks, or animation. A displayed price is the backend's last observed value — **static** — until the backend delivers a new one. (The 2026-05-30 audit found `useTickingPrice` animating a ±0.07%/3.4s random walk on a closed-market Saturday, with red/green "live" pulse — removed.) Color/pulse liveness cues are permitted ONLY when driven by a real backend update gated on a true market-open signal.
2. **Freshness / session labeling is mandatory.** Every price / quote / temperature surface MUST carry (or sit under) an as-of indicator derived from real signals. When the data's `tradeDate` is not the current trading session (weekend / holiday / pre-open / stale), the UI MUST say so (e.g. `休市 · 数据为 {tradeDate} 收盘`), computed from `meta.asof` vs `meta.tradeDate`. It MUST NOT imply liveness it does not have.
3. **Market-session state MUST be derived, not faked.** Any OPEN / CLOSED / PRE indicator MUST come from a real clock + exchange trading-calendar, or a backend `market[].state` that is itself calendar-correct. Hardcoded session states are forbidden. (Backend `market_temp_adapter` returning `OPEN` on a Saturday is a tracked defect.)
4. **Mock / degraded data MUST be visibly flagged.** When a section falls back to offline-mock, or the backend reports it unavailable (`meta.dataAvailability.{section} === false`), the UI MUST render a visible `示例数据 / 数据未就绪` marker for that section — never present mock as real. Governance-sensitive surfaces (calibration K-fold / Brier figures, L6 proposals) MUST read the real backend (`/api/calibration/reliability`, `/api/proposals`) and never ship hardcoded numeric literals.
5. **Per-symbol detail MUST come from the real endpoints in the live build.** Variants showing per-symbol strategy / factors / outcomes / K-line / LLM brief MUST source them from the real `/api/symbol/{T}/*` (via the enrichment layer), not from synthesized offline defaults.
6. **Price series MUST be corporate-action adjusted for continuity.** A price/OHLC series used for display OR factor computation MUST be split- and (where applicable) dividend-adjusted so it is continuous across corporate actions. An unadjusted split cliff is misleading data — discovered when 1306.T showed ¥3817 (pre-split) crashing to ¥382 (post-split) on its 2026-03-30 10:1 split; the K-line adapter now back-adjusts (heuristic, logged). Where no split calendar exists, auto-detection MUST be conservative (clean N:1 ratios only) and **logged for audit**; the robust long-term fix is an explicit splits table / source-level adjusted prices. The same un-adjusted prices feeding factor calcs (momentum / sharpe) is a tracked risk (see PROJECT_STATUS). Cross-reference: Rule 8.2 (PIT).

A surface that fabricates data, hides staleness, ships hardcoded governance figures, or presents an unadjusted corporate-action discontinuity is non-compliant. Cross-references: Rule 8.2 (PIT), Rule 9.4 (uncalibrated honesty), §12 (anti-FOMO). Remediation is staged — see the Rule 11.9 backlog in `PROJECT_STATUS.md`; item 1 (no simulated price) + item 2 (freshness label on V3) landed 2026-05-30.

### Rule 11.10: Interactive Write-Path Wiring Integrity (added 2026-05-31)

A frontend control that presents itself as performing an action — recording a fill / cash event, toggling a notifier channel, adding/removing a server-side watchlist entry, accepting/rejecting a proposal — MUST actually invoke its Rule 11.5-whitelisted backend endpoint. A control that only mutates local component state while looking live is a non-functional stub and is non-compliant. Discovered 2026-05-31 (frontend↔backend correspondence audit): the designer-redesign `ManualEntryModal` submit button only toggled a preview flag and never POSTed `/api/portfolio/fill`; `NotifierChip` flipped local state without calling `/api/notifier/toggle` or writing the Rule 12.7 audit log — both backends were complete, tested, and whitelisted.

1. **Wired or labeled — never silently dead.** Any actionable control is in exactly one of two states: (a) WIRED — it calls its real endpoint and surfaces the real result/error; or (b) explicitly LABELED a demo / placeholder per Rule 11.9.4 (a visible "演示 / 未接线" marker) AND tracked as deferred in `PROJECT_STATUS.md`. A control that looks functional but writes nothing is forbidden.
2. **No orphan write endpoints.** A Rule 11.5-whitelisted POST endpoint intended for UI use MUST have a frontend consumer, OR be explicitly recorded in `PROJECT_STATUS.md` as a known-deferred / unused endpoint with the reason. Silent orphan write endpoints (backend done, no caller, untracked) are non-compliant — they read as "feature shipped" when the user cannot reach it.
3. **Preview/commit honored.** Where the backend exposes a preview-vs-commit contract (e.g. fill/cash `commit: false|true`), the UI's two-step (preview → confirm) MUST map to it: the preview click sends `commit:false` and shows the returned before/after + warnings; the confirm click sends `commit:true` and surfaces the committed result (or the 409 re-validation error). The UI MUST NOT fabricate a "preview OK" without the real preview round-trip.
4. **Contract test asserts the call, not just the render.** For each whitelisted POST with a UI control, a contract test MUST assert the control's handler issues a POST (method + path) — a test that only checks the button renders is insufficient and is exactly what let the 2026-05-31 stubs pass.
5. **Interactive vs data/observability.** This rule binds INTERACTIVE controls (the user acts and expects a system effect). Pure data / observability / inspection endpoints (e.g. `reflection/snapshots|traces|funnels`) MAY exist without a UI consumer, provided clause 2's deferred-tracking is satisfied — a read-only diagnostic surface is frontend-optional; an action the user can click is not.

Rule 11.10 complements Rule 11.5 (which paths MAY be written) and Rule 11.9 (display honesty): 11.5 bounds the write surface, 11.9 forbids fake data, 11.10 forbids fake interactions.

### Rule 11.11: Historical Candidate Review (read-only, PIT-faithful) (added 2026-06-07)

The historical candidate review surface ("历史候选复盘卡") renders past daily candidate cohorts from the append-only decision log (`reports/predictions/*.jsonl` + `reports/outcomes/*.jsonl`). It is read-only and additionally bound by Rules 8.2 (PIT), 8.3 / 9.4 (no-probability / uncalibrated honesty), 12 (anti-FOMO), and 14.6 (backdated/live separation). Because it juxtaposes past picks with realized outcomes — the exact surface where survivorship, hindsight, and cherry-picking creep in — the following are fail-closed:

1. **PIT fidelity.** Every candidate attribute (research score, reason codes, reference price, decision cutoff) MUST be rendered from the stored `PredictionRecord` as of its `decision_cutoff`. The surface MUST NOT re-fetch or back-fill any roster field with data dated after the cutoff. The ONLY post-cutoff data shown is the joined `OutcomeRecord` realized return.
2. **Cohort-first, no cherry-pick.** Realized performance MUST be presented as a whole-cohort equal-weight aggregate (mean return + positive-share) for the selected trading day(s) BEFORE any per-candidate figure. The UI MUST NOT offer a control that selects a favorable subset of candidates or an arbitrary favorable date window; the only selectors are a single trade date or the full live series.
3. **Maturity honesty (survivorship guard).** Only `OutcomeRecord.status == "complete"` samples enter an aggregate. Candidates whose outcome window has not closed, or whose outcome is `malformed_data` / `symbol_not_found` / `insufficient_data` / `no_outcome`, MUST be shown explicitly and counted as excluded — never silently dropped.
4. **Benchmark-relative framing.** Every cohort return MUST be displayed alongside the same-window benchmark (1306.T TOPIX ETF, the passive baseline) return, indexed on the same horizon convention as `decision_log.outcome_join`. An absolute cohort gain in a rising market MUST NOT be presented as evidence of skill.
5. **Backdated separation.** Backdated / bootstrap predictions (`model_version` ending `-backdated`, or `extra.backdated == true`) MUST be excluded from the default view and from every live-track aggregate. They may appear only behind an explicit, separately labeled "合成样本 · 非真实战绩" control (Rule 14.6).
6. **No win-rate labeling.** Positive-share, mean return, excess return, and any cohort statistic are descriptive observations of an `uncalibrated_research_score` population. They MUST NOT be labeled win rate, profit probability, expected value, or any predictive claim, and a standing disclosure (e.g. "未校准研究分数 · 样本不足以得出胜率结论 · 仅供复盘观察") MUST be visible whenever paired-complete samples are below the Rule 8.2.1 sunset threshold. (The disclaimer may use the word 胜率 in the negative, exactly as the dashboard's "不是真实胜率"; that is a disclaimer, not a claim.)
7. **Read-only.** The surface and its API (`GET /api/candidates/history`, `GET /api/candidates/history/dates`) are read-only; no control mutates predictions, outcomes, journal, or watchlist. No POST route is added (it stays outside the Rule 11.5 write whitelist by construction). A contract test asserts the endpoints reject POST and carry the standing disclosure.

A review surface that reconstructs the roster with post-cutoff data, mixes backdated samples into a live aggregate, silently drops unmatured candidates, omits the benchmark, or labels a cohort statistic as a win rate is non-compliant. Cross-references: Rule 8.2 (PIT), Rule 9.4 (uncalibrated honesty), §12 (anti-FOMO), Rule 14.6 (sample separation), Rule 11.9.6 (corporate-action continuity — benchmark windows must not span an unadjusted split).

### Rule 11.12: News-Catalyst / Theme-Leader Badge Honesty (added 2026-06-14)

ADR-0009's hybrid engine reorders the dashboard candidates by a news theme-heat catalyst and marks the top blended-rank candidate in each hot theme as that theme's leader. The badges that surface this (`🔥 新闻催化`, `👑 龙头`) sit on the most persuasive part of the UI — the leader/candidate hero — and so are fail-closed bound by Rules 9.4 (uncalibrated honesty), 8.3 (no-probability), 4 (explicit weights), and 11.7 (all-variant parity):

1. **Ordering signal, not a win-rate.** The badge communicates that today's news theme-heat moved this candidate's *position in the list*. It MUST NOT be labeled, colored, or tooltip'd as a probability, win rate, expected value, or any predictive claim. A standing disclosure stating it is an ordering signal (not 胜率/概率) and that the displayed score remains the uncalibrated research score (排序≠分数) MUST accompany the badge.
2. **Score stays raw (order ≠ score).** The numeric score shown for a candidate remains the raw `uncalibrated_research_score` (Rule 9.4). The catalyst/blended score MAY explain the order but MUST NOT replace or inflate the displayed score, and the raw `catalyst_score` / `blended_score` MUST NOT be rendered as a standalone 0–1 figure that could be misread as a probability.
3. **PIT-derived, no fabrication.** The badge is rendered ONLY from served candidate fields (`newsCatalyzed`, `topTheme`, `isThemeLeader`, `catalystScore`). An uncatalyzed candidate (`newsCatalyzed == false`) renders no badge and is never designated a theme leader. The frontend MUST NOT synthesize catalyst/leader status the backend did not assert.
4. **Fail-open degradation.** When news/metadata are absent the serializer's catalyst is 0 and the order degrades to the pure price screener (Rule 11.9); the UI then simply shows no badge. A missing badge is the honest state, never an error to paper over.
5. **All-variant parity.** The badge is a single shared component (`CatalystBadges`) bound into every variant (Rule 11.7); content red-lines above bind all variants regardless of layout.
6. **Company evidence required for the persuasive badge (added 2026-06-18).** The backend MUST classify catalyst evidence as `company`, `sector`, or `none`. `company` requires a served news item to explicitly link the symbol (`linked_symbols`) or match the metadata company name in the title. A ticker that only maps to a hot theme through sector/industry metadata is `sector` evidence: it may receive only a weak ordering nudge, MUST expose `sectorCatalyzed=true`, MUST keep `newsCatalyzed=false`, and MUST NOT be designated a theme leader. This prevents broad theme heat from being presented as a company-specific catalyst.

A badge that presents the catalyst or blended score as a probability/win-rate, inflates the displayed score, designates an uncatalyzed candidate as a leader, or omits the ordering-not-win-rate disclosure is non-compliant. Cross-references: Rule 9.4 (uncalibrated honesty), Rule 8.3 (no-probability), Rule 4 (explicit weights), Rule 11.7 (all-variant parity), Rule 11.9 (graceful degradation), ADR-0009 (hybrid engine).

### Rule 11.13: Event Desk — Analysis Support, Not Event Prediction (added 2026-06-15)

The Event Desk (P16, 事件作战台) helps the owner trade discretionarily on events (ceasefire / AI-semi / storage / yen moves) by surfacing, for an event or theme: which liquid JP names are **exposed**, and a **priced-in read** of how far each has *already* moved (1d/5d/20d returns, excess vs the 1306.T passive benchmark, distance from the 20-session high, a descriptive freshness label). It is the most seductive surface in the product — an event narrative next to a stock list — so it is fail-closed bound by Rules 3 (advice-only), 8.3 (no-probability), 9.4 (uncalibrated honesty), and 12 (anti-FOMO):

1. **No event-outcome prediction.** The desk MUST NOT emit, label, or imply any probability that an event resolves a given way, nor any probability/win-rate/expected-value that an exposed name rises or falls. It surfaces what HAS happened to prices, not what WILL happen. The directional call is the owner's; the desk only shows exposure and already-realized price action.
2. **Every figure is verifiable price data.** Returns, excess, distance-from-high, and the freshness label are deterministic functions of observable closes. The freshness label (`fresh` / `extended` / `rolling_over` / `falling`) is an explicitly descriptive bucket of *how far it has already moved* — NOT a buy/sell signal, and it MUST be presented as such with a standing disclosure.
3. **Exposure is a data join, not a recommendation.** The event→theme→names mapping answers "who is exposed", surfaced equally; the desk MUST NOT rank names as "best to buy", and presence on the list is not advice to trade. A ceasefire's exposed names include the war-premium names a de-escalation would *hurt* — the desk shows them without implying a direction.
4. **Benchmark-relative + fail-open.** Price action is shown alongside the same-window 1306.T return (the passive alternative the owner already holds). Missing/short price history yields nulls + `unknown`, never a fabricated value or an exception (Rule 11.9). The price source is injected with recorded provenance (Rule 2.1).
5. **No write path, all-variant parity.** The desk and its API are read-only (outside the Rule 11.5 write whitelist); any UI surfacing it binds every variant (Rule 11.7) and carries the standing disclosure.

A desk that ranks exposed names as buys, attaches an up/down probability, presents the freshness label as a signal, or omits the not-a-prediction disclosure is non-compliant. Cross-references: Rule 3 (advice-only), Rule 8.3 (no-probability), Rule 9.4 (uncalibrated honesty), Rule 11.7/11.9/11.11 (parity / degradation / PIT review), §12 (anti-FOMO).

### Rule 11.14: Theme Rotation Overlay Honesty (added 2026-06-20)

The memory / semiconductor rotation overlay exists to keep the system aligned with the current hot-money battlefield without pretending that a theme narrative is a calibrated forecast. It may annotate and lightly reorder candidates, but it is still an uncalibrated research layer bound by Rules 3, 4, 8.2, 8.3, 9.4, 11.9, 11.12, 11.13, and 12.

1. **Overlay, not a replacement score.** `themeRegime`, `leaderExtended`, `chaseRisk`, `secondLineCandidate`, and `rotationScore` MAY explain candidate ordering. They MUST NOT replace the displayed raw research score, and MUST NOT be displayed as a probability, win-rate, or expected return.
2. **Core-theme data must be fresh.** When the active hot theme is memory or semiconductors, the core coverage basket MUST include Kioxia (`285A.T`) plus the configured major JP semiconductor names. If any core symbol has stale or missing price data for the latest trading date, the system MUST surface `coreThemeDataFresh=false` and MUST downgrade theme-confidence language. It MUST NOT present a clean memory/semi recommendation from an incomplete basket.
3. **Extended leaders are study-only unless reconfirmed.** A theme leader with extreme recent movement (for example high 20d/60d momentum, close near the 20d/52w high, and volume expansion) is `leaderExtended`. Such a name may remain visible as the reference leader, but BUY-style language and proactive alerts MUST be downgraded to `study_only` unless a later implementation records a fresh company catalyst or a consolidation/breakout reconfirmation.
4. **Second-line expansion requires multiple facts.** A `secondLineCandidate` label requires at least two independent facts among relative strength versus the leader, volume expansion, non-stale price data, company-level catalyst, better valuation/fundamental read, or improving 5d/20d trend. Sector membership alone is insufficient.
5. **Weights are explicit and small.** Rotation overlay boosts and penalties are explicit config under Rule 4. Until walk-forward evidence proves otherwise, the overlay may only lightly reorder candidates after the existing screener and quality-gated catalyst rerank; it must not hard-filter the universe except for stale data or alert downgrades.

A surface that promotes an extended leader as a clean buy, hides stale core-theme coverage, treats sector membership as company evidence, or presents the rotation overlay as a calibrated forecast is non-compliant.

### Rule 11.15: External ADR Catalyst Lane Honesty (SKHY) (added 2026-06-25)

The SK hynix ADR lane exists to observe an external semiconductor catalyst that may affect Japanese semiconductor candidates. It does not change HTR's primary market, which remains Japanese equities, and it does not create an order-routing, broker, or calibrated ADR-trading system.

1. **External input, not enabled trade market.** `SKHY`, `000660.KS`, `MU`, `NVDA`, SOX/semiconductor ETFs, and USDJPY may be used as external temperature/catalyst inputs. They MUST NOT become HTR candidate symbols, JP portfolio calibration samples, or enabled trade markets without a separate governance update.
2. **Listing status must be explicit.** If `SKHY` is not yet actively tradable, the ADR watch lane MUST show `pending_listing`, `unavailable`, or `stale` rather than manufacturing a price or substituting another symbol. A stale or missing SKHY feed MUST leave Japan candidate ranking unchanged.
3. **No probability or edge language.** SKHY overlay fields may describe event state, freshness, relative strength, volume, and sympathy reasons. They MUST NOT display win rate, probability, expected return, guaranteed chance, or calibrated edge unless future Rule 8.2/9.4/16 gates explicitly allow it.
4. **Small post-rerank annotation only.** Any SKHY-derived Japan semi overlay must run after the existing catalyst and Rule 11.14 rotation overlays, use explicit small weights under Rule 4, and require at least two independent facts before a positive `semiSympathy` label. Sector membership alone is insufficient.
5. **Manual ADR journal is record-only.** If the operator records an external SKHY ADR fill, the record MUST be manual, already completed in an outside broker, stored separately from JP portfolio/calibration data, and excluded from candidate outcomes, Rank-IC, cohort review, and model validation.
6. **Evidence before promotion.** Any claim that the SKHY overlay improves Japan semi selection requires live-only forward review under ADR-0010 and Rule 16, including cost hurdle context, event-cluster count, multiple-testing discipline, and an explicit `insufficient_data` verdict until the gate is actually cleared.

A surface that implies "SKHY listing means buy Japan semis now", hides stale ADR/listing status, mixes ADR PnL into JP validation, or presents the overlay as a calibrated money-making signal is non-compliant.

### Rule 11.16: Daily Action Board — Trade Plans, Not Predictions (added 2026-07-02, P21)

The Daily Action Board (交易计划板) answers the owner's operating question — "which candidates, at what price, under what conditions" — by ASSEMBLING already-gated components into one read-only surface: the existing blended candidate ordering (Rule 11.12), the 7-tier price ladder (Rule 9.6/11.6), chase-risk / rotation annotations (Rule 11.14), tradability arithmetic (Rule 5.1), and deterministic position-sizing arithmetic (Rule 11.8). Because it is the single most action-shaped surface in the product, and because the 2026-07-02 edge-search close-out established the system has NO demonstrated predictive edge, the board is fail-closed bound by ALL of the following:

1. **Plan, not prediction.** A board row is a STRUCTURE reference (entry zone / stop reference / take-profit references / size arithmetic / factual conditions). It MUST NOT emit or imply a probability, win rate, expected return, edge, or forecast that the candidate will rise. The standing disclosure ("交易计划 = 结构参考，不是预测；系统无 demonstrated edge；由你在外部券商手动决定与执行 — Rule 3") MUST be visible and non-collapsible.
2. **No new predictive score, existing order only.** The board inherits the served candidate ordering (blended rerank, Rule 11.12) and MUST NOT introduce a new ranking/score that could read as a fresh prediction layer. Rows may be *downgraded* by factual gates (chase risk, tradability, data sufficiency) but never re-ranked upward.
3. **planStatus is factual and fail-closed.** Every row carries `planStatus` ∈ {`plan_ready`, `watch_only`, `s_kabu_only`, `not_tradable`, `insufficient_data`}. `chase_risk == "study_only"` or `leader_extended == true` FORCES `watch_only` (Rule 11.14.3). Missing price/ladder/stop-geometry FORCES `insufficient_data`. Whole-lot infeasibility with S株 feasibility yields `s_kabu_only`; neither feasible yields `not_tradable`. A downgrade is never hidden.
4. **Price vocabulary binds.** All price levels use the Rule 11.6.5/11.6.6 noun forms (建议价位 / 止损参考 / 目标参考 / ladder 档位). "建议买入" and imperative buy/sell verbs against the system are forbidden (uncalibrated state, Rule 11.6.5).
5. **Sizing is Rule 11.8 arithmetic, visibly parameterized.** Share counts derive ONLY from an explicit risk-budget fraction of NAV (default 1%) against the entry→stop distance, capped by the Rule 12.5 concentration limit (20% NAV) and available cash, with the whole-lot (100株) and S株 (1株) variants both shown where relevant. The parameters (risk %, caps) are displayed. No expected-profit figure. When portfolio NAV is unavailable, sizing is absent (`sizing: null`) — never estimated from a fabricated NAV.
6. **Timing is a conditions checklist, not a clock call.** "When" is expressed ONLY as verifiable current facts: market session state (derived, Rule 11.9.3), chase-risk level, catalyst evidence level, data freshness, and distance-to-tier percentages. The board MUST NOT emit a time-of-day/date prediction or "buy now" urgency cue.
7. **PIT + read-only + parity.** Rows derive exclusively from already-served candidate/portfolio fields (no post-cutoff enrichment); the surface adds no POST route (outside the Rule 11.5 whitelist by construction); the shared card binds every variant (Rule 11.7 content red-lines) and fail-opens to nothing when the backend omits the board.
8. **Contract tests.** Unit tests MUST assert: the standing disclosure and Rule 3 framing render; forbidden vocabulary (建议买入/胜率/概率/期望收益/win rate/probability) is absent from board output **outside negated disclosures** (the Rule 11.11.6 reading — the mandated disclosure itself uses these words in the negative, e.g. "不含概率/胜率/期望收益"; clarified 2026-07-06); `watch_only` forcing on study_only/extended rows; sizing caps enforced; POST rejected on any board endpoint.

A board that ranks by a new predictive score, hides a chase/tradability downgrade, emits sizing from an invented NAV, uses imperative buy language, or drops the no-edge disclosure is non-compliant. Cross-references: Rule 3, 5.1, 8.3, 9.4, 9.6, 11.6, 11.8, 11.12, 11.14, 12.5.

### Rule 11.17: Position Exit Discipline Board — The Operator's Own Rules, Applied Arithmetically (added 2026-07-02, P22)

The Position Exit Discipline Board (持仓纪律板) answers 00_DESIGN §2 Q4 — "当前持仓应该止盈、止损、继续持有还是换仓" — for the CURRENT holdings. Its entire content is arithmetic between observed prices and the OPERATOR'S OWN declared discipline parameters (the "2-5% take-profit / rotate" strategy stated in 00_DESIGN §1). It predicts nothing. Fail-closed constraints:

1. **Parameters are the operator's declaration, not model output.** Take-profit references (default avg_cost +2% / +3% / +5%) and the stop reference (default avg_cost −4%) are explicit config under Rule 4, displayed on the surface, and labeled as the operator's discipline (纪律参数). They MUST NOT be presented as model-derived targets, forecasts, or optimal levels.
2. **Status is price-vs-reference arithmetic, fail-closed.** `exitStatus` ∈ {`within_plan`, `past_first_take_profit`, `stop_reference_breached`, `insufficient_data`}. Missing/invalid price or cost basis FORCES `insufficient_data` — never a fabricated status. A breached stop reference is surfaced prominently; hiding it is non-compliant.
3. **Vocabulary.** All levels use 止盈参考 / 止损参考 / 纪律参考 noun forms. The board MUST NOT render imperative sell/hold verbs against the system ("卖出"/"清仓" as commands), any probability/win-rate/expected-return, or a rotate directive. A rotate cross-reference may state only the FACT of how many actionable rows (`plan_ready` + `s_kabu_only`) today's Action Board has, and its rendered label MUST name exactly what is counted (clarified 2026-07-06 after governance review: the count includes S株-feasible rows because for this account they ARE the common actionable state; a label that says one thing while counting another is the violation).
4. **Advice-only, record-only.** The board never acts: exits happen at the external broker and are recorded through the existing manual journal path (Rule 3 / §14). The board adds no POST route.
5. **PIT + read-only + parity + degradation.** Rows derive exclusively from the served portfolio state (journal SSoT) and its recorded prices; the shared card binds every variant (Rule 11.7 content red-lines); when portfolio data is unavailable the board fail-opens to nothing (Rule 11.9.4 honest absence, never fabricated holdings).
6. **Contract tests** assert: disclosure renders; parameters visible; breached-stop surfacing; insufficient_data forcing; no forbidden vocabulary; no new POST.
7. **Mandate precedence (added 2026-07-20, P27).** The generic parameters in 11.17.1 encode the 00_DESIGN §1 swing strategy. They govern ONLY holdings the Section 17 owner mandate does not cover. For any symbol present in `sleeve_map`, the mandate supersedes them and the cost-anchored references MUST be suppressed for that row, replaced by the reference that actually binds:
   - **Sleeve A** → no per-symbol level. Its expected return IS the equity risk premium (Rule 17.1); a cost-anchored stop realizes the drawdown the sleeve is compensated for holding, and on a 2× instrument a −4% band sits inside two daily sigma of the mandate's own σ ≈ 18% assumption. The binding discipline is the portfolio-level β-adjusted exposure band (Rule 17.2).
   - **Sleeve B** → no per-symbol level; the position is pre-committed to its verdict date (Rule 17.5), and stopping out a measurement basket destroys the measurement it was bought to produce.
   - **Sleeve C** → the declared bilateral close bracket (Rule 17.4.6), else the review-drawdown trigger off the re-underwrite price (Rule 17.4.4). **Never the entry cost** — displaying a cost-anchored stop on a C position re-displays the exact anchor 17.4.6 exists to abolish, which is the disposition effect wearing a discipline badge.

   `exitStatus` gains `mandate_governed` / `mandate_exit_triggered` / `mandate_review_required`. Every row MUST carry `sleeve` and `disciplineSource` so a suppressed generic stop can never be misread as "no discipline", and the surface MUST state the generic parameters' scope rather than implying they are global. Absent or invalid mandate config → fail-open to the pre-P27 behaviour (generic parameters for all rows), never a fabricated sleeve. Rationale: two discipline layers that disagree on the same holding produce a false alarm that trains the owner to ignore the board — the layer that governs must be the layer that renders.

Cross-references: Rule 3, 4, 8.3, 9.4, 11.6, 11.7, 11.9, 11.16, §14, §17 (17.1, 17.2, 17.4.4, 17.4.6, 17.5).

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

### Rule 12.7: Notifier UI Toggle Discipline (added 2026-05-28)

When the frontend exposes a UI control to enable / disable a notifier channel (desktop / email / telegram), the toggle MUST satisfy:

1. **Default disabled**: every channel ships disabled. The UI on first load reflects this — never opt-out, always opt-in.
2. **Double-confirm on enable**: clicking an "enable" toggle MUST trigger a modal confirmation listing: (a) the channel name, (b) the literal text "我理解这会触发 {desktop|email|telegram} 推送", (c) Rule 12.0 stage 2 reminder, (d) cancel and confirm buttons. The toggle does NOT flip on without the confirm.
3. **Audit log append-only**: every toggle action (enable or disable) writes one row to `reports/observability/notifications/toggle_log.jsonl` containing `{ts, channel, action ∈ {enable, disable}, user_confirm_text}`. Schema is append-only — no overwrite, no truncation.
4. **Stage 2 gate**: enabling is rejected (with reason) if Rule 12.0 stage 2 prerequisites (P10-18 discipline filter passing) are not satisfied. The UI surfaces the rejection with a "blocked by stage 2" pill.
5. **Dry-run capability**: the UI MUST offer a separate "dry-run" button that synthesizes a sample alert and routes it through the discipline filter without invoking the real channel. Dry-run never appends to `toggle_log.jsonl`.
6. **No silent re-enable**: a disabled channel cannot be re-enabled by editing a JSON file or restarting the app — the toggle action is the only ingress, and re-enable requires another double-confirm.

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

### Rule 13.11: Validity Class Controls Allowed Action

`counterfactual_validity` is not only wording. It controls what P11-06 may intake:

- `exact_replay` / `partial_replay`: investigation, RCA, and parameter-change proposals are allowed if all Rule 13.3 / 13.7 / 13.17 gates pass.
- `universe_reconstructed`: investigation-only. No parameter / threshold / config change proposal.
- `price_only_replay`: price-path discussion only. No alert-behavior claim and no parameter / threshold / config change proposal.
- `data_too_stale`: freshness/data-source proposal only (`evidence_class=freshness_attribution`). No strategy or parameter proposal.
- `invalid`: no proposal. The system may log diagnostics but MUST NOT ask the user to accept an action.

### Rule 13.12: LLM Cannot Originate Proposals

P11-05 LLM output is narrative synthesis only. Any `proposed_actions` visible in a reflection brief MUST be structured actions derived by deterministic L3/L4 code, stamped with `source_layer` and `generator`, and passed through unchanged. The LLM may rephrase caveats, but it MUST NOT invent action ids, parameter values, evidence classes, sample sizes, confidence intervals, validity classes, or rationale pointers.

### Rule 13.13: Proposal Tiers and Blast Radius

Every proposal MUST be treated as one of:

- `diagnostic_only`: explains an observed issue; no config change.
- `data_quality`: refreshes, replaces, or repairs data inputs.
- `threshold_tuning`: changes numeric thresholds or budgets.
- `policy_change`: changes decision logic, filters, alert semantics, or user workflow.

`threshold_tuning` and `policy_change` are high-blast-radius. They require Rule 13.7 backtest evidence, Rule 13.17 reproducibility metadata, and Rule 13.14 shadow/canary before active use. Data-quality proposals may bypass market-performance backtest only when the proposal fixes stale/missing data and does not alter trading policy.

### Rule 13.14: Accepted Parameter Changes Start in Shadow

Accepting a reflection proposal does NOT immediately make a parameter / threshold / config change active. Accepted parameter changes default to `lifecycle_stage=shadow` with rollback required. Promotion path is:

1. `shadow`: compute candidate output side-by-side, no user-facing behavioral change.
2. `canary`: limited surface or limited time window, with explicit rollback criteria.
3. `active`: only after post-acceptance evidence confirms the expected improvement without violating Rule 12 or Rule 3.

### Rule 13.15: Anti-Oscillation Cooldown

The same `intervention_target` MUST NOT receive another parameter-change proposal inside a 14-day dwell window after a pending or accepted parameter-change proposal for that target, unless the new proposal is explicitly marked as emergency rollback. This prevents repeated threshold nudging caused by short noisy windows.

### Rule 13.16: Expiry and Rejection Semantics

Expired proposals MUST carry an `expiration_reason`. `unreviewed_timeout` may feed Rule 13.10 expiry-pattern meta-reflection. Operator-context reasons such as `operator_unavailable` MUST NOT be treated as generator failure. Rejections MUST keep a machine-readable reason plus optional free text; meta-reflection should distinguish insufficient evidence from out-of-scope or operator-context rejection.

### Rule 13.17: Reproducibility Metadata Required

Every proposal accepted by P11-06 intake MUST carry reproducibility metadata:

- `source_trace_ids`
- `config_before_hash`
- `candidate_config_hash`
- `outcome_window`
- `denominator_counts` (eligible universe / scored / suppressed / alerted / acted when available)

Without these fields, the proposal cannot be reproduced later and MUST be rejected before human review.

The five hard limits Section 13 does NOT relax: Rule 3 advice-only / Rule 8.2 PIT mandatory / Rule 8.3 LLM no probability / Rule 9.4 calibrated win rate threshold / §10 gate progression.

### Rule 13.18: L6 UI Accept / Reject Surface Discipline (added 2026-05-28)

When the frontend exposes UI controls to accept or reject a Proposal (P2.1 Decision Proposal Inbox), every accept / reject action MUST satisfy:

1. **Full metadata display**: the UI MUST render every Rule 13.6 field on the same screen as the action buttons — `proposal_id`, `evidence_class`, `sample_size`, `confidence_interval`, `counterfactual_validity`, `rationale_pointer`, `generator`, `created_ts`, `tier` (Rule 13.13), and any `source_trace_ids` (Rule 13.17). Hidden or collapsed fields are forbidden; a user accepting a Proposal must have seen its full provenance.
2. **Shadow disclosure on accept**: when accepting a `parameter_change` Proposal, the UI MUST display "this enters `lifecycle_stage=shadow`, not active production" (Rule 13.14) before the confirm button is enabled. The disclosure is dismissible only by a separate "I understand" checkbox.
3. **Bounded reject reasons**: rejecting MUST present a dropdown of `ALLOWED_REJECTION_REASONS` (per Rule 13.9, enforced by `reflection.decision_gate`). Free-text-only rejection is forbidden; an optional free-text note may accompany the structured reason.
4. **Expiry banner**: any Proposal whose age >= 7 days (Rule 13.5) MUST render a banner "⚠ expired — accepting now requires explicit override" and the accept button MUST be disabled by default, requiring a separate "override expiry" toggle. The toggle action is itself logged.
5. **No batch operations**: the UI MUST NOT offer "accept all" or "reject all" controls. Each Proposal is reviewed individually.
6. **Read-side parity**: the inbox UI MUST display pending / accepted / rejected / expired counts and let the operator browse each state, with the accepted and rejected directories also visible (for audit trail review, not editing).
7. **Audit trail**: every UI-side accept / reject writes the action via the existing `decision_gate.accept_proposal` / `reject_proposal` functions which are already append-only and atomic (Rule 13 / Rule 14.1 style). The UI never bypasses these functions.

### Rule 13.19: Auto-Generated Proposal Constraints (added 2026-05-28)

When proposals are generated by an automated reflection pipeline (i.e., NOT human-initiated), they MUST satisfy ALL of the following before being written to the inbox. These constraints are enforced by the pipeline itself; proposals that would violate them are dropped (with a `silent_drop` audit row) rather than written.

1. **Pipeline generator tag**: ``generator`` MUST be a versioned pipeline identifier, e.g. ``"auto_reflection_pipeline_v1"``. Human-submitted proposals continue to use a different tag (``"structured_rca_v1"`` or operator-named generators). This split lets reviewers filter by origin.
2. **Source trace IDs mandatory**: ``extra.source_trace_ids`` MUST contain at least one ``trace_id`` whose ``created_ts`` falls in the last 7 calendar days. Without traceable recent evidence, the proposal cannot be reproduced; pipeline drops it.
3. **Default tier = diagnostic_only**: ``extra.tier`` defaults to ``"diagnostic_only"`` unless the underlying RCA reports ``marginal_recovery > 0.05`` (5 percentage-point estimated improvement). Higher tiers (``threshold_tuning`` / ``policy_change``) require both stronger RCA evidence AND Rule 13.7 backtest evidence in ``backtest_evidence``; otherwise pipeline downgrades the tier.
4. **Per-day proposal cap = 3**: at most 3 auto-generated proposals are written per UTC day. Excess RCA findings are written to a `silent_findings.jsonl` log under `reports/reflections/silent/` for later manual review, not to the inbox. The cap protects the L6 reviewer from flood.
5. **Same-target 24h cooldown**: if any proposal (any state, including expired/rejected) for the same ``intervention_target`` was created in the last 24h, the new auto-generated proposal is dropped. Manual proposals do NOT count against this cooldown (operator override).
6. **Metadata source restriction**: every Rule 13.6 field on an auto-generated proposal MUST be derived from structured pipeline output (RCA findings, funnel report numbers, event detector p-values). LLM output is permitted ONLY as a `rationale_text` field separate from required metadata. This preserves Rule 13.12 LLM-cannot-originate-proposals: the structured fields are the proposal, the LLM text is annotation.

Implementation note: the pipeline writes a separate audit row to `reports/reflections/auto_pipeline_audit.jsonl` for every drop (whether by cap, cooldown, or metadata restriction). This is append-only and lets reviewers verify the pipeline is filtering correctly. The audit row schema: `{ts, generator, target, drop_reason, finding_summary}`.

The constraints in Rule 13.19 apply only to proposals where ``generator`` starts with ``"auto_"``. They do NOT relax any other Section 13 rule — Rule 13.1 (proposals never auto-apply), Rule 13.6 (full metadata), Rule 13.17 (reproducibility metadata), Rule 13.18 (L6 UI discipline) all continue to bind.

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

### Rule 14.8: Cutover Date Verification

Cutover day T MUST be recorded as an absolute ISO date (`YYYY-MM-DD`) plus the verified weekday in JST. Relative wording such as "this weekend", "next Sunday", or an unchecked weekday label is not sufficient.

Before a real migration run:

- Verify the weekday against the local calendar.
- Run `python tools/migrate_portfolio_from_project_optimized.py --cutover-date <T> --dry-run`.
- Retain the dry-run evidence path and NAV components in `PROJECT_STATUS.md`.
- Only then run the same command without `--dry-run`.

If T changes, update `PROJECT_STATUS.md`, `docs/01_TASKS.md` if task acceptance changes, and ADR-0008 status text together. A migration based on a mismatched date/weekday label is not accepted as complete.

### Rule 14.9: User-State Watchlist Storage (added 2026-05-28)

Server-side watchlist (`/api/watchlist`) is `user_state`, distinct from portfolio SSoT, with relaxed mutation semantics per Rule 11.3 (user-state vs system-state separation).

1. **Storage**: a single mutable JSON file at `reports/user_state/watchlist.json`. Shape: `{watchlist: [{symbol, added_ts, note}, ...], updated_ts, schema_version}`. The file is overwritten atomically on each mutation (`tempfile + os.replace`); not append-only.
2. **No calibration impact**: watchlist membership is NOT a Rule 14.6 calibration sample input. Adding or removing a symbol cannot bias the prediction / outcome record set.
3. **No background mutation**: only `/api/watchlist/add` and `/api/watchlist/remove` API handlers may write the file. No scheduled job, scanner, or reflection module may mutate watchlist. Mutation by any other writer is detected by `updated_ts + writer_token` fingerprint and rejected on read.
4. **No portfolio cross-write**: adding a ticker to watchlist does NOT create a journal entry, does NOT touch positions, does NOT alter `etf_buyhold`. The two stores never share schema.
5. **Validation**: every watchlist entry MUST pass the same `.T` symbol gate as portfolio (4-digit ticker + `.T` suffix, max watchlist size 100). Invalid input → 422.
6. **Cooling-off cross-reference**: when Rule 12.4 (24-hour cooling-off for new watchlist entries) applies, the cooling-off window is computed from `added_ts` in `watchlist.json`. Removal then re-add resets the cooling-off window (anti-evasion: alert discipline is enforced even if the user games the watchlist).
7. **No share / no broadcast**: watchlist is per-installation, never broadcast to other users or external services. No cloud sync.

This rule is the ADR-0008 §14 user_state carve-out referenced in Rule 11.3.


## 15. Local Beta v0 Deployment Discipline

Local Beta v0 is the only approved "go live" mode for Monday 2026-06-01. It is a single-user, single-machine operating mode for research, observation, manual record keeping, and forward sample collection. It does not certify model edge, calibrated probabilities, paper-trading readiness, broker execution readiness, or notification-channel readiness.

### Rule 15.0: Localhost Scope Only

Local Beta v0 MUST run only on the operator's own machine. The approved bind targets are `127.0.0.1` / `localhost`. The beta MUST NOT be exposed to LAN, VPN, cloud, public internet, reverse proxy, or any multi-user environment without a new deployment rule and a security review.

No login, account system, HTTPS termination, role permissions, or multi-user audit model is required for Local Beta v0 because there is exactly one local operator. The absence of those controls is itself a reason Local Beta v0 may not be widened beyond localhost.

### Rule 15.1: Beta Is Not A Calibration Ship Gate

Local Beta v0 may display uncalibrated research scores, K-fold downgrade state, raw dashboard observations, and manual portfolio records. It MUST NOT display or imply validated win rate, validated probability, live edge, or production trading readiness unless Rule 8.2.2 and Rule 9.4.1 have been honestly satisfied.

Backdated, replayed, accelerated-clock, or bootstrap evidence may guide research diagnostics only. It cannot by itself promote any UI label from `uncalibrated_research_score` / `insufficient_calibration` to calibrated probability.

### Rule 15.2: Pre-Beta Readiness Gate

Before each Local Beta v0 session, the operator must be able to complete a short readiness gate:

- working tree status reviewed and a rollback point or backup path identified;
- API/dashboard smoke test passes on localhost;
- data freshness / market-session labels are visible and not misleading;
- calibration state is visibly downgraded or insufficient unless a valid ship verdict exists;
- no broker/order/paper/live-signal routes are exposed beyond the existing manual-recording carve-outs;
- vectorbt / numba slow backtest tests are separated from the daily smoke gate and do not block dashboard readiness;
- known degraded sections are visibly labeled rather than silently rendered as real data.

If any item fails, Local Beta v0 may still be used only as a debugging session, not as the Monday operating cockpit.

### Rule 15.3: Allowed Activities

Local Beta v0 may be used for:

- pull-only dashboard review;
- symbol search and read-only drill-down;
- K-line, profile, news, macro overlay, strategy-card, and LLM narrative review under the existing no-probability rules;
- server-side watchlist add/remove as user state;
- manual recording of trades or cash events already completed in an external broker;
- proposal inbox review, accept/reject flows, and reflection observability;
- forward prediction/outcome collection through `emit_daily_predictions.py` and `sweep_pending_outcomes.py`;
- local dry-runs of notification discipline without invoking real push channels.

### Rule 15.4: Forbidden Activities

Local Beta v0 MUST NOT:

- submit, route, modify, cancel, or simulate live broker orders;
- run auto-execution or broker integration;
- enable real push notifications before Rule 12 Stage 2 is satisfied and explicitly turned on;
- call the system's output a win rate, probability, expected return, or edge when the calibration gate is downgraded;
- use a UI click, watchlist membership, or manual fill to upgrade a score label;
- run historical accelerated replay as a substitute for forward samples;
- widen access beyond the local operator's machine.

### Rule 15.5: Monday Operating Rhythm

For each trading day during Local Beta v0:

1. Pre-open: run the local smoke gate, confirm data freshness labels, confirm no degraded section is hidden.
2. During session: use the dashboard as a read-only research cockpit. Any real market action happens outside HTR and remains the operator's responsibility.
3. After any external broker action: record the already-completed fill or cash event manually through the approved record-only path.
4. After close: run forward sample collection (`emit` then `sweep`) and retain outcomes for Rule 8.2.1 sunset / Rule 9.4 validation.
5. End of day: record blockers, degraded data, and any rule exception in `PROJECT_STATUS.md`.

### Rule 15.6: Test Gate Split

Local Beta v0 has two verification lanes:

- **Daily smoke lane**: fast API, dashboard, frontend contract, calibration endpoint, watchlist, manual-recording, proposal, and reflection tests. This lane must be suitable for pre-open use.
- **Research regression lane**: slow vectorbt / numba backtest tests, live network smoke tests, and long-running validation jobs. These are important but do not block Local Beta v0 when the daily smoke lane is green and the slow lane is explicitly recorded as deferred.

A slow research regression must not be silently counted as daily readiness. Conversely, a green daily smoke lane must not be claimed as proof of model edge.

### Rule 15.7: Rollback And Stop Conditions

The operator must stop using Local Beta v0 as an operating cockpit and treat it as debugging-only if any of the following occurs:

- dashboard fails to load on localhost;
- prices, market state, or freshness labels are missing or contradictory;
- a mock/degraded section is rendered without a visible warning;
- calibration downgrade disappears without a valid Rule 9.4.1 ship verdict;
- any broker/order/paper/live-signal endpoint appears outside the approved manual-recording carve-outs;
- manual portfolio records no longer derive to a coherent NAV / holdings view;
- forward sample collection starts writing malformed or duplicate prediction/outcome records.

### Rule 15.8: Promotion Beyond Local Beta

Promotion from Local Beta v0 to any broader mode requires a separate rule update. At minimum, the next mode must define access control, exposure boundary, backup/restore process, notification policy, data freshness SLA, and whether the calibration gate remains downgraded or has honestly passed.

### Rule 15.9: Remote Personal Access Mode (added 2026-07-02, P22 — the Rule 15.8 promotion for single-operator remote use)

Remote Personal Access is the FIRST and ONLY approved widening beyond localhost. It exists so the single operator can reach their own cockpit from their own devices. It is NOT multi-user, NOT public, and changes no other Local Beta v0 discipline (Rules 15.1-15.7 continue to bind).

1. **Exposure boundary — private overlay network only.** The approved transport is a private overlay network the operator controls (e.g. Tailscale / WireGuard) or an SSH tunnel. Binding to a public interface, port-forwarding through a router to the internet, reverse proxies on public hosts, and any multi-user access remain FORBIDDEN without a further rule and security review.
2. **Access control — token, fail-closed at startup.** Non-loopback serving is permitted ONLY through the guarded runner (`tools/serve_remote.py`), which MUST refuse to start on a non-loopback bind unless `HTR_ACCESS_TOKEN` is set (fail-closed). When the token is configured, every request — API and pages — MUST present it (header `X-HTR-Token`, `Authorization: Bearer`, or the session cookie set by `/login?token=…`); anything else receives 401. Loopback serving without a token remains exactly Local Beta v0.
3. **Token hygiene.** The token lives in the operator's environment (never committed, never logged in full, never rendered in a page). Rotating it = restarting with a new value; there is no account system to manage.
4. **Same single operator, same carve-outs.** The write surface stays the Rule 11.5 whitelist (manual record-only). Remote access grants no new write path, no broker/order capability, no notification activation (Rule 12 unchanged).
5. **Data stays the operator's.** Remote mode serves the operator's own licensed market data to the operator's own devices. Serving any other person is redistribution and is forbidden (see `docs/04_DATA_AND_OPEN_SOURCE.md` source terms).
6. **Backup / freshness / calibration unchanged.** The Rule 15.2 readiness gate, freshness labeling, and the downgraded calibration state apply identically on remote devices; remote rendering MUST NOT strip degradation banners or disclosures.
7. **Contract tests** assert: guarded runner refuses non-loopback without token; with token configured, un-authenticated API requests 401 and authenticated ones succeed; loopback-no-token behavior unchanged.

### Rule 15.10: Pipeline Health State Contract (added 2026-08-11, P37-01)

`daily_routine`'s `ok` boolean answered one question — did core collection succeed — and was then read as if it answered a different one: is the pipeline healthy. It does not. Research-maintenance steps are non-fatal by deliberate design (Rule 16.6: a diagnostic must never block collection), so their failures were recorded as return codes nobody aggregated and nobody surfaced. Measured on the real log: TDnet polling returned non-zero on **five** afterclose sessions (2026-07-07, 07-17, 07-28, 08-07, 08-10) with `ok: true` every time, and event-universe maintenance reported `event_universe_partial` on 2026-08-10 with `ok: true`. TDnet disclosure documents are served for only ~31 days, so a silent degraded day there is permanent data loss, not a retryable blip.

The fix is a second, separate aggregate — not a redefinition of the first.

1. **`ok` is NOT redefined.** It continues to mean exactly "core collection succeeded" (candidate refresh produced a snapshot, and emit/sweep either wrote new forward samples or was an idempotent re-run of an already-collected session). Existing consumers keep working unchanged. Silently widening a published boolean is how a green signal loses its meaning without anyone noticing.
2. **`health_status ∈ {healthy, degraded, failed}` is the aggregate.**
   - `failed` — a CORE gate failed: candidate refresh, emit, sweep, or the zero-new-sample guard.
   - `degraded` — core collection succeeded, but one or more DECLARED non-fatal steps failed, were partial, or did not run.
   - `healthy` — core collection succeeded and every declared step succeeded.
   - **Invariant:** `health_status == "failed"` if and only if `ok is False`. `degraded` therefore never masquerades as a failure and never hides inside `healthy`.
3. **Every non-fatal step declares a stable component code.** `degraded_components` carries `{component, code, detail}` per degraded step, and `components` carries the full roster with its status. Codes are stable strings, not free text or bare return codes: a degradation that cannot be named cannot be counted, and one that cannot be counted cannot be trended.
4. **Silence is not health.** A declared component that produced no result — an early return that skipped it, a crashed step, a missing field — is `not_run`, which is `degraded`. It is never absent from the roster and never scored as success. (Rule 11.9.4: absence of data is reported, never imputed in the flattering direction.)
5. **Perishable components are labelled as such.** A component whose upstream data expires (`tdnet_poll`, `revision_capture` — TDnet ~31 days) declares `perishable: true`, because "we will catch it on the next run" is false for those and true for the rest. Health reporting MUST carry that distinction.
6. **CLI exit codes:** `0` healthy, `3` degraded, `1` failed. `3` is already this codebase's partial-maintenance code (`refresh_htr_price_db` exits 0/3/1 under P35-02), so degraded is distinguishable from both success and failure by a scheduler without parsing JSON.
7. **A degraded aggregate is forbidden without its components.** Dashboard, CLI, and log MUST name every degraded component wherever the aggregate is shown. Rendering "degraded" alone recreates the defect one level up. This binds all four UI variants (Rule 11.7 content red-lines) and extends Rule 15.5.1's "confirm no degraded section is hidden".
8. **Health is not edge.** `healthy` says data collection worked. It is not evidence about any signal, and MUST NOT be reported alongside or in place of a research verdict (Rule 15.6's converse clause).
9. **Contract tests** MUST cover: a TDnet non-zero exit, an event-universe partial, multiple simultaneous degradations, a core failure, a fully healthy run, a component that did not run at all, and an idempotent re-run.

一个任务只有同时满足以下条件才能标记 done：

- 对应文件已创建或修改。
- 验证命令已运行。
- 输出或失败原因已写入 `PROJECT_STATUS.md`。
- 没有绕过 advice-only 约束。

## 16. Cross-Sectional & Multi-Signal Research Discipline (ADR-0011)

These rules govern *how* signals are evaluated and combined for the edge-seeking effort defined in ADR-0010. They are a methodology layer: they ADD to, and never relax, Rule 5.1 (execution / tradability gate) and ADR-0010's anti-overfit promotion gate. Evidence base: Phase −1 (2026-06-23) — see ADR-0011 Context.

### Rule 16.0: Break-Even Derivation Precedes Build

Before any signal or strategy is implemented in code, a written break-even derivation MUST show it can clear transaction costs: the cost hurdle `IC > τ·c_rt / σ_r` (or an equivalent net-of-cost inequality), computed with the account's *real* τ (per-rebalance turnover), `c_rt` (round-trip cost from the Rule 5.1 JPX tick ladder), and σ_r (cross-sectional dispersion of forward returns at the intended horizon). If the realistically achievable IC does not exceed the hurdle, the signal is NOT built. This is the cheap a-priori pre-screen *before* ADR-0010's heavy promotion gate. (Phase −1: 1–3D hurdles 0.09–0.17 are unachievable; ≥5D hurdles 0.025–0.07 are the only economic region.)

### Rule 16.1: Cross-Sectional Rank-IC Is the Primary Signal Metric

A signal's primary evaluation is per-day cross-sectional Rank-IC (Spearman of score vs forward *relative* return), averaged over days with a t-stat. Absolute-direction AUC/Brier may be reported as secondary context but MUST NOT be the gate. Returns MUST be beta-residualized when candidate betas are heterogeneous (else the common market factor leaks into the score).

### Rule 16.2: Live-Only Evaluation, No Bootstrap Pooling

Signal skill statistics (IC, AUC, dispersion) MUST be computed on live (forward) samples only. Pooling bootstrap/backdated samples with live samples for skill evaluation is forbidden — it produces Simpson's-paradox artifacts (2026-06-23: pooled AUC 0.57 masked live AUC 0.46; pooled vs live Rank-IC diverged in sign). Bootstrap may seed calibration per ADR-0006 but never enters signal-skill evaluation.

### Rule 16.3: Orthogonality Before Stacking

A signal may enter a composite only if BOTH hold: (a) its standalone live Rank-IC is positive with t-stat clearing the Rule 16.6 tier; AND (b) its cross-sectional rank-correlation with every existing composite member satisfies |ρ| < 0.5. Rationale: the composite IC ceiling is IC/√ρ, so stacking correlated signals buys ~nothing; redundant signals are rejected.

### Rule 16.4: Equal-Weight + Shrinkage by Default

Composites combine members by equal-weight rank-average by default — the maximal-shrinkage, noise-optimal choice for weak/noisy signals (James-Stein domination for ≥3 estimators). Fitted/optimized weights are a parameter change requiring Rule 13.3's sample tier (≥300) AND purged+embargoed CV evidence; absent that, fitted weights are rejected as overfitting.

### Rule 16.5: Complexity Cap (K vs N)

The number of stacked signals K MUST stay small relative to the live sample count N (random-matrix noise bound: spurious eigenvalues up to (1+√(K/N))²). At the current N≈600, K is capped at single digits. A correlation/weight matrix estimated where K/N is not small is treated as noise and may not drive allocation.

### Rule 16.6: Horizon, Decay, and Promotion Lifecycle

Signals operate at horizon ≥ 5 trading days (Phase −1: shorter horizons cannot clear the cost hurdle, Rule 16.0). IC decay across horizons MUST be reported; a signal with IC only at 1D (microstructure noise) is presumed noise, not edge. Promotion research → shadow → (human-gated) influence reuses the Rule 13.14 shadow/canary lifecycle AND ADR-0010's anti-overfit promotion gate (Deflated Sharpe + PBO/CPCV + embargo ≥ label horizon + written numeric prior + Harvey t≥3 + append-only forward log). Clearing the Rule 16.0 hurdle is NECESSARY, never SUFFICIENT, for promotion.

### Rule 16.7: Versioning and No Silent Refit

Composite membership and any weights are versioned; changes go through the Rule 13 reflection/proposal gate with reproducibility metadata (Rule 13.17). Re-fitting on the same window to chase a metric (Brier, IC) is forbidden — that is the V5–V8 and isotonic-calibration failure mode (overfit that dies out-of-sample).

## 17. Owner Risk Mandate & Sleeve Discipline (ADR-0012, declared 2026-07-13)

The owner declared the ¥400k account to be EXPERIMENTAL capital with an explicit −75% drawdown tolerance and requested a risk-accepting architecture. Section 17 encodes that mandate as *discipline*, not as an edge claim. It ADDS risk-budget structure on top of every existing rule and RELAXES NONE of them — in particular Rule 3 (advice-only, manual external execution), Section 8 (no fabricated probabilities), Section 12 (anti-FOMO), Section 14 (journal SSoT), and Section 16 (signal promotion gates) all continue to bind. Accepting more risk does not create edge; it widens outcome variance. The only positive-expectation engine in the mandate is compensated market beta.

### Rule 17.0: Mandate Provenance and Boundary

The mandate's parameters (experimental capital, kill-switch floor, target exposure, band, sleeve caps) are OWNER-DECLARED values recorded in `configs/risk_mandate.json`, not model outputs. Any change to them follows Rule 4 (field / old / new / reason / expected impact / verification) in the PROJECT_STATUS Change Log. The quantitative derivation behind the defaults (fractional-Kelly λ ≤ 0.75 from P(hit floor) ≤ 10%; target β-adjusted exposure 1.4× NAV) is recorded in ADR-0012 with its assumptions (μ ≈ 5.5%, σ ≈ 18%) labelled as RESEARCH ASSUMPTIONS with material estimation uncertainty — never as predictions.

### Rule 17.1: Sleeve Architecture

Every holding maps to exactly ONE sleeve via the declarative `sleeve_map` in `configs/risk_mandate.json`:

- **Sleeve A — leveraged-beta engine**: broad-market beta (incl. leveraged ETFs). The only sleeve with positive expected return, sourced from the equity risk premium.
- **Sleeve B — value/E-P live experiment**: the Rule 16 forward-track skin-in-the-game basket. Expected return ≈ 0 (evidence purchase); it exists to measure real execution cost and to bind the owner to the 63D verdict (~2026-08-26).
- **Sleeve C — conviction bets**: owner discretionary positions. ZERO demonstrated edge, pure variance, budgeted as loss-tolerant.

The Section 14 journal is NOT modified — sleeve assignment is a read-only overlay; journal schema, append-only invariants, and entry ids are untouched. A holding whose symbol is absent from `sleeve_map` is FAIL-CLOSED into an `UNASSIGNED` bucket and surfaced as a warning; it is never silently attributed to a sleeve.

### Rule 17.2: Exposure Target and Rebalance Discipline

Total β-adjusted exposure (Σ market_value × β × leverage_factor) targets the mandate ratio (default 1.4× NAV) inside the declared band (default [1.2, 1.6]). Outside the band, the surface MUST show a rebalance-needed state. The drawdown arithmetic that justifies the mandate (P(hit floor) ≤ 10%) is CONDITIONAL on constant-fraction rebalancing — reducing yen exposure as NAV falls. Freezing yen exposure while NAV falls (letting effective leverage rise) voids the derivation; the surface must say so when the band is breached downward. All rebalancing is owner-executed externally (Rule 3).

### Rule 17.3: Kill-Switch

The kill-switch is a NAV floor (default ¥100,000). The surfaces MUST always show the current buffer to the floor. If NAV < floor: the risk surface enters `kill_switch_breached`, every sleeve's status line becomes "exit + post-mortem", and no advice other than de-risking may be rendered until the owner records a post-mortem. The system cannot and does not liquidate anything (Rule 3) — the kill-switch is a binding commitment device for the owner, not an order.

### Rule 17.4: Sleeve C Discipline

1. **Cap**: C's mark-to-market value MUST NOT exceed 20% of NAV (aligned with Rule 12.5). Above the cap → `cap_breached` flag; no BUY advice for C names while breached.
2. **No averaging down**: adding to a C position below its re-underwrite price is a discipline violation and MUST be flagged if recorded.
3. **Mandatory thesis**: every C position REQUIRES a written thesis + invalidation trigger recorded in the mandate config. A C position without one carries a `thesis_missing` flag (fail-closed, prominently rendered) — re-underwriting without a thesis is "forgot to sell" wearing a new badge.
4. **Review trigger**: a C position at or below its declared review drawdown (default −20% from re-underwrite) enters `review_required`: the surface demands an explicit owner decision (hold-with-reason / exit), logged.
5. Re-underwriting a position INTO C (e.g. 8035.T at ¥71,300 on 2026-07-13) resets its discipline reference price but never its recorded cost basis or P&L history.
6. **Bilateral exit bracket (P26)**: a C position MAY declare a two-sided close-price exit bracket (`exit_upper_jpy` / `exit_lower_jpy`) in its thesis. Evaluated on the latest CLOSE only (aligned with the afterclose routine, never intraday). On either side breached → `exit_triggered` flag + an advice-only line ("下 S 株市价卖单了结"; execution is the owner's, Rule 3). The bracket exists to defeat the disposition effect: a "sell only when it recovers to cost" plan is a one-sided cap on upside with unlimited downside — the exit condition MUST be a declared price on BOTH sides and MUST NEVER be the entry cost. A written bracket IS a valid thesis (Rule 17.4.3) for a position being wound down.
7. **Discipline-flag sunset (P26)**: `thesis_missing` / `review_required` are not allowed to linger indefinitely. When a flag has been continuously open for ≥ `flag_sunset_sessions` sessions (default 7), read from the append-only risk trace, it escalates: the daily snapshot prints a SUNSET line and records it in the trace `sunset` field, demanding resolution (write thesis / re-underwrite / exit). A rule with no deadline is a rule the owner can defer forever.

### Rule 17.5: Sleeve B Pre-Commitment

B is sized for MEASUREMENT (execution cost, slippage vs paper IC), not for alpha allocation — default cap ¥60k. Its lifecycle is PRE-COMMITTED at declaration to defeat mid-experiment FOMO/sunk-cost resizing: (a) if the 63D E/P live forward verdict (~2026-08-26, Rule 16.6 gate) CONFIRMS, B may grow to at most ¥150k via a Rule 4 change; (b) if it FAILS, B is unwound back to Sleeve A. Resizing B before the verdict, in either direction, is a Section 12 discipline violation absent a written owner override.

### Rule 17.6: Honest Expectation Labelling

Every risk-mandate surface (API, frontend card, daily trace) MUST carry per-sleeve expectation labels: A = "compensated beta (期望≈2×股权溢价−损耗)"; B = "期望≈0(买证据)"; C = "零 demonstrated edge·纯方差". Doubling-time arithmetic, probability-of-floor numbers, and Kelly math may be shown ONLY as derivation provenance with assumptions attached — never as forward-looking win rates. All Section 8 prohibitions on probability/win-rate language apply unchanged.

### Rule 17.7: Sector Look-Through (P26)

The risk surface MUST expose PENETRATED sector concentration, not just direct holdings: a passive index ETF carries embedded sector weight that a per-symbol view hides. Look-through = Σ direct theme-tagged market value (`theme_map`) + Σ (index-ETF market value × leverage_factor × embedded sector weight `benchmark_sector_weights`). Embedded weights are RESEARCH ESTIMATES (Rule 17.6) with an `_asof`/`_source` stamp, refreshed via Rule 4 when materially stale — never predictions. This answers "how much of NAV actually moves with theme X" across sleeves; e.g. a single semi conviction name plus TOPIX's ~11% semi weight (direct + leveraged) can put >20% of NAV on one theme while each holding looks small. Observability only: it issues no advice and no target — surfacing the number is the control.
