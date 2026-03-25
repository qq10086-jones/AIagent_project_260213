# DESIGN: v1.5 Sharpe and Fundamental Patch

Status: Draft
Date: 2026-03-24
Owner: PM + Quant Architect

## 0. Calibration Update (2026-03-25)

This document remains directionally correct, but the project state needs a PM/quant calibration update after the latest paper-trading and factor-health review.

### 0.1 Current truth, not aspiration

- The strongest part of the system is now the research-and-operations loop:
  - shadow mode comparison
  - promotion governance
  - closed-loop paper execution
  - execution provenance and fail-closed pricing checks
- The weakest part is still not compute speed. It is evidence quality:
  - PIT fundamentals are implemented but not yet validated end to end with live J-Quants credentials
  - the current production mode remains `ridge`
  - `shadow_eq` / `shadow_ic` look better in backtest, but formal promotion is still `hold`
- The quant slice should therefore be described as a strong research-and-ops platform, not yet a fully proven institutional production strategy stack.

### 0.2 Strategy calibration

- The current live family is still dominated by technical / momentum-adjacent signals.
- The more important issue is not only style concentration. It is that only a small subset of factors currently clear the t-stat guard consistently.
- Equal-weight and IC-weight shadow composites can produce nearly identical backtest results.
  - Working interpretation: the active factor set is still too collinear.
  - Consequence: factor de-correlation is now a higher priority than adding another similar technical factor.

### 0.3 Risk calibration

- Liquidity assumptions were too loose in the default config.
  - `min_adv=5,000,000 JPY` and `max_adv_frac=1.0` are too permissive for a realistic small-/mid-cap Japan workflow.
- Drawdown scaling exists, but ex-ante volatility targeting did not.
  - That means the system could de-risk after pain, but not target a steadier risk budget before the drawdown happened.
- News sentiment should remain an overlay / gating layer until it proves incremental IC over the technical and PIT-fundamental baseline.

### 0.4 Updated design rule

The near-term optimization priority is now:

1. tighten liquidity and participation assumptions
2. preserve cash buffers end to end
3. add portfolio volatility targeting
4. add factor de-correlation diagnostics / residualized composites
5. validate PIT fundamentals live
6. require longer paper evidence before mode promotion

## 1. Decision Summary

- Sharpe should be added to the current system, but not as a standalone primary regression target in P0.
- In the current project, Sharpe is most useful in three places:
  - as a portfolio-level evaluation and promotion metric
  - as a small family of risk-adjusted factors
  - as a portfolio construction and risk-governance constraint
- Public financial statements should be added as point-in-time (PIT) factors, not as a free-form LLM guess about business quality.
- The first production goal is not "replace everything with Sharpe regression". The first goal is "add risk-adjusted and fundamental factors into the existing IC and shadow-promotion framework".

## 2. Current Baseline

Current production truth from the codebase:

- Default config still uses `signal_mode: ridge`.
- Research results already show `shadow_eq` and `shadow_ic` are stronger candidates than `ridge`.
- The current model is:
  - price and volume derived factors
  - `PanelRidge` cross-sectional prediction
  - long-only mean-variance optimizer
  - ATR and max-drawdown risk controls
  - optional news overlay
  - IC-based post-trade learning

Current Sharpe-like usage already exists, but only partially:

- `vol_adj_mom20 = ret20 / vol20` is already a Sharpe-like factor proxy.
- `target = forward_return / vol20` is already a risk-adjusted target.
- The system does not currently implement:
  - Sharpe regression
  - Sharpe-optimized signal promotion logic
  - a dedicated rolling-Sharpe factor family
  - PIT fundamental factors from earnings and financial statements

## 3. Why Sharpe Matters, and Why It Should Not Be the First Alpha Target

### 3.1 Why Sharpe is useful

- Sharpe is the cleanest compact summary of return per unit of risk.
- It is a better model-promotion metric than raw return alone.
- It can improve ranking quality when used as a factor family for trend quality and return consistency.
- It makes the PM/operator discussion cleaner because it ties expected reward to volatility cost.

### 3.2 Why "Sharpe regression" is not the first patch

- Rolling single-name Sharpe is noisy on short windows.
- The denominator is unstable when realized volatility is very low.
- A pure Sharpe objective can over-favor low-vol names and suppress alpha.
- The current system is cross-sectional. It is usually more stable to predict future return or relative score, then apply risk penalties, than to regress directly on Sharpe.

Conclusion:

- Add Sharpe as a metric, a factor family, and a promotion gate first.
- Do not introduce `sharpe_regression` as the new default production model in this patch.

## 4. Patch Goals

This patch adds two missing layers:

1. A PIT fundamental operator based on public financial statements and earnings events
2. A formal Sharpe-aware evaluation and risk-adjusted factor layer

The patch should answer two questions:

1. Is the stock fundamentally improving or deteriorating?
2. Is the return path efficient enough relative to risk to justify allocation?

## 5. Data Layer Patch

### 5.1 New tables

Add two new point-in-time tables in `japan_market.db`:

#### `fundamental_snapshots`

Purpose:
- store PIT fundamentals aligned to report availability time

Recommended columns:
- `symbol`
- `fiscal_period_end`
- `published_ts`
- `available_ts`
- `source`
- `currency`
- `revenue`
- `operating_income`
- `net_income`
- `eps`
- `book_value_per_share`
- `dividend_per_share`
- `operating_cf`
- `free_cf`
- `total_assets`
- `total_equity`
- `total_debt`
- `shares_outstanding`
- `guidance_revenue`
- `guidance_operating_income`
- `guidance_eps`

#### `earnings_events`

Purpose:
- store earnings and guidance deltas as event features

Recommended columns:
- `symbol`
- `published_ts`
- `event_type`
- `headline`
- `revenue_yoy`
- `operating_income_yoy`
- `eps_yoy`
- `guidance_delta_revenue`
- `guidance_delta_op`
- `guidance_delta_eps`
- `surprise_score`
- `source`

### 5.2 PIT rule

No factor may be visible before `available_ts`.

This is mandatory. The patch must fail closed if report publication time is unknown and the data source cannot provide a safe availability timestamp.

## 6. New Factor Families

### 6.1 Fundamental factors

The first batch should remain simple and interpretable:

- `value_bp`
  - book-to-price
- `quality_roe`
  - return on equity
- `quality_cfo`
  - operating cash flow / net income
- `margin_op`
  - operating margin
- `growth_rev_yoy`
  - revenue growth
- `growth_op_yoy`
  - operating income growth
- `guidance_delta`
  - normalized management guidance revision
- `leverage_safety`
  - inverse leverage proxy, for example equity / debt or interest coverage proxy
- `dividend_yield`
  - dividend per share / price

### 6.2 Sharpe-aware and risk-adjusted factors

Add a small family instead of a single magic signal:

- `sharpe_20`
  - rolling 20-day return / rolling 20-day volatility
- `sharpe_60`
  - rolling 60-day return / rolling 60-day volatility
- `sortino_60`
  - rolling 60-day return / downside volatility
- `vol_stability`
  - inverse volatility-of-volatility
- `resid_mom_60`
  - optional, market-adjusted momentum quality

### 6.3 Normalization rules

For both fundamental and Sharpe-aware factors:

- winsorize by cross-section
- z-score by cross-section
- neutralize by sector where possible
- require minimum history before activation
- do not allow any factor to enter live composite before IC logging exists

## 7. Signal Construction Patch

### 7.1 What stays the same

Keep the current framework:

- `ridge` remains a benchmark
- `shadow_eq` and `shadow_ic` remain comparison baselines
- IC logging and `factor_registry` remain the promotion backbone

### 7.2 What changes

Add new factors into the existing feature stack and compare them through the same shadow process.

Recommended new experimental mode after data is ready:

- `shadow_hybrid_ic`

Definition:
- price factors and fundamental factors are both logged into `factor_signals`
- each factor receives IC-based weight update
- portfolio alpha score is the weighted sum of active price, fundamental, and risk-adjusted factors

### 7.3 Explicit non-goal

This patch does not promote:

- an LLM-only business-quality score
- a direct "predict future Sharpe" regression model
- any mode selected only because its Sharpe is high on one backtest

## 8. Promotion and Governance Rules

Sharpe should become a formal promotion gate, not a narrative nice-to-have.

A signal mode may be promoted only if it passes all of:

- positive production-universe IC mean
- acceptable t-stat on production-universe diagnostics
- backtest Sharpe improvement versus current baseline
- no worse than threshold max drawdown
- acceptable turnover and cost stability
- paper-trading stability over multiple cycles

Recommended initial promotion thresholds:

- backtest Sharpe > 1.0
- max drawdown <= 20%
- no material deterioration in turnover-adjusted performance
- paper-trading period >= 20 trading sessions before promotion

## 9. File and Module Impact

Expected code touch set for the future implementation patch:

- `trade_schema.py`
  - add new PIT tables
- `db_update.py` or new `update_fundamentals.py`
  - ingest J-Quants or fallback source
- `ss7_sqlite_news_overlay.py`
  - extend feature loader
- `compute_ic.py`
  - extend `FACTOR_NAMES`
  - reuse existing IC and shadow framework
- `daily_run.py`
  - add fundamentals update stage before model run
- `config.yaml`
  - add `fundamental` and `risk_adjusted` config blocks
- `make_decision.py`
  - expose factor-family contributions in decision report

## 10. Rollout Plan

### P0

- Document design and promotion rules
- Add schema and config stubs
- Make Sharpe an explicit reporting metric in decision and backtest outputs
- Tighten default liquidity assumptions so paper/backtest does not rely on 100% ADV participation
- Preserve sub-100% target weights as real cash buffers instead of renormalizing them away
- Add portfolio volatility targeting as a de-risking layer without introducing leverage

### P1

- Add PIT financial-statement ingestion
- Add first batch of fundamental factors
- Add first batch of Sharpe-aware factors
- Extend `compute_ic.py` and diagnostics
- Validate J-Quants ingestion end to end with real credentials and `available_ts` discipline
- Add factor correlation diagnostics and at least one residualized or de-correlated composite candidate

### P2

- Add `shadow_hybrid_ic`
- compare against `ridge`, `shadow_eq`, `shadow_ic`
- add operator-facing report fields for factor-family contribution
- keep news sentiment as overlay by default; only promote toward standalone alpha if incremental IC is demonstrated
- require evidence across at least one earnings season before promoting hybrid or news-sensitive modes

### P3

- Evaluate whether optimizer-level ex-ante Sharpe objective is worth adding
- only after P1/P2 proves the factor layer is real and stable
- expand test coverage from utility-only tests to pipeline, decision, and paper-execution hardening

## 11. Final Judgment

Sharpe is worth adding, but the mathematically correct way in this project is:

- first as a promotion metric
- second as a small factor family
- third as a portfolio-level risk objective

It is not worth rushing in as a new standalone regression target before PIT fundamentals and risk-adjusted diagnostics are in place.
