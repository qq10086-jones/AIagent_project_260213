# TASKS: 2026-03-24 System Hardening and Sharpe/Fundamental Patch

Reference design:
- `worker-quant/quant_trading/Project_optimized/docs/design/DESIGN_v1.5_Intelligence_Augmented_Quant.md`

## [S] Immediate / 1-2 Days

- [x] Convert the current Sharpe discussion into a formal promotion rule: report Sharpe, Sortino, max drawdown, and turnover for every signal mode comparison.
- [x] Keep `ridge` as benchmark only; do not promote a new default only from narrative judgment.
- [x] Add clean config placeholders for `fundamental` and `risk_adjusted` factor families in `config.yaml`.
- [x] Add a patch note in operator-facing output explaining that Sharpe is now a governance metric, not a direct regression target.
- [x] Add fail-closed wording for PIT fundamentals: if `available_ts` is unknown, the factor must not enter live scoring.

## [M] Near-Term / 1-2 Weeks

- [x] Add SQLite schema for `fundamental_snapshots`.
- [x] Add SQLite schema for `earnings_events`.
- [x] Implement a PIT fundamentals ingestion stage from J-Quants first, with fallback source only for research mode.
- [x] Compute first-batch fundamental factors:
  - `value_bp`
  - `quality_roe`
  - `quality_cfo`
  - `margin_op`
  - `growth_rev_yoy`
  - `growth_op_yoy`
  - `guidance_delta`
  - `leverage_safety`
  - `dividend_yield`
- [x] Compute first-batch risk-adjusted factors:
  - `sharpe_20`
  - `sharpe_60`
  - `sortino_60`
  - `vol_stability`
- [x] Extend `compute_ic.py` to log and evaluate the new factors through the existing IC workflow.
- [x] Add sector-neutral normalization and winsorization before factor logging.
- [x] Extend decision output so PM/operator can see price-factor, fundamental-factor, and risk-adjusted-factor contributions separately.

## [P1] Model Comparison / 2-3 Weeks

- [x] Add `shadow_hybrid_ic` mode after PIT fundamentals are available.
- [x] Compare `ridge`, `shadow_eq`, `shadow_ic`, and `shadow_hybrid_ic` under the same execution-cost model.
- [x] Add promotion gates:
  - positive production-universe IC
  - acceptable t-stat
  - Sharpe improvement versus baseline
  - no max-drawdown breach
  - acceptable turnover stability
- [x] Require at least 20 trading sessions of paper-trading stability before default-mode promotion.

## [P2] Productization / 1 Month+

- [x] Add a daily fundamentals update stage into `daily_run.py`.
- [x] Add factor-health report for PM use: IC, ICIR, Sharpe contribution, drawdown contribution, turnover contribution.
- [x] Add event-study diagnostics for earnings surprises and guidance revisions.
- [x] Evaluate whether optimizer-level ex-ante Sharpe objective is worth adding after hybrid factor evidence is stable.

## [R1] 2026-03-25 PM/Quant Recalibration

- [x] Tighten default liquidity posture in config and pipeline:
  - raise screener liquidity floor
  - reduce default ADV participation cap
- [x] Preserve cash buffers end to end instead of normalizing every target-weight vector to 100% gross exposure.
- [x] Add portfolio volatility-target controls to the backtest / target-weight generation path.
- [ ] Add the same cash-buffer / volatility-target semantics to every downstream sizing and execution audit report that still assumes weights sum to exactly 1.
- [ ] Surface operator-facing diagnostics for:
  - target weight sum
  - forecast portfolio volatility
  - volatility-target scale applied

## [R2] Evidence and Model Hardening

- [ ] Add factor collinearity diagnostics:
  - rolling factor correlation matrix
  - effective rank / redundancy flags
  - warning when `shadow_eq` and `shadow_ic` rankings are materially identical
- [ ] Add one de-correlated composite candidate:
  - residualized momentum
  - PCA-neutralized family score
  - or Gram-Schmidt orthogonalized shadow composite
- [ ] Validate J-Quants PIT ingestion end to end with live credentials and real `available_ts`.
- [ ] Keep news sentiment as overlay by default; require incremental IC evidence before promoting it to a standalone alpha factor.
- [ ] Require promotion evidence across at least one earnings season before switching the default mode to a hybrid or news-sensitive stack.

## [R3] Test Coverage and Execution Safety

- [ ] Add tests for `run_pipeline.py` error handling and config propagation.
- [ ] Add tests for `make_decision.py` so sub-100% target weights preserve cash instead of being renormalized.
- [ ] Add tests for `ss7_sqlite_news_overlay.py` volatility-target scaling and weight-sum behavior.
- [ ] Add tests for `paper_execute.py` quote validation and latest-price fail-closed behavior.

## Explicit Non-Tasks

- [ ] Do not ship a standalone `sharpe_regression` mode in this patch.
- [ ] Do not let LLM-only narrative business scoring bypass PIT factors and IC validation.
- [ ] Do not promote any signal mode on Sharpe alone without IC, drawdown, and paper-trading evidence.
