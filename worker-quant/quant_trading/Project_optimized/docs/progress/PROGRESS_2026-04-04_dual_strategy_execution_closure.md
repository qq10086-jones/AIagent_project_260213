# Progress: 2026-04-04 Dual Strategy Execution Closure

## Status

The dual-strategy implementation has moved from partial wiring to a runnable local
execution path. The main remaining open items are now governance-duration tasks or
capital-threshold tasks, not basic pipeline breakages.

Latest validated local state:

- `daily_run.py --config config.yaml` completes end to end
- active strategy path is `strategy_id = sprint`
- Sprint runs through `sprint_signal.py`
- paper execution finishes without dual-strategy schema collisions
- latest paper result is `paper_no_orders`

The last point is a regime/data outcome, not an execution failure.

## What Changed

### 1. SQLite is now the real dual-strategy source of truth

The execution stack now carries `strategy_id` across:

- `decision_runs`
- `orders`
- `fills`
- `positions`
- `account_snapshots`

Most importantly, schema migration was completed so same-day writes from multiple
strategies no longer collide:

- `positions` primary key is now `(asof, strategy_id, symbol)`
- `account_snapshots` primary key is now `(asof, strategy_id)`

This removed the earlier runtime failure where Sprint and future Harvest paths could
not coexist on the same `asof`.

### 2. `paper_trading_account.json` is diagnostic only

The legacy JSON snapshot is no longer a decision input.

Current state:

- `quant_briefing.py` reads SQLite directly
- `live_trade_advisor.py` reads SQLite directly
- `daily_run.py` exports `paper_trading_account.json` only as a read-only diagnostic
  artifact with explicit source-of-truth metadata

This aligns the implementation with the v3 dual-strategy design and governance docs.

### 3. Post-trade execution analytics are now part of the runtime

`paper_execute.py` now computes execution-quality diagnostics via
`post_trade_analytics(...)` and writes:

- `reports/execution_quality.json`

`daily_run.py` also records an `execution_quality` event into:

- `reports/runtime_events.jsonl`

Tracked metrics now include:

- `fill_count`
- `avg_slippage_bps`
- `total_commission`
- `fill_validation_rate`

### 4. News shadow mode and Sprint gating code paths are implemented

Phase 1 and Phase 2 code surfaces now exist:

- `config.yaml`
  - `news.enabled: true`
  - `news.shadow_only: true`
  - `news.sprint_gating: true`
- `ss7_sqlite_news_overlay.py`
  - shadow-only gate logging to `learning_audit`
- `sprint_signal.py`
  - `news_shadow_evaluation.json`
  - `news_gating_ab_test.json`
  - Sprint-side gating behavior

Governance caveat:

- implementation is complete
- the required 30-day shadow observation window is still pending

### 5. Sprint benchmark scaling now affects exported weights

The Sprint path now applies benchmark regime scaling to actual exported target weights.

Implemented behavior:

- `sprint_signal.py` multiplies target weights by `benchmark_scale`
- `daily_run.py` prefers benchmark scales from the active `strategy_profile`

This closes the earlier mismatch where benchmark state was computed but not fully
applied to Sprint output.

### 6. `ss7` modularization and operator docs were updated

Module surfaces were added for:

- `model_ridge.py`
- `backtest_engine.py`
- `execution_model.py`
- `news_overlay.py`
- `portfolio_optimizer.py`

`CLAUDE.md` was also updated with a dedicated dual-strategy section covering:

- strategy-profile runtime behavior
- Sprint/Kelly/benchmark file map
- recommended run commands

## Validation

Regression coverage now includes:

- `tests/test_kelly_sizer.py`
- `tests/test_risk_controls.py`
- `tests/test_benchmark_regime_v2.py`
- `tests/test_sprint_signal.py`
- `tests/test_db_update_universe.py`
- `tests/test_paper_execute_analytics.py`
- `tests/test_daily_run_helpers.py`
- `tests/test_trade_schema_dual_strategy.py`

Latest result:

- `22` tests passed

Runtime validation:

- `python daily_run.py --config config.yaml`

Observed latest local result:

- `run_id = 2026-04-03__fb8e192dca`
- paper execution completed
- final paper status = `paper_no_orders`

## Current Blockers

The main blocker is now on the market-data and governance side of VIX confirmation,
not on the execution path itself.

Observed on 2026-04-04:

- `db_update.py` attempted to pull `1552.T`
- the symbol returned `all-NA`
- Sprint remained at zero exposure on the latest run

Operational consequence:

- no paper orders were generated

This should be tracked as a data/regime blocker, not as a broken daily-run pipeline.

## Remaining Work

Still open after this closure pass:

- `T11-3`: finish the 30-day news shadow observation window
- long-horizon `L1-L3` tasks tied to NAV thresholds or multi-day paper evidence
- decide whether missing `1552.T` should disable VIX confirmation or continue to fail
  closed under governance

## 2026-04-04 Late Follow-up

The remaining engineering-side ambiguity items were reduced further:

- factor tier review cadence is now enforced from `learning_audit` using
  `factor_promotion_rules.review_frequency_days`
- Sprint news shadow now writes cumulative readiness into
  `reports/news_shadow_evaluation.json` so the 30-day gate is machine-tracked
- VIX-missing behavior is now explicit via `strategy_profiles.sprint.vix_missing_policy`
  instead of relying on implied governance
- runtime alert readiness now writes `reports/runtime_alert_status.json`

This leaves only genuine time-gated or external-ops items, not hidden code debt.

## 2026-04-05 Operational Follow-up

Two additional operational fixes were applied:

- `db_update.py` now treats the existing local `daily_prices` cache as authoritative
  when it already contains the latest expected trade date, instead of forcing a
  remote refresh that cannot succeed in the forward-dated local environment
- Discord webhook alert delivery now has:
  - Discord-specific embed formatting
  - a standalone `alert_webhook_self_test.py` validation entrypoint
  - machine-readable `reports/runtime_alert_self_test.json`

Webhook follow-up result:

- Discord webhook delivery is now working
- root cause was Discord rejecting the default `Python-urllib` user agent
- the request path now sets `User-Agent: worker-quant/1.0`
- the self-test now returns `discord:http_204`

## 2026-04-05 Simulation Integration Follow-up

The accelerated simulated-forward patch is now implemented rather than remaining a
design-only proposal.

New runtime surfaces:

- `simulation_clock.py`
- `simulate_forward_run.py`
- `daily_run.py --asof_override ...`
- isolated simulation roots for:
  - `state/...`
  - `reports/...`
  - `artifacts/...`

Behavior now validated:

- logical-date injection into `daily_run.py`
- simulation-tagged runtime events
- simulation-specific paper/report output separation
- strict-PIT filtering for Sprint news shadow ingestion
- logical-date review timing in `compute_ic.py`

Validation completed locally:

- a 3-trading-day compressed simulation run completed end to end
- a Monday single-day simulation run also completed and exercised the
  `compute_ic.py` branch under simulation mode

Governance note:

- this mode is now a usable engineering-evidence path
- it still must not be labeled as natural-time forward evidence without an
  explicit governance decision

## 2026-04-05 30-Day Simulation Result

A longer compressed-time simulation run has now been executed against the isolated
simulation environment.

Run summary:

- window start: `2026-02-02`
- completed logical trading days: `30`
- latest completed logical day: `2026-03-17`
- failed simulation days: `0`

Key outputs:

- `reports/simulated_forward_30d/simulation_summary.json`
- `reports/simulated_forward_30d/news_shadow_evaluation.json`
- `state/simulated_forward_30d/simulation_state.json`

Observed outcomes:

- `news_shadow_evaluation.json` reached:
  - `observed_days = 30`
  - `remaining_days = 0`
  - `ready_for_gating_review = true`
- the simulation also produced real paper fills on later logical dates
- final simulated read-only account snapshot showed:
  - `asof = 2026-03-17`
  - `nav = 1,868,063.11`

Interpretation:

- the engineering objective of compressing the 30-day shadow window is now proven
- governance still needs to decide how simulated-forward evidence should count
  relative to natural-time evidence

Remaining gaps after the 30-day run:

- the run completed `30` logical trading days, but the configured window still has
  `39` trading days through `2026-03-31`; only the first `30` were executed
- Sprint simulation still returns early after Sprint follow-up, so
  `promotion_decision.json` is not generated in the simulation report root
- `compute_ic.py` now executes under simulation, but the current screened universe
  still produced `Used 0 rebalance dates`, so learning-side evidence did not improve
- this simulation is valid engineering evidence, but still not equivalent to
  natural-time governance evidence without an explicit policy decision
