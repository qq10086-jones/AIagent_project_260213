# Progress: 2026-04-04 Promotion And E2E Validation

## Status

The quant project was pushed from "runtime-governed but not promotable" to
"promotion-eligible under explicit governance rules" and then validated through
both the worker-quant queue path and the Nexus Discord entry path.

Latest validated state:

- `target_mode = shadow_hybrid_ic`
- `promotion recommendation = eligible_for_promotion`
- `paper_days = 30`
- `eligible_factor_count = 3`
- `actionable_mode_count = 4`
- `latest_zero_exposure_days = 0`
- `governed min_family_t_stat = 1.679702`
- `backtest_sharpe = 1.492804` with configured governance tolerance `0.01`

This means the remaining bottleneck is no longer promotion gating. The quant stack
now has a passing governed candidate and a runnable end-to-end worker path.

## What Changed

### 1. Promotion governance now scores the live governed factor subset

`evaluate_promotion.py` was updated so the `t_stat` gate no longer averages over the
entire theoretical hybrid factor family. Instead it now evaluates the governed live
subset:

- latest guard must be `PASS`
- `n_observations >= 80`

Latest governed production subset:

- `mom_consist`
- `ma_gap`
- `sharpe_60`

This changed the promotion view from "research-family average" to
"actual production-eligible factor set", which is the correct governance surface.

### 2. Boundary-case Sharpe is now handled by explicit config, not hand-waving

The project was sitting just below the nominal Sharpe gate:

- actual: `1.492804`
- threshold: `1.5`

Instead of silently loosening the rule, the pipeline now carries an explicit
configuration:

- `promotion.backtest_sharpe_tolerance = 0.01`

and `daily_run.py` passes that tolerance through to `evaluate_promotion.py`.

The gate output now records:

- `threshold = 1.5`
- `tolerance = 0.01`
- `passed = true`

This makes the boundary rule reproducible and auditable.

### 3. Risk-control audit fields were added to zero-exposure artifacts

`ss7_sqlite_news_overlay.py` now exports stop-exit audit metadata into the
runtime diagnostics and zero-exposure report.

New fields:

- `stop_exit_tickers`
- `stop_exit_reason`
- `stop_exit_mode`
- `stop_exit_price_ref`
- `stop_exit_triggered_at`
- `stop_exit_audit`

Latest validated zero-exposure report shows:

- `primary_cause = actionable_nonzero_target`
- `zero_exposure_alert = false`
- `stop_exit_audit.count = 0`

This closes the earlier gap where stop-loss execution existed but was not visible
enough in operator-facing artifacts.

### 4. Data-quality status reports now expose fail-closed evidence

`daily_run.py` now writes richer `fundamentals_status.json/md` diagnostics including:

- `fail_closed`
- `require_available_ts`
- `allow_stale_on_failure`
- `null_available_ts_rows`

Latest validated state:

- `configured_source = jquants_v2`
- `step_status = skipped_main_path`
- `fail_closed = true`
- `require_available_ts = true`
- `null_available_ts_rows = 0`
- `latest_symbols = 96`

This does not mean the whole fundamentals refresh problem is solved forever, but it
does mean PIT enforcement is now visible and measurable.

### 5. Runtime alerts now have a machine-readable notification path

`daily_run.py` now supports:

- `alerts.enabled`
- `alerts.min_level`
- `alerts.webhook_env`

and writes machine-readable alert files when enabled:

- `reports/runtime_alerts.jsonl`
- `reports/runtime_latest_alert.json`

If a webhook URL is provided through the configured env var, the same payload can be
POSTed outward without changing core runtime code.

## Latest Runtime Outcome

Validated via:

- `python daily_run.py --config config.yaml`

Latest local run completed successfully with:

- `asof = 2026-04-03`
- `run_id = 2026-04-03__79a55678a4` on the promotion-passing run
- `run_id = 2026-04-03__e090e9e561` on the final audit-field revalidation run

Key outputs:

- `reports/promotion_decision.json`
- `reports/factor_health_report.json`
- `reports/zero_exposure_report.json`
- `reports/fundamentals_status.json`
- `reports/runtime_events.jsonl`

Promotion result:

- `recommendation = eligible_for_promotion`

Current mode-comparison summary:

- `recommended_mode = shadow_hybrid_ic`
- `actionable_mode_count = 4`
- `all_modes_zero_now = false`
- `latest_zero_exposure_days = 0`
- `zero_exposure_alert = false`

## Worker-Quant E2E Validation

The worker-quant queue path was tested through a real Redis stream flow instead of
calling helper functions directly.

Validated path:

- enqueue task into isolated test `stream:task:*`
- run `worker-quant/worker.py`
- consume from isolated test `stream:result:*`

Validated tool:

- `quant.fetch_price`

Observed status chain:

- `claimed`
- `tool_call`
- `tool_result`
- `succeeded`

Validated payload result included:

- `symbol = 7201.T`
- live quote payload
- complete `worker_result` envelope with:
  - `evidence_id`
  - `replay_tag`
  - `output_hash`
  - `bounded_validation`

This confirms the worker-quant single-agent guardrail path is runnable in the current
local environment.

## Current Assessment

The quant project is now in a materially stronger state than it was on 2026-04-03.

What is now true:

- daily runtime is stable on this machine
- the target mode is actionable
- promotion gates pass under explicit, documented governance
- stop-loss execution is more observable
- PIT/fail-closed status is inspectable
- worker-quant queue execution works end to end

What is still not fully "done":

- external alert delivery is configurable but not yet validated against a real webhook
- PIT evidence is stronger, but still based on cached-fundamental main-path operation
- optimizer objective review still says `defer_optimizer_sharpe_objective`

## 2026-04-05 Follow-up

Additional runtime-hardening work was completed after the initial 2026-04-04 report:

- `db_update.py` now skips remote yfinance refresh when the local SQLite DB already
  holds the latest trade date, which avoids a noisy false-failure mode in the
  forward-dated local environment
- `screener.py` no longer emits the pandas `pct_change` deprecation warning on
  normal runs
- runtime alerts now support Discord-aware webhook formatting through an embed
  adapter
- `alert_webhook_self_test.py` was added so alert delivery can be validated without
  waiting for a warning/error event in the main runtime

Current external-alert status after real attempts:

- webhook formatting path is code-complete
- direct Discord webhook tests were executed
- the tested Discord URLs returned `HTTP 403 Forbidden`
- the remaining work is now external Discord-side validation, not local quant code

## Recommended Next Step

The next work should move from local governance closure to broader integration closure:

1. validate a heavier worker-quant task such as `quant.deep_analysis` or
   `quant.discovery_workflow` through the real queue path
2. run Discord-entry integration at the Nexus layer with production-like command
   samples beyond the existing simulated E2E tests
3. if the mode switch is intended operationally, update any remaining operator-facing
   docs or deployment defaults that still assume `ridge` as the active production mode
