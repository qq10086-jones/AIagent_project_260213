# Progress: 2026-04-03 Runtime Governance Hardening

## Status

The quant project was reworked and revalidated locally on 2026-04-03 with two goals:

1. keep the daily pipeline runnable on this machine without blocking on first-time J-Quants refresh
2. turn the current "no orders" state into an explicit QA-governed failure mode instead of a vague runtime symptom

The full local run completed successfully with:

- `asof = 2026-04-03`
- `run_id = 2026-04-03__6182eb6a07`
- `paper_execute status = paper_no_orders`
- `promotion recommendation = hold`

This means the project is no longer suffering from a fragile main-path runtime dependency, but it is still not trading. The remaining blocker is strategy readiness and exposure recovery, not pipeline survivability.

## What Changed

### 1. Fundamentals refresh was removed from the critical daily path

`daily_run.py` now treats fundamentals refresh as a cache-maintenance concern rather than a hard dependency of the daily production run.

- default config keeps `fundamental.source = "jquants_v2"`
- but `fundamental.run_on_main_path = false`
- `fundamental.blocking = false`
- `fundamental.allow_stale_on_failure = true`

This allows the daily chain to continue using the latest PIT snapshots already stored in `japan_market.db` instead of stalling on a full-universe J-Quants refresh.

New diagnostics were added:

- `reports/fundamentals_status.json`
- `reports/fundamentals_status.md`

Latest validated state:

- `step_status = skipped_main_path`
- `latest_source = yfinance`
- `latest_symbols = 96`
- `latest_available_ts = 2026-04-02T23:31:45`

### 2. Benchmark regime is now diagnosable and less black-box

The benchmark regime path in `ss7_sqlite_news_overlay.py` was hardened in two ways:

- `off` no longer implies a silent hard zero by default
- benchmark diagnostics are now exported with price / MA / hysteresis context

New config knobs:

- `model.benchmark_off_scale = 0.25`
- `model.benchmark_caution_scale = 0.60`

New zero-exposure report fields now include:

- benchmark price
- benchmark fast MA
- benchmark slow MA
- benchmark enter / exit lines
- fast-minus-slow %
- price-minus-slow %

Latest validated state:

- `benchmark_state = off`
- `benchmark_scale = 0.25`
- `risk_off = false`
- `primary_cause = benchmark_regime_capped_exposure`

This is an important clarification. The project is not in a pure hard risk-off shutdown anymore; it is in a benchmark-capped state, but still exporting zero latest target weights.

### 3. Zero exposure is now a QA alert, not just an observation

`compare_signal_modes_report.py` was upgraded to enforce a zero-exposure QA window.

New summary fields:

- `latest_zero_exposure_days`
- `max_zero_exposure_days_allowed`
- `zero_exposure_alert`
- `qa.status`

Latest validated result:

- `all_modes_zero_now = true`
- `latest_zero_exposure_days = 5`
- `max_zero_exposure_days_allowed = 3`
- `zero_exposure_alert = true`
- `qa.status = alert`

This makes the current state operationally visible: the system has remained flat too long to be treated as a healthy live posture.

### 4. Promotion gating now includes real production-readiness checks

`evaluate_promotion.py` was hardened with three new gates:

- `eligible_factors`
- `zero_exposure_window`
- `actionable_mode_available`

New thresholds added to config:

- `promotion.min_eligible_factors = 3`
- `promotion.max_zero_exposure_days = 3`
- `promotion.require_actionable_mode = true`

Latest validated gate failures:

- `eligible_factors: actual=1, threshold=3`
- `zero_exposure_window: actual=5, threshold=3`
- `actionable_mode_available: actual=0, threshold=1`

The project now fails promotion for the right reasons. It is not enough that the chain runs; the strategy also has to be tradable and supported by more than one production-eligible factor.

### 5. Factor health reporting now carries QA context

`factor_health_report.py` now includes the new QA state from promotion:

- `eligible_factor_count`
- `actionable_mode_count`
- `latest_zero_exposure_days`
- `latest_weights_zero`

Latest validated factor-health outcome:

- production-eligible factors: `mom_consist` only
- cleanup candidates: `ret20`, `rsi14`, `slope60`, `vol_adj_mom20`

## Latest Runtime Outcome

Validated via:

- `python daily_run.py --config config.yaml`

Key outputs:

- `reports/fundamentals_status.json`
- `reports/zero_exposure_report.json`
- `reports/signal_mode_compare_report.json`
- `reports/promotion_decision.json`
- `reports/factor_health_report.json`

Decision-package evidence:

- `artifacts/decision/2026-04-03/2026-04-03__6182eb6a07/decision_snapshot.json`
- `artifacts/decision/2026-04-03/2026-04-03__6182eb6a07/execution_report.md`

Outcome summary:

- daily chain completed successfully
- data update succeeded
- screening succeeded
- backtest/report generation succeeded
- decision packaging succeeded
- paper execution succeeded
- latest order count remained `0`

## Current Assessment

The quant project is in a materially better engineering state than before this patch.

- the daily run is now resilient to slow first-time fundamentals refresh
- the benchmark regime path is inspectable
- prolonged flat exposure is now an explicit QA failure condition
- promotion logic now rejects non-tradable strategy states more honestly

However, the strategy is still not ready for live promotion.

The current blocker is no longer "runtime may crash" and no longer simply "benchmark risk-off". The latest evidence points to a deeper issue:

- all compared modes still export zero latest weights
- zero exposure has persisted for 5 trading days
- only one factor is currently production-eligible

## Recommended Next Step

The next work should focus on strategy recovery, not infrastructure:

1. trace why latest target weights remain zero even under `benchmark_scale = 0.25`
2. inspect whether `news_gate = 0.0` is suppressing the final target set too aggressively
3. rework the latest export logic so the system can hold reduced-risk nonzero exposure instead of falling into repeated flat mode
4. widen the production-eligible factor set before any further promotion discussion
