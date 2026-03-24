# Progress: 2026-03-25 Sharpe/Fundamental Patch Closeout

## Status

- Delivery tasks in `TASKS_2026-03-24_System_Hardening.md` are complete.
- Remaining unchecked items are explicit non-tasks / guardrails, not missing implementation work.

## What Landed

- Added PIT fundamentals storage and ingestion path:
  - `fundamental_snapshots`
  - `earnings_events`
  - `update_fundamentals.py`
- Added first-batch fundamental factors:
  - `value_bp`
  - `quality_roe`
  - `quality_cfo`
  - `margin_op`
  - `growth_rev_yoy`
  - `growth_op_yoy`
  - `guidance_delta`
  - `leverage_safety`
  - `dividend_yield`
- Added first-batch risk-adjusted factors:
  - `sharpe_20`
  - `sharpe_60`
  - `sortino_60`
  - `vol_stability`
- Added `shadow_hybrid_ic` signal mode.
- Added sector-neutral normalization and winsorization in both:
  - `compute_ic.py`
  - `ss7_sqlite_news_overlay.py`
- Added promotion governance:
  - Sharpe
  - Sortino
  - max drawdown
  - turnover
  - production IC
  - t-stat
  - paper-trading day requirement
- Added operator-facing outputs:
  - `promotion_decision.json`
  - `promotion_note.txt`
  - `factor_health_report.json/md`
  - `factor_health_families.csv`
  - `factor_health_factors.csv`
  - `signal_mode_compare_report.json/md`
  - `earnings_event_study.json/csv/md`
  - `optimizer_objective_evaluation.json/md`
  - `factor_family_contributions.csv`
  - `factor_family_summary.json`

## Pipeline Changes

- `daily_run.py` now runs:
  - optional fundamentals update
  - model / backtest
  - learning (`compute_ic.py`)
  - promotion evaluation
  - factor health report
  - signal mode comparison report
  - earnings event study
  - optimizer objective evaluation
- `run_pipeline.py` now mirrors the same research / governance reporting flow.
- `make_decision.py` now copies the new report artifacts into the run-scoped decision package.

## Validation

- `py_compile` passed for all touched Python modules in this patch.
- Smoke runs completed for:
  - `shadow_hybrid_ic`
  - promotion evaluation
  - factor health report
  - mode comparison report
  - earnings event study
  - optimizer objective evaluation
  - decision packaging

## Known Limits

- J-Quants ingestion code path is implemented but not end-to-end validated in this environment.
  - Reason: no active `JQUANTS_MAIL/JQUANTS_PASSWORD` credentials and no live network verification during this patch.
- The smoke reports used short windows and should not be treated as production evidence.
- Real promotion still depends on longer live/paper evidence, not only the newly added governance machinery.
