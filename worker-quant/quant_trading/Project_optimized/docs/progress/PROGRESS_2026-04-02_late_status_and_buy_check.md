# Progress: 2026-04-02 Late Status and Buy Check

## Status

The quant pipeline is operational again as of 2026-04-02, and the local Windows runtime has now been stabilized.

There is still no valid buy opportunity under the current live rules, but the issue is no longer "pipeline broken". The current problem is "pipeline runs, governance evidence is cleaner, but latest target weights remain zero".

## Operational Result

- `daily_run.py` now completes successfully for `asof=2026-04-02` on this machine
- the default runtime has been switched to a stable fundamentals path:
  - `fundamental.source: "yfinance"`
  - `fail_closed: false`
  - `require_available_ts: false`
- `jquants_v2` remains available as an optional enhancement path, but is no longer the default daily dependency
- Windows console encoding issues in `daily_run.py`, `update_fundamentals.py`, and `news_to_db.py` were patched so the daily run no longer crashes under `cp932`

## Buy-Signal Assessment

Current system state remains:

- `orders = 0`
- `NAV = 400,000 JPY`
- recommendation = `hold`

This is currently expected behavior, not a new quant crash.

The latest local QA produced a more precise diagnosis:

- latest `target_weights.csv` row is still zero
- last non-zero target was `2026-03-12`
- latest `zero_exposure_report.json` identifies the current immediate blocker as `benchmark_risk_off`
- all compared modes (`ridge`, `shadow_eq`, `shadow_ic`, `shadow_hybrid_ic`) currently export zero latest weights, so simply switching signal mode would not create trades today

## QA View

- pipeline health: restored and locally repeatable
- governance stats: improved
- live trading readiness: guarded, not promoted
- buy opportunity today: `no`

Additional QA findings from the local runtime pass:

- `paper_execute.py` is not the primary failure point; it runs and now writes `paper_no_orders` for zero-order runs
- `evaluate_promotion.py` was updated to count paper activity from `account_snapshots` as well as fills, so `paper_days` now correctly shows `2` instead of staying at `0`
- factor quality remains weak for production:
  - only `mom_consist` currently qualifies as a production-eligible factor
  - `ret20`, `rsi14`, `slope60`, and `vol_adj_mom20` are now explicitly listed as cleanup / exclusion candidates in the factor cleanup report

## Next Step

Keep the stabilized daily pipeline running, but do not treat "no orders" as a pure execution-layer bug.

The next quant tasks should focus on:

- tracing why the system has remained zero-exposure since `2026-03-12` even before the benchmark regime turned fully `off`
- deciding whether the current 20-trading-day rebalance cadence is too sparse for the desired live behavior
- cleaning low-confidence factors out of the effective production candidate set before any strategy-promotion discussion
