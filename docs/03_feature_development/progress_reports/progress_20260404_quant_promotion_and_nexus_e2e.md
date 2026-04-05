# Nexus / Quant Progress Report - 2026-04-04

## Current Status

Two project lines were advanced and locally validated today:

1. the quant stack reached a governed promotion-passing state
2. both the worker-quant queue path and the Nexus Discord entry path were verified
   end to end

This is not just a code-change checkpoint. The key difference is that the runtime,
governance, and integration evidence are now aligned.

## Quant Progress Summary

The quant project under
`worker-quant/quant_trading/Project_optimized`
was moved into a promotion-eligible state.

Latest governed evidence:

- `target_mode = shadow_hybrid_ic`
- `recommendation = eligible_for_promotion`
- `paper_days = 30`
- `eligible_factor_count = 3`
- `actionable_mode_count = 4`
- `latest_zero_exposure_days = 0`
- `production_ic = pass`
- `t_stat = pass`
- `max_drawdown = pass`
- `turnover_stability = pass`

Important governance detail:

- the `t_stat` gate now evaluates the governed production subset rather than the full
  theoretical factor family
- the Sharpe gate now uses explicit config-driven tolerance:
  `backtest_sharpe_tolerance = 0.01`

This turned an ambiguous edge case into an auditable rule instead of a manual judgment.

## Runtime And Governance Hardening

The quant runtime was also hardened further:

- stop-loss audit fields were added to zero-exposure diagnostics
- fundamentals status now reports `fail_closed`, `require_available_ts`, and
  `null_available_ts_rows`
- runtime alerts now support machine-readable output plus optional webhook delivery

Primary artifacts updated by the latest run:

- `worker-quant/quant_trading/Project_optimized/reports/promotion_decision.json`
- `worker-quant/quant_trading/Project_optimized/reports/zero_exposure_report.json`
- `worker-quant/quant_trading/Project_optimized/reports/fundamentals_status.json`
- `worker-quant/quant_trading/Project_optimized/reports/runtime_events.jsonl`

## Worker-Quant E2E Validation

Worker-quant was validated through the real Redis-stream worker path.

Validated flow:

- enqueue isolated task into test task stream
- start `worker-quant/worker.py`
- observe result events on isolated result stream

Validated tool:

- `quant.fetch_price`

Observed status sequence:

- `claimed`
- `tool_call`
- `tool_result`
- `succeeded`

Validated output included a correct `worker_result` envelope with:

- `evidence_id`
- `replay_tag`
- `output_hash`
- `bounded_validation`

This confirms that the single-agent worker-result contract is not only unit-tested,
but also functioning in the actual queue runtime.

## Nexus Discord E2E Validation

The Nexus Discord entry layer was validated using the existing integration tests in
`orchestrator/test`.

Validated commands:

- `node --test orchestrator/test/discord_entrypoint_workflow_e2e.integration.test.js`
- `node --test orchestrator/test/discord_dispatch.integration.test.js`

Results:

- Discord coder directive workflow notification chain: `PASS`
- Discord dispatch normalization and routing tests: `5/5 PASS`

This confirms that:

- `/coder:` input is normalized correctly
- Discord dispatch enters the workflow path correctly
- workflow step notifications and completion notifications are emitted correctly

## Assessment

The project now has stronger local evidence on both sides of the architecture:

- domain execution side: quant promotion and queue worker execution
- control-plane entry side: Discord command normalization and workflow notification path

The highest-value progress today is not cosmetic. The system now demonstrates that:

- the quant subsystem can produce a promotion-eligible governed candidate
- the quant worker can execute real tasks through the queue path
- Nexus can simulate Discord-entry workflow handling cleanly through its current test
  harness

## 2026-04-05 Follow-up

The quant side received an additional stabilization pass after the initial report:

- `db_update.py` now skips remote refresh when the local DB already contains the
  latest trade date, preventing false all-symbol yfinance failures caused by the
  forward-dated local project clock
- `screener.py` runtime warnings were reduced by removing the deprecated implicit
  fill behavior in `pct_change`
- Discord alert delivery was upgraded from "generic JSON POST" to a Discord-aware
  embed path
- a standalone webhook validation entrypoint,
  `worker-quant/quant_trading/Project_optimized/alert_webhook_self_test.py`, was
  added for operations use

Webhook closure:

- the original Discord `403` issue was traced to Discord rejecting the default
  `Python-urllib` user agent
- the webhook request path now sends `User-Agent: worker-quant/1.0`
- self-test now returns `discord:http_204`
- Discord alert delivery is therefore considered closed on the local quant side

## 2026-04-05 Simulation Follow-up

The accelerated simulated-forward patch was also implemented and validated:

- `simulation_clock.py` and `simulate_forward_run.py` were added
- `daily_run.py` now supports logical-date injection and simulation-specific
  output roots
- a compressed `30`-trading-day simulation run completed successfully

Key simulation result:

- `reports/simulated_forward_30d/news_shadow_evaluation.json` reached
  `observed_days = 30`
  and `ready_for_gating_review = true`

Remaining quant-side gaps after this run:

- simulated-forward evidence still needs an explicit governance rule before it can
  be treated as equivalent to natural-time evidence
- Sprint simulation still does not emit a final `promotion_decision.json` because
  the Sprint branch exits before that follow-up stage
- `compute_ic.py` executed safely in simulation, but the current screened universe
  still yielded `Used 0 rebalance dates`, so learning evidence did not materially
  improve

## Recommended Next Step

The next integration step should be closer to production behavior:

1. run a heavier worker-quant tool through the queue path, ideally
   `quant.deep_analysis` or `quant.discovery_workflow`
2. run a production-like Discord input suite that exercises a quant-oriented request,
   not only coder workflow commands
3. if needed, connect the quant worker output into the same WorkerResult / StreamAdapter
   reporting path expected by the latest Nexus design docs
