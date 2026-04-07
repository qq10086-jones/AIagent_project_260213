# Quant Worker Contract for Nexus v4

## Purpose

This document defines how `worker-quant` and `Project_optimized` should be treated inside Nexus v4.

It is not a research note.
It is a worker contract for:

- orchestrator
- PM
- architect
- QA
- release / governance review

## Role in Nexus

`worker-quant` is a terminal execution worker for quant-related workflows.

It is responsible for:

- executing approved quant tools
- generating reports and evidence artifacts
- updating quant-local trading state
- emitting queryable execution evidence

It is not responsible for:

- top-level user intent interpretation
- workflow decomposition across domains
- approval policy decisions
- release gating outside its own artifact contract

Those responsibilities remain with Nexus orchestrator and policy layers.

## Supported Capability Surface

Authoritative capability registration lives in:

- [capability_registry.json](E:/AIagent_project_260213/configs/registry/capability_registry.json)
- [tools.json](E:/AIagent_project_260213/configs/tools.json)
- [quant_design_latest.md](E:/AIagent_project_260213/docs/01_design/quant/quant_design_latest.md)

Current quant-facing tools relevant to this project:

| Tool | Purpose | Current implementation anchor |
|------|---------|-------------------------------|
| `quant.fetch_price` | Quote / price retrieval | `worker-quant/worker.py` |
| `quant.deep_analysis` | Single-symbol analysis and report | `worker-quant/worker.py` |
| `quant.discovery_workflow` | Candidate discovery and position proposal | `worker-quant/worker.py` |
| `quant.compute_news_risk_factor` | News risk extraction | `worker-quant/worker.py` |
| `quant.calc_limit_price` | Order support utility | `worker-quant/worker.py` |
| `quant.run_optimized_pipeline` | Full quant execution pipeline | `worker-quant/quant_trading/Project_optimized/run_pipeline.py` |
| `portfolio.set_account` | Account state write | `worker-quant/worker.py` |
| `portfolio.record_fill` | Fill recording | `worker-quant/worker.py` |

## Tool-to-Script Mapping

For `Project_optimized`, the operational chain is:

1. `db_update.py`
2. `screener.py`
3. `ss7_sqlite_news_overlay.py`
4. `make_decision.py`
5. `post_trade.py`
6. `compute_ic.py`

The current pipeline entrypoint is:

- [run_pipeline.py](E:/AIagent_project_260213/worker-quant/quant_trading/Project_optimized/run_pipeline.py)

The current daily operator entrypoint is:

- [daily_run.py](E:/AIagent_project_260213/worker-quant/quant_trading/Project_optimized/daily_run.py)

## Input Contract

### For `quant.run_optimized_pipeline`

Minimum required context:

- DB path
- execution mode / signal mode
- screener parameters
- risk parameters
- output directory

Recommended future explicit params:

- `signal_mode`
- `compare_signal_modes`
- `paper_trading`
- `approval_context`
- `asof`

### For `quant.deep_analysis`

Minimum required context:

- symbol
- market
- optional capital and risk profile

## Output Contract

For Nexus `quant_execution`, the worker must produce normalized artifacts compatible with registry requirements.

Required artifacts for `quant_execution`:

- `risk_report`
- `plan`
- `execution_log`

Recommended concrete mapping for this project:

| Required artifact | Current / target implementation |
|-------------------|---------------------------------|
| `risk_report` | strategy summary + drawdown / stop-loss / signal mode report |
| `plan` | target weights + decision package + mode selection summary |
| `execution_log` | pipeline step log + signal comparison + audit trail |

Additional domain artifacts:

- `target_weights.csv`
- `weights_history.csv`
- `signal_mode_compare.csv`
- `strategy_report.html`
- `strategy_report_extras.html`
- `weights_heatmap.html`
- `learning_audit` rows
- `factor_signals` rows
- `screening_history` rows

## State and Storage Boundary

### Nexus-wide

- Queue / orchestration state: Redis Streams
- global metadata / assets / facts: Postgres + MinIO

### Quant-local

- trading/account state and research state: SQLite
- path: `worker-quant/quant_trading/Project_optimized/japan_market.db`

Important boundary rule:

SQLite is a quant-local execution store, not the global Nexus source of truth for orchestration metadata.

## Approval and Risk Boundary

`quant.run_optimized_pipeline` is medium-risk and approval-gated in tool config.

That means:

- worker-quant may execute the pipeline only after orchestrator/policy approval
- worker-quant must not silently escalate itself to real-money action
- real-money transition requires explicit human/PM/operator approval

## Safe Degradation Rules

Quant worker must degrade safely under these conditions:

1. Missing or stale market data
   Result: fail closed, produce execution log with no-trade / incomplete-data reason.

2. Unsupported or unapproved signal mode
   Result: fall back to approved baseline mode or reject execution.

3. Missing screening universe
   Result: backfill only for research evaluation; do not silently treat historical backfill as live execution.

4. Learning uncertainty / insufficient evidence
   Result: do not update live weights; record audit only.

5. News pipeline unavailable
   Result: disable overlay and continue with explicit note, unless policy requires strict dependency.

## Queryability Requirements

Every quant execution should remain queryable after the fact.

Minimum evidence sources:

- `screening_history`
- `factor_signals`
- `factor_registry`
- `learning_audit`
- generated reports in `reports/`
- decision package artifacts

Required PM questions that should be answerable post-run:

- What signal mode was used?
- Was it approved?
- What universe was traded?
- What risk controls were active?
- What output artifacts were produced?
- Did the run degrade, fallback, or fail closed?

## Current Compliance Status

### Aligned

- quant capability exists and is registered
- project has executable domain logic
- learning evidence tables exist
- screened-universe evaluation is implemented
- production-side signal logging is implemented

### Not yet fully aligned

- config-level `signal_mode` is not fully wired through Nexus entrypoints
- artifact outputs are not yet normalized to `risk_report / plan / execution_log`
- fail-closed semantics are not yet documented at script-entry level
- paper-trading operator mode is not yet first-class

## Immediate Nexus Compliance Tasks

1. Add explicit `signal_mode` support to the quant pipeline contract.
2. Normalize pipeline outputs to required Nexus quant artifacts.
3. Add a fail-closed execution summary for no-data / degraded / rejected runs.
4. Add paper-trading mode as the default promotion step before live-capital usage.
5. Update orchestrator-facing docs and prompts to reference this contract instead of assuming quant is a standalone subsystem.
