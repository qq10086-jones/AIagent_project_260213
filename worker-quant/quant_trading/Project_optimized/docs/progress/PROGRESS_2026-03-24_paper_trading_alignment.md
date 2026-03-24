# Quant Project Progress - 2026-03-24

## Current Focus
`worker-quant` shifted from "paper-trading capable in principle" to "paper-trading closed loop with pricing QA guardrails".

The main work in this session was not model research. It was execution correctness:

- paper orders now flow through a simulated execution bridge into `fills`
- positions / NAV / execution reports update automatically after paper fills
- same-day simulation no longer silently reuses stale prior-day prices
- legacy unverified paper fills are now explicitly marked as untrusted history

## What Changed

### 1. Paper-Trading Closed Loop

Implemented a full local paper-trading execution bridge:

- added `paper_execute.py`
- converts `orders.status=proposed` into simulated `fills`
- rebuilds `positions`
- rebuilds `account_snapshots`
- regenerates `execution_report`
- syncs `orders.status` and `decision_runs.status`

This closes the operational gap between:

`make_decision.py -> orders -> fills -> positions -> NAV -> report`

### 2. Sizing Logic Fix

`make_decision.py` no longer relies only on the manual `--cash` argument for sizing.

It now prefers the latest `account_snapshots.cash` and `account_snapshots.nav` when available, so new paper decisions size against the actual paper account instead of a fixed notional placeholder.

This fixed the prior failure mode where fresh proposals could oversize positions and push the simulated account into nonsensical negative cash purely due to stale capital assumptions.

### 3. Real-Time Quote Path

Added intraday quote infrastructure for same-day paper simulation:

- added `intraday_quotes` table
- added `intraday_update.py`
- added `market_data_utils.py`
- added `yf_runtime.py` to force `yfinance` cache into a writable local project directory

`paper_execute.py` now supports `price_mode=latest` and will preferentially use the latest intraday quote rather than yesterday's daily bar.

`daily_run.py` and `config.yaml` were updated so the paper path defaults to `latest`.

### 4. Stale-Data Fail Closed

Before this session, the execution path could simulate "today" using the last available prior-day price if the database had not yet refreshed.

This is now blocked.

`make_decision.py` and `paper_execute.py` will:

- refresh market data when requested
- validate requested `asof` against the latest available DB date
- fail closed if the DB is still behind the requested trade date

This removes the earlier silent mismatch between wall-clock date and simulated execution date.

### 5. Pricing QA / Provenance Guardrails

The `fills` table was extended with explicit pricing provenance fields:

- `price_source`
- `price_ts`
- `price_mode`
- `quote_open`
- `quote_high`
- `quote_low`
- `quote_close`
- `price_validated`
- `validation_note`

`paper_execute.py` now validates simulated fills against the source quote range:

- if a simulated fill price falls outside the quote `low-high` range, execution fails
- new paper fills therefore have auditable quote provenance and a validation result

This is the main QA hardening added in this session.

### 6. Legacy Paper Fill Repair

Discovered that older `paper_trader_bridge` fills had no auditable pricing provenance and could not be treated as reliable execution truth.

Actions taken:

- relabeled old bridge fills to `legacy_paper_trader_bridge_unverified`
- marked them `price_validated=0`
- added `repair_legacy_paper_fills.py`
- fetched 1-minute historical bars
- repaired the legacy `5020.T`, `7267.T`, and `7201.T` 2026-03-23 15:20 fills from intraday quotes

Repaired prices:

- `5020.T`: `1286.0`
- `7267.T`: `1283.5`
- `7201.T`: `330.8999938964844`

Each repaired fill now carries:

- `source = repaired_from_intraday_quote`
- `price_source = yfinance:1m`
- `price_ts = 2026-03-23T06:20:00+00:00`
- `price_validated = 1`

## Operational Result

After repairing sizing and re-running paper execution, the paper account was brought back to a coherent state.

Latest verified paper account state during this session:

- cash: about `37,494 JPY`
- holdings: `4005.T` 400 shares, `9432.T` 1100 shares
- same-day real-time revaluation using intraday quotes: NAV about `400,614 JPY`

This means the paper account can now be evaluated with:

- historical daily mark
- same-day intraday mark
- auditable execution provenance

## Key Risk Still Open

The system now has a robust paper-execution path, but the market-data stack is still not fully institutional-grade.

Current limitations:

- intraday quotes still depend on Yahoo / `yfinance`
- browser-based quote verification is not yet integrated into the quant worker
- no broker-native or J-Quants real-time source has been wired in yet
- legacy fills already stored in historical tables may still need selective cleanup if they were created before provenance checks existed

## Recommended Next Step

The next engineering step should not be model tuning.

It should be quote-source alignment and cross-source verification:

- add a quote verification layer
- compare `intraday_quotes` against browser-visible Yahoo price or a stronger real-time source
- define a deviation threshold
- fail execution or raise an operator warning when the live quote mismatch exceeds tolerance

## Files Added / Updated

Added:

- `paper_execute.py`
- `market_data_utils.py`
- `intraday_update.py`
- `yf_runtime.py`
- `repair_legacy_paper_fills.py`

Updated:

- `make_decision.py`
- `daily_run.py`
- `config.yaml`
- `trade_schema.py`
- `import_fills.py`
- `execution_report.py`

## PM Judgment

The project is now materially closer to real paper-trading readiness than it was before this session.

The important improvement is not alpha.

It is that paper execution now has:

- a closed loop
- fail-closed date checks
- real-time quote support
- explicit pricing provenance
- a path to historical execution repair

That is the right direction for production readiness.
