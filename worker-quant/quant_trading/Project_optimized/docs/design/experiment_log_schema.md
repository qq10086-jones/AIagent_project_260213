# Experiment Log — Preregistration Protocol

**Status**: active from 2026-04-13
**File**: `reports/experiment_log.jsonl` (append-only)
**Module**: `experiment_log.py`

## Why

Deflated Sharpe Ratio and FDR require the **true number of trials** attempted, not just the winners. Without this log, every IC we report is inflated and every "significant" factor may be a survivor. From today onward, **no factor / threshold / rule enters a walk-forward training window without a preregister entry**.

## When to log

Register **before** running the experiment. Categories:

| category | when |
|---|---|
| `factor` | new or modified factor definition |
| `threshold` | signal / risk / promotion cutoff value |
| `rule` | filter, gate, kill-switch |
| `weight_scheme` | factor combination or portfolio weight method |
| `regime_rule` | LLM / news / cross-asset regime rule |
| `cost_param` | fee / slippage / impact parameter |
| `universe` | stock universe change |
| `frequency` | rebalance frequency variant |

## Schema

### Preregister entry
```json
{
  "schema_version": "1.0",
  "experiment_id": "2026-04-13__factor__a3f9c1b2d4e5",
  "ts_utc": "2026-04-13T16:42:11+00:00",
  "status": "preregistered",
  "category": "factor",
  "hypothesis": "roa_op IC > 0 on 5d horizon, JP universe",
  "params": {"horizon_days": 5, "winsorize": 0.01, ...},
  "param_hash": "a3f9c1b2d4e5",
  "data_window": {
    "train_start": "2021-01-01",
    "train_end":   "2024-12-31",
    "validation_start": "2025-01-01",
    "validation_end":   "2025-06-30"
  },
  "author": "jones",
  "notes": "replaces prior version that used ttm revenue"
}
```

### Outcome entry (appended later, never overwrites)
```json
{
  "schema_version": "1.0",
  "experiment_id": "2026-04-13__factor__a3f9c1b2d4e5",
  "ts_utc": "2026-04-20T09:15:00+00:00",
  "status": "executed",   // or "abandoned" / "paradigm_shift"
  "category": "factor",
  "metrics": {"ir": 0.42, "t_stat_nw": 2.1, "n_obs": 1250, "half_life_days": 8}
}
```

## Rules

1. **Append-only.** Never rewrite past entries. Corrections are new entries with a reference in `notes`.
2. **Preregister before execution.** Params must be frozen at preregister time. If params change, it is a new experiment.
3. **Hypothesis must be one concrete testable statement.** Not "explore momentum" — write "12-month momentum, skip last month, IC > 0 on JP large-cap universe".
4. **Data window locked at preregister.** Leaking validation data back into a training window is the exact bias this log exists to prevent.
5. **Paradigm shift trigger.** If the same hypothesis family is modified more than 3 times, `paradigm_shift_flag()` returns True — that family is frozen for the current walk-forward cycle.
6. **Count trials by preregister entries**, not executed. Abandoned experiments still consumed a statistical "try".

## Downstream consumers

- **Deflated Sharpe Ratio**: `N = count_trials(category="factor", since=train_start)` enters the DSR formula.
- **FDR (Benjamini-Hochberg)**: rank p-values from executed entries; `m = N`.
- **Factor whitelist**: can only contain factors whose `executed` outcome beat the DSR-adjusted threshold.
- **Paradigm shift guard**: blocks a factor from entering next walk-forward window if flagged.

## Workflow example

```python
from experiment_log import preregister, record_outcome

# Step 1 — BEFORE running IC calculation
eid = preregister(
    category="factor",
    hypothesis="accruals_inv has negative IC (quality anomaly) on 20d horizon",
    params={"horizon_days": 20, "winsorize_pct": 0.01, "neutralize": "sector"},
    data_window={
        "train_start": "2021-01-01", "train_end": "2024-12-31",
        "validation_start": "2025-01-01", "validation_end": "2025-06-30",
    },
    author="jones",
    notes="v2 definition — uses cfo_assets denominator instead of total_assets",
)

# Step 2 — run IC calculation, collect metrics

# Step 3 — AFTER execution
record_outcome(
    eid,
    metrics={"ic_mean": -0.018, "t_stat_nw": -2.4, "half_life_days": 18, "n_obs": 980},
    status="executed",
)
```

## What NOT to log here

- Individual daily signals / trades (those live in `orders`, `fills`, `signals` tables)
- Operational events (those live in `runtime_events.jsonl`)
- Code refactors without methodology change

## Audit

`experiment_log.jsonl` is committed to git. Any squash/rebase/delete touching it requires a written justification in the PR description. This file is **evidence**, not state.
