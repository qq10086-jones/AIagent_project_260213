# Windows Task Scheduler Setup — 2026-04-15 multi-strategy pipeline

## Schedule layout (all times JST)

| Time | Task | Script | Purpose |
|---|---|---|---|
| 09:00 | morning_briefing | `scheduled_morning_briefing.cmd` (existing) | Pre-open briefing |
| **10:00** | **monthly_rebalance** (NEW) | `scheduled_monthly_rebalance.cmd` | Emits real alpha-factor orders on month-start; no-ops other days |
| 14:45 | intraday_decision (DEMOTED) | legacy `intraday_decision.py` | Watchlist-only mode (default now) |
| **17:00** | **all_strategies** (NEW) | `scheduled_all_strategies.cmd` | Full pipeline: daily_run + paper lanes + dashboard |

## Replacing legacy tasks

- **Disable** the old `scheduled_daily_run.cmd` trigger — its work is
  now called by `run_all_strategies.py` as step 1.
- Keep the 14:45 `intraday_decision.py` task active; it remains useful
  for monitoring (now defaults to `--watchlist_only`).

## Creating a new task (Windows 11)

```powershell
# Example: 17:00 JST daily
schtasks /Create /SC DAILY /TN "worker-quant\all_strategies" `
    /TR "E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized\scheduled_all_strategies.cmd" `
    /ST 17:00 /F

# Example: 10:00 JST daily (script is calendar-gated)
schtasks /Create /SC DAILY /TN "worker-quant\monthly_rebalance" `
    /TR "E:\AIagent_project_260213\worker-quant\quant_trading\Project_optimized\scheduled_monthly_rebalance.cmd" `
    /ST 10:00 /F
```

## Verifying

```powershell
schtasks /Query /TN "worker-quant\all_strategies" /V /FO LIST
# Logs: logs\scheduled_all_strategies_YYYY-MM-DD.log
```

## Manual runs (test)

```bash
# Full pipeline right now (uses today's asof)
python run_all_strategies.py

# Force monthly rebalance even if not first of month (test only)
python monthly_rebalance.py --force
```

## Failure mode notes

- `run_all_strategies.py` exits non-zero if ANY sub-step fails, but each
  step runs independently (allow_fail=True). Check the log to see which.
- `monthly_rebalance.py` SKIPs silently on non-first-of-month days —
  this is correct behavior; check the log to confirm.
- If `intraday_decision.py` is emitting orders instead of watchlist,
  check that `--watchlist_only` is the active default (post-2026-04-15).
