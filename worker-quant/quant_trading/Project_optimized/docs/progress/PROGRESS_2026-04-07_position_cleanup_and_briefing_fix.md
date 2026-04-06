# Progress: 2026-04-07 Position Cleanup & Briefing Report Fix

## Status

Manual positions (3 stocks, sprint strategy) are now the sole source of truth in
the database. The v2 briefing report has been fixed to correctly display positions,
stop-loss, and take-profit levels.

## What Changed

### 1. Database position cleanup

All legacy `default` strategy data was removed from SQLite:

- `positions` table: deleted 73 rows of paper-trading history
- `fills` table: deleted all `default` strategy fills
- `account_snapshots` table: deleted all `default` strategy snapshots
- Pre-entry `sprint` snapshots (before 2026-04-06) also cleaned

Remaining data reflects only the user's actual holdings:

| Symbol | Qty | Avg Cost | Strategy |
|--------|-----|----------|----------|
| 4005.T | 100 | 528.8 | sprint |
| 7267.T | 100 | 1259.5 | sprint |
| 9432.T | 400 | 156.9 | sprint |

Account snapshot: NAV=400,770, Cash=158,410

### 2. Fixed: `read_live_state` asof matching (quant_briefing.py)

**Bug**: `read_live_state` used exact `asof=?` matching for positions and
account_snapshots. On non-trading days (e.g., 2026-04-07 querying positions
written on 2026-04-06), this returned empty results, causing the report to
show "empty positions."

**Fix**: Changed to `asof<=?` with `MAX(asof)` lookup for positions, and
`asof<=? ORDER BY asof DESC` for account_snapshots. This correctly retrieves
the latest available data on or before the target date.

### 3. Added: Take-profit calculation in `_enrich_positions_with_stop_loss`

The function previously computed only stop-loss. Now also computes take-profit:

- `TP_MULT = 8.0` (ATR multiplier, vs 6.0 for stop-loss, giving R:R > 1)
- `TP_FLOOR = 0.08` (8% minimum)
- `TP_CAP = 0.30` (30% maximum)
- New fields: `take_profit_price`, `take_profit_pct`, `tp_triggered`

### 4. Fixed: v2 report `pos_v2` hardcoding

**Bug**: `write_report_v2` constructed `pos_v2` with hardcoded
`stop_triggered: False` and no stop-loss/take-profit fields, discarding
all data computed by `_enrich_positions_with_stop_loss`.

**Fix**: `pos_v2` now passes through all enriched fields:
`stop_loss_price`, `stop_loss_pct`, `stop_triggered`,
`take_profit_price`, `take_profit_pct`, `tp_triggered`, `stop_note`,
and derives `action_hint` from trigger state (HOLD / STOP_LOSS / TAKE_PROFIT).

### 5. Updated: v2 markdown table (section 3)

The position health table now includes:

| Column | Before | After |
|--------|--------|-------|
| Headers | 6 columns | 8 columns (+止盈线, +状态) |
| 止损线 | "N/A" always | ¥423(-20.0%) |
| 止盈线 | missing | ¥687(+30.0%) |
| 状态 | missing | HOLD / ⚠️止损 / 🎯止盈 |
| 数值格式 | raw float | formatted (.0f / .1f) |

### 6. Fixed: morning_briefing.bat missing --strategy_id

Added `--strategy_id sprint` to the scheduled morning briefing command
so automated runs correctly pick up sprint strategy positions.

### 7. Fixed: stdout summary in quant_briefing.py main()

The terminal summary now prints position count alongside order status,
instead of only checking orders (which always showed "空仓无挂单").

## System Decision Output

The daily_run on 2026-04-06 produced:

- `target_weights.csv`: only `4005.T` at 25% weight
- `orders_proposal.csv`: SELL 7267.T (100), SELL 9432.T (400)
- Benchmark state: `off` (MA20 < MA60, risk-off)
- Kelly: fallback 25% (sample_count=0, min_samples=30 not met)

User decision: follow system recommendation, sell 7267.T and 9432.T,
retain only 4005.T.

## Files Modified

- `quant_briefing.py` — 5 fixes (read_live_state, enrichment, v2 pos_v2, v2 table, stdout)
- `morning_briefing.bat` — added `--strategy_id sprint`
- `japan_market.db` — cleaned default strategy data

## Validation

- `python quant_briefing.py --mode full --output-version v2 --strategy_id sprint`
  - positions correctly displayed with stop-loss and take-profit
  - v2 JSON includes full enriched fields
- `python daily_run.py --config config.yaml` — completed RC=0
- Stock analysis generated: `reports/stock_analysis/2026-04-07_4005T_7267T_9432T.md`

## Current Position Summary (4005.T only after planned sells)

| Parameter | Value |
|-----------|-------|
| Cost | 528.8 |
| Current | 533.2 |
| P&L | +0.83% |
| Stop-loss | ¥423 (-20.0%, ATR-capped) |
| Take-profit | ¥687 (+30.0%, ATR-capped) |
| Benchmark | risk-off (MA20 < MA60) |
| Kelly weight | 25% fallback |
