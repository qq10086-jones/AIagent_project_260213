# PIT Parity Check — asof=2026-04-08

Database: `japan_market.db`  |  n_symbols in A: 2631  |  stored in DB: 96

## A vs B — does SQL asof bound change what the current code computes?

- rows compared: 2631  cols: 19  cells: 49989
- cells with zero diff: 49945
- cells with NaN diff: 44
- cells non-zero diff: 0
- max abs diff: 0
- mean abs diff: 0

**Interpretation**: if non-zero cells == 0 and max_abs_diff == 0, the SQL
fix is a performance-only change; no contamination existed via this path
(caller already did `close[close.index <= asof_dt]` slicing).

## A vs C — does existing feature_daily match the PIT-correct recompute?

- rows compared: 96  cols: 19  cells: 1824
- cells with zero diff: 1824
- cells with NaN diff: 0
- cells non-zero diff: 0
- max abs diff: 0
- mean abs diff: 0

**Interpretation**: if the stored feature_daily matches a fresh PIT-
correct recompute, no historical rebuild is required.

## Conclusion

- **A ≡ B**: SQL asof bound is performance-only. Existing compute path was already PIT-correct at the caller level.
- **A ≡ C**: feature_daily stored values match PIT recompute. No rebuild needed.
