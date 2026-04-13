# Reality Check — sprint (2026-02-27 → 2026-04-13)

_generated 2026-04-13T14:39:47Z_  
_n_snapshots = 7_

> ⚠️ n_snapshots < 20 — Sharpe/IR/t-stat are not statistically meaningful yet. Read the level of the returns, not the ratios.

## Headline

- NAV: 400,100 → 400,045  (-0.01%)
- Max drawdown (snapshot-to-snapshot): -0.52%
- Annualised Sharpe (naive): -0.08
- Daily-return t-stat: -0.01

## vs Benchmarks (cumulative)

| benchmark | cum return | sharpe | notes |
|---|---|---|---|
| **strategy (sprint)** | **-0.01%** | -0.08 | — |
| TOPIX ETF (1321.T) price-return | -3.21% | -1.57 | price-return only; total-return unavailable without dividend data |
| Held-universe equal-weight | -11.12% | -6.87 | members: 9432.T, 7267.T, 4005.T, 3041.T |
| Cash (0%) | +0.00% | n/a | ignores JPY cash rate (~0%) |
| Sector-neutral | n/a | n/a | TBD — requires TSE 33 sector mapping per Phase -1 D-1 data hygiene |

## Excess

- Strategy − TOPIX = +3.19%
- Strategy − held-universe = +11.11%
- Information ratio vs TOPIX (annualised, naive): 1.62

## NAV series

| date | NAV | TOPIX close | held-univ avg |
|---|---|---|---|
| 2026-02-27 | 400,100 | 61,120.00 | 705.88 |
| 2026-04-06 | 400,100 | 55,900.00 | 634.33 |
| 2026-04-07 | 399,670 | 55,950.00 | 628.78 |
| 2026-04-08 | 402,150 | 58,960.00 | 645.68 |
| 2026-04-09 | 400,392 | 58,500.00 | 641.20 |
| 2026-04-10 | 400,545 | 59,520.00 | 635.55 |
| 2026-04-13 | 400,045 | 59,160.00 | 627.35 |

## Interpretation guide

1. If **strategy cum return < TOPIX cum return** after costs: the active strategy is not adding value vs just buying 1321.T.
2. If **strategy cum return < held-universe equal-weight**: even picking the same tickers but equal-weighting beats the strategy → the timing/sizing is destroying alpha.
3. Sharpe / IR at n<20 are directional only. Do not promote, demote, or change parameters based on them.
4. TBD sector-neutral benchmark (Phase -1 D-1): until TSE 33 sector mapping is wired, cannot rule out that any edge is just sector beta.
