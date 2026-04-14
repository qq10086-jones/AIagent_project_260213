# Execution Gap Report

_Generated 2026-04-15_

| Strategy | Tier | State | Months | Real | Paper | WF-Expected | Real−Paper | Real−Expected |
|---|---|---|---|---|---|---|---|---|
| sprint | real | paused_sunk_only | 1.51 | +0.01% | +22.34% | +1.72% | -22.33% | -1.71% |
| high52w | real | active | 0.0 | - | +0.00% | - | - | - |
| amihud | real | active | 0.0 | +0.00% | +0.00% | - | +0.00% | - |
| sprint_paper | paper | active | 0.13 | +22.34% | - | - | - | - |
| high52w_paper | paper | active | 0.0 | +0.00% | - | - | - | - |
| amihud_paper | paper | active | 0.0 | +0.00% | - | - | - | - |
| amihud_k5_paper | paper | active | 0.0 | +0.00% | - | - | - | - |
| amihud_k30_paper | paper | active | 0.0 | +0.00% | - | - | - | - |
| amihud_adv50_paper | paper | active | 0.0 | - | - | - | - | - |
| min_ret_paper | paper | active | 0.0 | +0.00% | - | - | - | - |
| mom_high52w_paper | paper | active | 0.0 | +0.00% | - | - | - | - |

## Diagnostic rules
- `Real−Paper` large negative → real slippage > assumed bps. Raise cost model.
- `Real−Paper` positive → paper fill price is worse than real (rare).
- `Paper−Expected` large negative → live sample below WF. Regime shift or overfit.
- `Real−Expected` >> benchmark → strategy is working better than history. Suspect luck.