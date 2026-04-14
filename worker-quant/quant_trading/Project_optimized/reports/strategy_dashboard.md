# Strategy Dashboard — 2026-04-15

## Capital allocation + gate status

| Strategy | Tier | State | Cap | G1 Entry | G2 Retain | G3 Kill | G4 Promo | Recommend |
|---|---|---|---|---|---|---|---|---|
| sprint | real | paused_sunk_only | JPY250,000 | FAIL | - | OK | - | paused_sunk_only |
| high52w | real | active | JPY100,000 | OK | - | OK | - | active |
| amihud | real | active | JPY100,000 | OK | - | OK | - | active |
| sprint_paper | paper | active | - | - | - | - | - | active |
| high52w_paper | paper | active | - | - | - | - | - | active |
| amihud_paper | paper | active | - | - | - | - | - | active |
| amihud_k5_paper | paper | active | - | - | - | - | - | active |
| amihud_k30_paper | paper | active | - | - | - | - | - | active |
| amihud_adv50_paper | paper | active | - | - | - | - | - | active |
| min_ret_paper | paper | active | - | - | - | - | - | active |
| mom_high52w_paper | paper | active | - | - | - | - | - | active |

## Performance: real vs paper vs walk-forward expected

| Strategy | Months | Real PnL | Paper PnL | WF-Expected | Exec Gap | Exp Gap |
|---|---|---|---|---|---|---|
| sprint | 1.51 | +0.01% | +22.34% | +1.72% | -22.33% | -1.71% |
| high52w | 0.0 | - | +0.00% | - | - | - |
| amihud | 0.0 | +0.00% | +0.00% | - | +0.00% | - |
| sprint_paper | 0.13 | +22.34% | - | - | - | - |
| high52w_paper | 0.0 | +0.00% | - | - | - | - |
| amihud_paper | 0.0 | +0.00% | - | - | - | - |
| amihud_k5_paper | 0.0 | +0.00% | - | - | - | - |
| amihud_k30_paper | 0.0 | +0.00% | - | - | - | - |
| amihud_adv50_paper | 0.0 | - | - | - | - | - |
| min_ret_paper | 0.0 | +0.00% | - | - | - | - |
| mom_high52w_paper | 0.0 | +0.00% | - | - | - | - |

## Live state

### sprint (real)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 3041.T | 400 | JPY585.00 |

### amihud (real)

Open orders: **3**

### sprint_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 3041.T | 200 | JPY588.65 |
| 7984.T | 100 | JPY899.55 |

### high52w_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2243.T | 100 | JPY3,426.71 |

### amihud_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2090.T | 100 | JPY4,492.24 |
| 5341.T | 100 | JPY279.14 |
| 9980.T | 100 | JPY103.05 |

### amihud_k5_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2090.T | 100 | JPY4,492.24 |
| 5341.T | 100 | JPY279.14 |

### amihud_k30_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2090.T | 100 | JPY4,492.24 |
| 5341.T | 100 | JPY279.14 |
| 9980.T | 100 | JPY103.05 |

### amihud_adv50_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2323.T | 100 | JPY364.18 |
| 2370.T | 800 | JPY29.01 |
| 2648.T | 100 | JPY3,468.73 |
| 2857.T | 100 | JPY631.40 |
| 7602.T | 100 | JPY203.00 |

### min_ret_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2012.T | 100 | JPY239.22 |
| 2344.T | 100 | JPY1,487.74 |
| 2620.T | 100 | JPY369.38 |
| 2857.T | 100 | JPY631.40 |
| 2882.T | 100 | JPY1,978.99 |

### mom_high52w_paper (paper)
| Symbol | Qty | Avg Cost |
|---|---|---|
| 2294.T | 100 | JPY3,041.52 |
| 2874.T | 100 | JPY1,645.82 |


## Gate reasons (where gates fired)

**sprint**
- G1 FAIL: evidence.monthly_excess_vs_ew=-0.0013 <= 0
