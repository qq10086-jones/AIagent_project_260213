# Factor Health Report: shadow_hybrid_ic

- Promotion recommendation: eligible_for_promotion
- Mean IC: 0.0513
- Mean t-stat: 1.1658
- Sharpe: 1.4928
- Max drawdown %: -9.58
- Avg turnover %: 6.45
- Production-eligible factors (80+ obs + PASS): sharpe_60, ma_gap, mom_consist
- QA eligible factor count: 3
- QA actionable mode count: 4
- QA latest zero exposure days: 0

## Family Summary

- fundamental: IC=0.0000, t=0.0000, coverage=0/10, positive_ic_ratio=0.00
- risk_adjusted: IC=0.0943, t=1.0468, coverage=4/4, positive_ic_ratio=1.00
- technical: IC=0.0226, t=1.2451, coverage=6/6, positive_ic_ratio=1.00

## Factor Eligibility

- accruals_inv | family=fundamental | guard=None | obs=0 | eligible=False
- cfo_assets | family=fundamental | guard=None | obs=0 | eligible=False
- dividend_yield | family=fundamental | guard=None | obs=0 | eligible=False
- growth_op_yoy | family=fundamental | guard=None | obs=0 | eligible=False
- growth_rev_yoy | family=fundamental | guard=None | obs=0 | eligible=False
- guidance_delta | family=fundamental | guard=None | obs=0 | eligible=False
- leverage_safety | family=fundamental | guard=None | obs=0 | eligible=False
- margin_op | family=fundamental | guard=None | obs=0 | eligible=False
- roa_op | family=fundamental | guard=None | obs=0 | eligible=False
- value_bp | family=fundamental | guard=None | obs=0 | eligible=False
- sharpe_20 | family=risk_adjusted | guard=FAIL | obs=2 | eligible=False
- sharpe_60 | family=risk_adjusted | guard=PASS | obs=225 | eligible=True
- sortino_60 | family=risk_adjusted | guard=FAIL | obs=2 | eligible=False
- vol_stability | family=risk_adjusted | guard=FAIL | obs=2 | eligible=False
- ma_gap | family=technical | guard=PASS | obs=105 | eligible=True
- mom_consist | family=technical | guard=PASS | obs=273 | eligible=True
- ret20 | family=technical | guard=FAIL | obs=50 | eligible=False
- rsi14 | family=technical | guard=FAIL | obs=50 | eligible=False
- slope60 | family=technical | guard=FAIL | obs=50 | eligible=False
- vol_adj_mom20 | family=technical | guard=FAIL | obs=50 | eligible=False
