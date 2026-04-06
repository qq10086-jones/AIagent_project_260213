# Factor Health Report: shadow_hybrid_ic

- Promotion recommendation: eligible_for_promotion
- Mean IC: 0.0512
- Mean t-stat: 1.0607
- Sharpe: 1.4928
- Max drawdown %: -9.58
- Avg turnover %: 0.00
- Production-eligible factors (80+ obs + PASS): sharpe_60, mom_consist
- QA eligible factor count: 3
- QA actionable mode count: 4
- QA latest zero exposure days: 0

## Family Summary

- fundamental: IC=0.0000, t=0.0000, coverage=0/10, positive_ic_ratio=0.00
- risk_adjusted: IC=0.0944, t=1.0900, coverage=4/4, positive_ic_ratio=1.00
- technical: IC=0.0225, t=1.0411, coverage=6/6, positive_ic_ratio=1.00

## Factor Eligibility

- accruals_inv | tier=fundamental_pending | family=fundamental | guard=None | obs=0 | eligible=False
- cfo_assets | tier=fundamental_pending | family=fundamental | guard=None | obs=0 | eligible=False
- dividend_yield | tier=candidate | family=fundamental | guard=None | obs=0 | eligible=False
- growth_op_yoy | tier=candidate | family=fundamental | guard=None | obs=0 | eligible=False
- growth_rev_yoy | tier=candidate | family=fundamental | guard=None | obs=0 | eligible=False
- guidance_delta | tier=candidate | family=fundamental | guard=None | obs=0 | eligible=False
- leverage_safety | tier=fundamental_pending | family=fundamental | guard=None | obs=0 | eligible=False
- margin_op | tier=fundamental_pending | family=fundamental | guard=None | obs=0 | eligible=False
- roa_op | tier=fundamental_pending | family=fundamental | guard=None | obs=0 | eligible=False
- value_bp | tier=candidate | family=fundamental | guard=None | obs=0 | eligible=False
- sharpe_20 | tier=candidate | family=risk_adjusted | guard=FAIL | obs=2 | eligible=False
- sharpe_60 | tier=core | family=risk_adjusted | guard=PASS | obs=170 | eligible=True
- sortino_60 | tier=candidate | family=risk_adjusted | guard=PASS | obs=2 | eligible=False
- vol_stability | tier=candidate | family=risk_adjusted | guard=FAIL | obs=2 | eligible=False
- ma_gap | tier=core | family=technical | guard=FAIL | obs=50 | eligible=False
- mom_consist | tier=core | family=technical | guard=PASS | obs=218 | eligible=True
- ret20 | tier=excluded | family=technical | guard=FAIL | obs=50 | eligible=False
- rsi14 | tier=excluded | family=technical | guard=FAIL | obs=50 | eligible=False
- slope60 | tier=excluded | family=technical | guard=FAIL | obs=50 | eligible=False
- vol_adj_mom20 | tier=excluded | family=technical | guard=FAIL | obs=50 | eligible=False
