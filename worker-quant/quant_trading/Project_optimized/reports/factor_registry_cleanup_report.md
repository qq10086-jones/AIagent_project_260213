# Factor Registry Cleanup Candidates: shadow_hybrid_ic

- Eligible count: 2
- Exclude count: 18

- accruals_inv | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- cfo_assets | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- dividend_yield | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- growth_op_yoy | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- growth_rev_yoy | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- guidance_delta | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- leverage_safety | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- margin_op | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- roa_op | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- value_bp | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=None | obs=0
- sharpe_20 | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=2
- sharpe_60 | action=keep | reason=eligible | guard=PASS | obs=170
- sortino_60 | action=exclude_from_production | reason=insufficient_observations | guard=PASS | obs=2
- vol_stability | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=2
- ma_gap | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=50
- mom_consist | action=keep | reason=eligible | guard=PASS | obs=218
- ret20 | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=50
- rsi14 | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=50
- slope60 | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=50
- vol_adj_mom20 | action=exclude_from_production | reason=insufficient_observations_and_guard_fail | guard=FAIL | obs=50
