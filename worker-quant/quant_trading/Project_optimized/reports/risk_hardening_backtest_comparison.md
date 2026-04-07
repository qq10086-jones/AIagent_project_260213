# 风控加固回测对比报告
**日期**: 2026-04-07

## 参数对比

| 参数 | Baseline (旧) | Variant (Sprint加固) |
|------|---------------|---------------------|
| stop_loss_vol_mult | 6.0 | 3.0 |
| stop_loss_min_pct | 6% | 4% |
| stop_loss_max_pct | 20% | 12% |
| trailing_activate | OFF | 3% |
| trailing_stop | - | 2% |
| max_dd_half | 12% | 8% |
| max_dd_full | 18% | 12% |
| max_position_pct | 50% | 35% |

## 回测结果

| 指标 | Baseline | Variant | 变化 |
|------|----------|---------|------|
| Final Equity | ¥635,379 | ¥602,430 | -32,948 |
| Total Return | 58.84% | 50.61% | -8.24pp |
| Sharpe | 0.921 | 1.177 | +0.256 |
| Sortino | 1.471 | 2.117 | +0.646 |
| Max Drawdown | -13.52% | -7.50% | +6.02pp |
| Annual Vol | 14.02% | 9.37% | -4.66pp |
| Win Rate | 0.0% | 0.0% | +0.0pp |
| Profit Factor | 0.00 | 0.00 | +0.00 |
| StopLoss Triggers | 10 | 301 | +291 |
| Avg Turnover | ¥9,365 | ¥37,486 | - |
| Avg Cost | ¥0 | ¥0 | - |

## 分析

**结论**: Variant 在风险调整收益(Sharpe)和最大回撤两方面均优于 Baseline。风控加固有效。

- 止损触发次数变化: 10 → 301（更严格的止损线导致更频繁触发）
- Trailing stop 新增，预期减少盈利回吐

---
*自动生成 by run_risk_backtest_comparison.py*