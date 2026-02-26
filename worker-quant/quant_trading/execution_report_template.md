# Execution Report — Rebalance Run

- **Run ID**: {{run_id}}
- **As-of Date**: {{asof_date}}  (signals built on/through this date)
- **Execution Date**: {{exec_date}} (rebalance action date)
- **Universe**: {{universe_desc}}  (e.g., TOPIX subset / custom list)
- **Benchmark**: {{benchmark_ticker}}  (e.g., 1321.T)
- **Horizon (H)**: {{horizon_days}} trading days
- **Rebalance Every**: {{rebalance_every}} trading days
- **Train Window**: {{train_window}} trading days
- **Currency**: {{ccy}}  (JPY / USD)

---

## 1) 核心调仓指令 (Executive Rebalance Orders)

### 1.1 全局风控状态 (Global Risk Regime)
- **Risk State**: {{risk_state_badge}}  (🟢 Risk-ON / 🔴 Risk-OFF)
- **Risk Trigger**: {{risk_trigger_desc}}
- **Action if Risk-OFF**: **Target weights forced to 0% (100% cash)**

### 1.2 执行前检查 (Pre-Trade Checklist)
| Item | Value |
|---|---:|
| Portfolio Value (pre) | {{pv_pre}} |
| Cash (pre) | {{cash_pre}} |
| Turnover Notional (est.) | {{turnover_notional}} |
| Estimated Total Costs | {{cost_total_est}} |
| Estimated Cash After Trades | {{cash_post_est}} |
| Lot Size Rule | {{lot_size_rule}} |
| Rounding / Residual Handling | {{rounding_policy}} |

> **Notes**: {{pretrade_notes}}

### 1.3 本期目标权重 (Target Weights)
> Source: `target_weights.csv` (or equivalent)

| Ticker | Name | Current Weight | Target Weight | Delta |
|---|---|---:|---:|---:|
{{target_weights_table}}

**Target Weights Summary**
- #Positions (target > 0): {{n_pos_target}}
- Concentration (Top 3 weights): {{top3_concentration}}
- Cash Target (implicit): {{cash_target_implied}}

### 1.4 具体交易清单 (Trade List)
> Generated from `execute_rebalance()` + `lot_size` rounding.

| Ticker | Side | Qty (shares) | Qty (lots) | Est. Price | Notional | Reason |
|---|---|---:|---:|---:|---:|---|
{{trade_list_table}}

**Execution Notes**
- Orders are **long-only** (no short).
- Trades respect: lot size, cash constraint, max ADV fraction, and impact model (see Section 3).

---

## 2) 组合特征与模型视角 (Portfolio Analytics)

### 2.1 关键因子读数 (Key Factor Readings)
> From `make_features()` at as-of date.

| Ticker | Weight | slope60 | rsi14 | vol20 | z20 | ma_gap | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
{{feature_readings_table}}

**Interpretation Rules (fixed, auditable)**
- slope60: higher = stronger medium-term trend
- rsi14: {{rsi_rule_text}}
- z20: {{z20_rule_text}}
- vol20: {{vol_rule_text}}

### 2.2 模型预测 (Alpha Score — PanelRidge)
> Predicts **risk-adjusted return** over next **H={{horizon_days}}** days.

| Ticker | Alpha Score | Rank | Selected? | Comment |
|---|---:|---:|---|---|
{{alpha_score_table}}

**Model Snapshot**
- Model: {{model_name}} (e.g., PanelRidge)
- Fit method: {{fit_method}}
- Regularization: {{reg_desc}}
- Feature set: {{feature_set_desc}}

### 2.3 本期贡献归因 (Top Contributors / Detractors)
> Contribution ≈ weight × realized (or next-period) return.

**Top Positive Contributors**
| Rank | Ticker | Avg Weight | Return | Contribution |
|---:|---|---:|---:|---:|
{{top_contrib_table}}

**Top Negative Contributors**
| Rank | Ticker | Avg Weight | Return | Contribution |
|---:|---|---:|---:|---:|
{{bottom_contrib_table}}

---

## 3) 风险与流动性评估 (Risk & Execution Constraints)

### 3.1 流动性警告 (Liquidity Checks)
> Constraint: trade_notional <= max_adv_frac × ADV × price (approx).

| Ticker | Trade Notional | ADV (shares/day) | ADV Notional | max_adv_frac | Utilization | Flag |
|---|---:|---:|---:|---:|---:|---|
{{adv_check_table}}

- **Liquidity Flags**: {{liquidity_flag_summary}}
- **If flagged**: {{liquidity_mitigation_policy}} (e.g., split orders / cap trade / skip name)

### 3.2 预估冲击成本与滑点 (Impact / Slippage Estimate)
> Impact model: impact_bps ≈ impact_k × (trade_notional / ADV_notional)^{{impact_power}}

| Cost Component | Estimate (bps) | Estimate ({{ccy}}) | Notes |
|---|---:|---:|---|
| Fees | {{fee_bps}} | {{fee_amt}} | {{fee_notes}} |
| Slippage | {{slip_bps}} | {{slip_amt}} | {{slip_notes}} |
| Market Impact | {{impact_bps}} | {{impact_amt}} | k={{impact_k}}, power={{impact_power}} |
| **Total** | **{{total_cost_bps}}** | **{{total_cost_amt}}** | |

### 3.3 风险摘要 (Risk Summary)
| Metric | Value |
|---|---:|
| Gross Exposure | {{gross_exposure}} |
| Net Exposure | {{net_exposure}} |
| #Holdings | {{n_holdings}} |
| Largest Position | {{largest_pos_desc}} |
| Est. Vol (portfolio) | {{port_vol_est}} |
| Est. Max DD (model-based, if available) | {{model_dd_est}} |

> **Risk Notes**: {{risk_notes}}

---

## 4) 回测性能背书 (Backtest Performance Track Record)

### 4.1 净值曲线与资金变化 (Equity & Capital)
- Initial Capital: {{initial_capital}}
- Final Equity: {{final_equity}}
- Period: {{bt_start}} → {{bt_end}}
- #Rebalances: {{n_rebalances}}

| Item | Strategy | Benchmark |
|---|---:|---:|
| Total Return | {{ret_total}} | {{ret_bench}} |
| CAGR (if applicable) | {{cagr}} | {{cagr_bench}} |
| Volatility | {{vol}} | {{vol_bench}} |
| Sharpe | {{sharpe}} | {{sharpe_bench}} |
| Max Drawdown | {{max_dd}} | {{max_dd_bench}} |

### 4.2 成本与换手统计 (Costs & Turnover)
| Metric | Value |
|---|---:|
| Avg Turnover (notional) | {{turnover_avg}} |
| Median Turnover (notional) | {{turnover_median}} |
| Total Costs Paid | {{cost_paid_total}} |
| Avg Cost (bps) | {{cost_bps_avg}} |
| % Days Invested (Risk-ON) | {{pct_risk_on}} |

### 4.3 数据与稳定性声明 (Data & Stability)
- Data Source(s): {{data_sources}}
- Survivorship / Corporate actions handling: {{corp_action_policy}}
- Leakage control: {{leakage_control_desc}}
- Missing data handling: {{missing_data_policy}}
- Known limitations: {{known_limits}}

---

## Appendix

### A) Full Target Weights (Raw)
{{target_weights_raw_block}}

### B) Full Trade Blotter (Raw)
{{trade_blotter_raw_block}}

### C) Config Snapshot
```json
{{config_json}}
```

### D) Notes / Changelog
- {{changelog_item_1}}
- {{changelog_item_2}}
