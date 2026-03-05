# Discord Command Cheatsheet (2026-03-05)

## 1) Coding Team
```text
/coder: Build a minimal CRM web app with customer list, detail page, and add/edit form. Keep changes reviewable and output test/risk/verification artifacts.
/coder: Build a small browser game with score tracking and restart flow. Keep implementation reviewable and output required workflow artifacts.
/coder: @gpt-5.3 Build a minimal CRM admin dashboard with customer CRUD and basic validation.
/coder: @minimax Build a simple browser game with score and restart.
```

## 2) Approvals
```text
/approve: <task_id>
/reject: <task_id>
```

## 3) Quant Stable Commands
```text
/run portfolio.set_account {"starting_capital":400000,"ccy":"JPY"}
/run news.active_hot_search {"lookback_hours":24,"top_n":8,"include_positions":true}
/run quant.discovery_workflow {"market":"JP","goal":"Find 2 candidates for next-day staged entry under 400k JPY","risk_profile":"medium","capital_base_jpy":400000,"quick_mode":true,"time_budget_s":75,"max_attempts":2,"min_candidates":2,"auto_expand_market":false}
/run quant.deep_analysis {"symbol":"7203.T","capital_base_jpy":400000,"max_position_pct":0.25}
/run news.tdnet_close_flash {"date":"2026-03-05","freshness_hours":24}
```

## 4) Generic Tool Format
```text
/run <tool_name> [json_payload]
```

## 5) Validation (Local Terminal)
```bash
npm.cmd --prefix orchestrator run canary:coding_team -- --n 1 --strict true --input crm_mini.json --timeout-sec 900
npm.cmd --prefix orchestrator run canary:coding_team -- --n 1 --strict true --input game_mini.json --timeout-sec 900
npm.cmd --prefix orchestrator run canary:coding_team -- --n 5 --strict true --input crm_mini.json --timeout-sec 900
npm.cmd --prefix orchestrator run canary:coding_team -- --n 5 --strict true --input game_mini.json --timeout-sec 900
```
