# Command Execution Runbook (Coding Team + Quant)

Date: `2026-03-05`

## 1. Quick Start (Discord)

### 1.1 Generic tool execution
```text
/run <tool_name> [json_payload]
```

Example:
```text
/run news.tdnet_close_flash {"date":"2026-03-05"}
```

### 1.2 Coding task delegation
```text
/coder: <your coding task>
```

Model override:
```text
/coder: @gpt-5.3 Build a minimal CRM dashboard with customer list/detail/form.
/coder: @minimax Build a simple browser game with score + restart.
```

### 1.3 Approval controls
```text
/approve: <task_id>
/reject: <task_id>
```

---

## 2. Coding Team Manual

### 2.1 One-command CRM (Discord)
```text
/coder: Build a minimal CRM web app with customer list, detail page, and add/edit form. Keep changes reviewable and output test/risk/verification artifacts.
```

### 2.2 One-command Game (Discord)
```text
/coder: Build a small browser game with score tracking and restart flow. Keep implementation reviewable and output required workflow artifacts.
```

### 2.3 Deterministic canary (local, recommended)
Run from project root:
```bash
npm.cmd --prefix orchestrator run canary:coding_team -- --n 1 --strict true --input crm_mini.json --timeout-sec 900
npm.cmd --prefix orchestrator run canary:coding_team -- --n 1 --strict true --input game_mini.json --timeout-sec 900
```

Batch stability:
```bash
npm.cmd --prefix orchestrator run canary:coding_team -- --n 5 --strict true --input crm_mini.json --timeout-sec 900
npm.cmd --prefix orchestrator run canary:coding_team -- --n 5 --strict true --input game_mini.json --timeout-sec 900
```

### 2.4 Go/No-Go verification
Run from `orchestrator/`:
```bash
npm run validate:go-nogo -- --workflow-run-id <workflow_run_id>
```

---

## 3. Quant Manual (Stable Mode)

## 3.1 Principle
For quant, prefer `/run + structured JSON` over pure free-form chat.  
Reason: this avoids intent-route misses and reduces unstable narrative output.

### 3.2 Price check
```text
/run quant.fetch_price {"symbol":"7203.T"}
```

### 3.3 Deep analysis (single symbol)
```text
/run quant.deep_analysis {"symbol":"7203.T","capital_base_jpy":400000,"max_position_pct":0.25}
```

### 3.4 Discovery workflow (recommended baseline)
```text
/run quant.discovery_workflow {"market":"JP","goal":"next-day operation plan for no-position account","risk_profile":"medium","capital_base_jpy":400000,"quick_mode":true,"time_budget_s":75,"max_attempts":2,"min_candidates":2,"auto_expand_market":false}
```

### 3.5 Active hot news
```text
/run news.active_hot_search {"lookback_hours":24,"top_n":8,"include_positions":true}
```

### 3.6 TDNet close flash (JP)
```text
/run news.tdnet_close_flash {"date":"2026-03-05","freshness_hours":24}
```

### 3.7 Account and fill state
```text
/run portfolio.set_account {"starting_capital":400000,"ccy":"JPY"}
/run portfolio.record_fill {"symbol":"7203.T","side":"BUY","qty":100,"price":2800}
```

---

## 4. Quant Quality Playbook (fix "analysis is messy")

### 4.1 Use this execution sequence
1. `portfolio.set_account` (if account context is missing)
2. `news.active_hot_search` (market context)
3. `quant.discovery_workflow` (candidate + plan)
4. `quant.deep_analysis` (single-symbol drilldown for shortlisted names)

### 4.2 Payload rules that improve consistency
- Always set `market` (`JP`/`US`/`ALL`).
- Always set `goal` with one clear objective sentence.
- For fast decision support: keep `quick_mode=true`, `time_budget_s<=90`, `max_attempts<=2`.
- For broader search: set `quick_mode=false`, `time_budget_s=150~300`, `max_attempts=3~4`.

### 4.3 Prompt anti-chaos template (Discord)
```text
/run quant.discovery_workflow {"market":"JP","goal":"Find 2 candidates for next-day staged entry under 400k JPY with medium risk","risk_profile":"medium","capital_base_jpy":400000,"quick_mode":true,"time_budget_s":75,"max_attempts":2,"min_candidates":2,"auto_expand_market":false}
```

---

## 5. Ops and Troubleshooting

### 5.1 Check pending approvals (Web UI)
- Open:
  - `http://localhost:3000/ui/approvals`

### 5.2 Query run status (API)
```text
GET /runs/:run_id/status
GET /runs/:run_id/timeline
GET /runs/:run_id/artifacts
GET /workflow-runs/:workflow_run_id
GET /workflow-runs/:workflow_run_id/validate-pack
```

### 5.3 When a test appears "stuck"
1. Check if it is batch mode (`--n 5`/`--n 20`) instead of single run.
2. Check `workflow_run`/`tasks` for `running` rows.
3. If interrupted manually, perform cleanup policy:
   - mark residual running rows as `MANUAL_ABORT`
   - restart `worker-coder`
   - re-run with `n=1` first.

### 5.4 Common input errors
- Invalid JSON payload in `/run` -> fix JSON syntax first.
- Unknown tool name -> check `configs/tools.json`.
- Waiting approval -> use `/approve:` or `/reject:`.

---

## 6. Current Recommended Baselines
- Coding Team smoke gate:
  - CRM `n=5 strict` pass
  - Game `n=5 strict` pass
- Quant execution:
  - prefer `/run` structured payloads for stability and clearer outputs.
