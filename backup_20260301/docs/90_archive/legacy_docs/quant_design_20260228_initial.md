# Quant Design

## 1. Scope
Quant subsystem provides executable research/analysis tools for Nexus, including:
- Single-symbol analysis and execution suggestions
- Multi-candidate discovery with capital/risk constraints
- News-driven risk factor extraction
- Portfolio account/fill state management
- Market news intelligence (daily + active hot search)

Primary implementation:
- `worker-quant/worker.py`

## 2. Architecture
### 2.1 Runtime roles
- Orchestrator: intent routing, workflow decomposition, task dispatch, result fan-in
- Worker-Quant: tool execution, report/artifact generation, fact recording
- Brain: model-mediated synthesis and narrative assembly for quant workflows

### 2.2 Transport and storage
- Queue: Redis Streams (`stream:task`, `stream:result`)
- Metadata: Postgres (`runs`, `tasks`, `fact_items`, `event_log`, `assets`)
- Artifacts: MinIO (`nexus-artifacts`, `nexus-evidence`)
- Trading/account local state: SQLite (`/app/quant_trading/Project_optimized/japan_market.db`)

### 2.3 Reliability model
- At-least-once task consumption
- Retry with capped attempts (`MAX_RETRIES`)
- Dead-letter queue (`stream:dlq`) on terminal failure
- Event logging for claim/retry/success/failure

## 3. Tool Matrix
Current quant-related tools (registered in `TOOLS`):
- `quant.fetch_price`
- `quant.deep_analysis`
- `quant.discovery_workflow`
- `quant.compute_news_risk_factor`
- `quant.calc_limit_price`
- `quant.run_optimized_pipeline`
- `portfolio.set_account`
- `portfolio.record_fill`
- `news.daily_report`
- `news.active_hot_search`
- `news.preclose_brief_jp`
- `news.tdnet_close_flash`

## 4. Core Flows
### 4.1 Deep Analysis (`quant.deep_analysis`)
Input:
- `symbol`
- optional capital/risk fields (`capital_base_jpy`, `max_position_pct`, etc.)

Process:
- Quote + metrics + merged news
- Signal and sizing suggestion generation
- Markdown/HTML report rendering
- Artifact archive + fact record

Output:
- Structured analysis payload + report artifacts

### 4.2 Discovery Workflow (`quant.discovery_workflow`)
Input:
- `market` (`JP`/`US`/`ALL`), `goal`, `risk_profile`, capital/constraints

Process:
- Watchlist universe selection
- Candidate scoring (alpha + affordability + size penalties)
- Multi-attempt adaptive search (supports learning store)
- Position plan construction under budget/lot constraints

Output:
- Candidates, search attempts, selected strategy, position plan

### 4.3 Active Hot Search (`news.active_hot_search`)
Input:
- `lookback_hours` (default 24)
- `top_n` (bounded 5-10)
- `include_positions` (default true)

Process:
- Global multi-source hot news collection (GDELT + Google RSS multi-language)
- Freshness filtering + hotness scoring
- Holdings-linked news enrichment (if positions exist)
- Next-day structured advice generation (`next_day_advice`)

Output:
- `hot_news`, `position_news`, `positions`, `next_day_advice`, optional evidence artifacts

## 5. Market/Currency/Position Policy
### 5.1 Market locking
- If user explicitly sets `market`, discovery does not auto-expand market by default.
- Geo-impact JP routing sends `market=JP` and `auto_expand_market=false`.

### 5.2 Capital currency alignment
- Account state (`account_state.base_ccy`) can seed discovery capital context.
- Discovery normalizes to JPY for allocation math (`capital_base_jpy`).

### 5.3 Holdings-priority news
- When holdings exist, `position_news` is explicitly generated and prioritized in summary context.
- Holdings symbols are normalized (e.g., JP numeric ticker `9432` -> `9432.T`).

## 6. Evidence and Artifacts
### 6.1 Screenshot capture
- Browser screenshots can be captured via OpenClaw and archived.
- Near-blank screenshot filtering is applied before returning artifacts.

### 6.2 Report artifacts
- Markdown/HTML reports are archived to MinIO and referenced by object keys.

## 7. Config and Env
Important env/config knobs:
- `NEWS_FOCUS_MARKET`
- `DISCOVERY_LEARNING_PATH`
- `OPENCLAW_BASE_URL`
- `QWEN_MODEL`, `QUANT_LLM_MODEL`, provider API keys
- `configs/universe_us.json` and JP watchlist files under `quant_trading/Project_optimized`

## 8. Observability
- Task lifecycle in Postgres `tasks` + `event_log`
- Facts in `fact_items`
- Artifacts in `assets` (task-bound)
- Container logs:
  - `nexus-orchestrator`
  - `nexus-worker-quant`

## 9. Known Limits
- News quality depends on upstream feed freshness/availability.
- Some symbols may still have sparse quote/news coverage.
- Discovery outputs are screening-oriented and should be treated as decision support, not execution guarantee.

## 10. Current Baseline (2026-02-28)
- Active hot news supports global sourcing, holdings prioritization, and next-day structured advice.
- JP geo-impact workflows are market-locked by default to prevent unintended US expansion.
- Screenshot artifact pipeline includes blank-image suppression.
