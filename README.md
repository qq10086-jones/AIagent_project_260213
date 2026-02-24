# OpenClaw Nexus Daily Reports (Project Snapshot)

This repo contains an OpenClaw-based control plane with daily report pipelines for:
- Market news (multi-language sources with CN summary)
- GitHub agent skills discovery

The system uses OpenClaw as the AI gateway and orchestrator, with worker-quant executing report generation and MinIO storing artifacts.

## What’s Included
- `news.daily_report`: Daily market news aggregation (multi-language) → HTML + PNG
- `github.skills_daily_report`: Daily GitHub agent skills summary → HTML + PNG
- OpenClaw browser evidence capture (optional screenshots)
- Streamlit UI for quick launches and artifact browsing

## Architecture (High Level)
- **OpenClaw Gateway**: AI runtime + browser control
- **Orchestrator**: workflow dispatch + task tracking
- **Worker (worker-quant)**: report generation + MinIO artifact archive
- **MinIO**: artifact storage
- **Postgres**: task + asset metadata

## Key Paths
- Worker logic: `worker-quant/worker.py`
- Dependencies: `worker-quant/requirements.txt`
- Tool registry: `configs/tools.json`
- US stock universe: `configs/universe_us.json`
- Compose stack: `infra/docker-compose.yml`
- UI: `ui/app.py`

## Quick Start (Docker)
```bash
docker compose -f infra/docker-compose.yml up -d
```

Open:
- UI: `http://localhost:8501`
- Orchestrator health: `http://localhost:3000/health`
- MinIO: `http://localhost:9001`

## Trigger Reports (Manual)
Use the UI “Quick Launch” buttons or POST to the orchestrator:
```bash
curl -X POST http://localhost:3000/workflows \
  -H "Content-Type: application/json" \
  -d '{"name":"Daily Market News","definition":{"steps":[{"tool_name":"news.daily_report","payload":{},"risk_level":"low"}]}}'

curl -X POST http://localhost:3000/workflows \
  -H "Content-Type: application/json" \
  -d '{"name":"Daily GitHub Skills","definition":{"steps":[{"tool_name":"github.skills_daily_report","payload":{},"risk_level":"low"}]}}'
```

## Notes
- GDELT can lag by 1–2 days; the pipeline automatically widens the query window.
- Optional: set `GITHUB_TOKEN` for higher GitHub API rate limits.
- Optional: `NEWS_USE_LLM=1` / `GITHUB_USE_LLM=1` to enable CN summaries via local LLM.

