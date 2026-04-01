# System Design (Latest)

## Scope
This document is the latest consolidated system-level design for OpenClaw Nexus.

## Runtime Architecture
- `orchestrator`: intent routing, workflow decomposition, guardrail fallback, task dispatch.
- `brain`: model-driven synthesis and planning bridge for quant workflows.
- `worker-quant`: executable tool runtime (quant/news/portfolio/web capture).
- `infra`: Redis (queue), Postgres (metadata/facts), MinIO (artifacts), OpenClaw adapter.

## Data and Control Flow
1. User input enters `orchestrator` (`Discord` or `/chat`).
2. Router decides `chat` vs `run`; rule fallback forces tool mode when needed.
3. Tasks enqueue to Redis Streams (`stream:task`).
4. Worker executes tool, writes `stream:result` and facts/events.
5. Orchestrator aggregates outputs and returns narrative/artifacts.

## Reliability Model
- At-least-once execution with retry and DLQ.
- Task lifecycle tracked in Postgres (`runs`, `tasks`, `event_log`).
- Artifact storage and attachment references via MinIO object keys.

## Current Baseline
- Geo-impact market queries can be hard-routed to tool workflows.
- Discovery can be market-locked (e.g. `JP` + `auto_expand_market=false`).
- News tool supports holdings-priority and artifact blank-image filtering.

## Related Docs
- Task protocol: `docs/01_design/system/task_queue_protocol.md`
- Quant design: `docs/01_design/quant/quant_design_latest.md`
- Learning design: `docs/01_design/learning/learning_design_latest.md`
- Web augmented chat design: `docs/01_design/web/web_augmented_chat_design_latest.md`
