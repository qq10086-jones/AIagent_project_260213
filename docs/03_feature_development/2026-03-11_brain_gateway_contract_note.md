# Brain Gateway Contract Note

- Date: 2026-03-11
- Scope: current typed boundary between `brain` and `orchestrator`

---

## 1. Supported Endpoints

### `GET /brain/facts/latest`

Purpose:

- latest fact lookup for a given `run_id` and `agent_name`

Query contract:

- `run_id` required
- `agent_name` required
- `tool_name` optional

Responses:

- `200` fact found
- `404` fact not found
- `400` missing required query fields
- `500` internal query failure

Schema files:

- `orchestrator/contracts/brain_gateway_latest_fact_request.schema.json`
- `orchestrator/contracts/brain_gateway_latest_fact_response.schema.json`

### `POST /brain/routing-decisions`

Purpose:

- ingest routing decision events from `brain` into orchestrator event history

Request contract:

- `run_id` or `workflow_run_id` required
- `event_name` optional, defaults to `brain.routing.decision`
- `payload` optional object

Responses:

- `200` event ingested
- `400` missing identity fields
- `500` ingest failure

Schema files:

- `orchestrator/contracts/brain_gateway_routing_decision_ingest_request.schema.json`
- `orchestrator/contracts/brain_gateway_routing_decision_ingest_response.schema.json`

---

## 2. Ownership

- `brain` owns request initiation and caller-side retry
- `orchestrator` owns persistence, input validation, and response semantics
- direct PostgreSQL reads from `brain` are not part of the supported contract

---

## 3. Current Limitation

The gateway surface is intentionally narrow.

It does not yet authorize:

- arbitrary fact queries
- broad event ingestion categories
- cross-service schema-free payloads
