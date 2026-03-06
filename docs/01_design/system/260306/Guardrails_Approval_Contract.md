# Guardrails & Approval Contract

## Scope

This contract defines the minimum acceptable slice for Workstream `WS-09 Guardrails + Approval`.

It answers the following:
- What defines low, medium, and high risk?
- When does a task transition require human approval?
- What tools are permitted for which roles?

## Components

### 1. Risk Classification

The system evaluates incoming Task Envelopes and proposed execution steps against a set of risk triggers.

**Schema:** [risk_classification.schema.json](../../../../orchestrator/contracts/guardrails/risk_classification.schema.json)

**Definitions:**
- `low`: Safe read-only or scoped operations (e.g., chat, local document reading). Auto-approved.
- `medium`: Scoped writes (e.g., writing new code files within a defined workspace). Auto-approved locally, might require review in CI.
- `high`: Potentially destructive or sensitive actions (e.g., file deletion, broad codebase rewrites, touching `.env` or system configurations). **Requires explicit Human Approval.**

### 2. Approval Checkpoint

When a `high` risk action is detected, the Orchestrator must yield a `DispatchSuccessResponse` with `response_mode: approval_request` instead of executing the tool immediately. The execution context halts until explicit consent is given.

### 3. Tool Permission Boundaries

Defines a strict mapping between `role_name` and `tool_name`. Agents cannot invoke tools outside their defined purview.

**Schema:** [tool_permission.schema.json](../../../../orchestrator/contracts/guardrails/tool_permission.schema.json)

## Non-Scope
- No Web UI built for approvals (CLI or Discord bot text replies only).
- No complex multi-layer approval chains (simple Yes/No boolean is sufficient).
