# Observability Contract

## Scope

This contract defines the minimum slice for Workstream `WS-10 Observability + UI` permitted under the strict "No UI" governance policy.

Current phase scope:
- `Task 10-03`: Discord progress notifications
- `Task 10-04`: Failure reporting

## Components

### 1. Progress Notifications (`Task 10-03`)
Provides a structured text string summarizing major workflow transitions, suitable for a `progress_update` dispatch response.

- **Trigger:** Workflow step completes successfully and moves to next step.
- **Output:** Human-readable transition message (e.g., "✅ PM Spec completed. ⏳ Starting Architecture Design...").

### 2. Failure Reporting (`Task 10-04`)
Provides a formatted text block when an execution layer throws an error, separating user-friendly context from developer logs.

- **Trigger:** Engine or Tool failure resulting in step `error_code`.
- **Output:** Human-readable failure explanation + code block with technical trace.

## Definition of Done (DoD)
- Must not rely on free-text generation models at runtime (must be deterministic template strings).
- Must have JSON schemas for input payloads to ensure the reporter does not crash on malformed data.
- Must not expose internal system secrets in error logs.

## Non-Scope
- **Blocked:** `Task 10-01 Task dashboard` (UI banned).
- **Blocked:** `Task 10-02 Workflow timeline UI` (UI banned, mitigated by `WS-08-04` CLI replay).
