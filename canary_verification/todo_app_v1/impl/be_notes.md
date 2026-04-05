# Implementation Notes - Backend

## Decisions
- Using Express for simplicity.
- Using in-memory store for prototype.

## Task Status (SP-03 Verified)
- [COMPLETED] T-BE-1: Initialize Express and CORS | Rationale: Basic server scaffolding is ready.
- [COMPLETED] T-BE-2: Implement GET /api/todos | Rationale: Verified with curl.
- [SKIPPED] T-BE-3: Implement POST /api/todos | Rationale: Pending DB persistence layer (moved to v2).

## Run Instructions
`cd impl/be_changes && npm install && node server.js`
