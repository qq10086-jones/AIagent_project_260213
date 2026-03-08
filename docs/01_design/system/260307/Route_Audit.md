# Route Audit
## Date: 2026-03-07
## Status: WS-14-01 — Complete
## Purpose: Classify all HTTP routes in orchestrator/src/index.js as canonical or deprecated

---

## Canonical Routes (North Star path)

These routes are authoritative. They must not be removed.

| Method | Path | Line | Purpose |
|--------|------|------|---------|
| GET | `/health` | 2141 | Health check — required by all live validators |
| POST | `/vnext/route` | 2142 | Route preview / contract inspection |
| POST | `/vnext/dispatch` | 2168 | Direct vNext dispatch (test/debug entry) |
| GET | `/runtime/config` | 2201 | Runtime configuration inspection |
| POST | `/tasks/:task_id/approve` | 2270 | Approval entrypoint (WS-09) |
| POST | `/tasks/:task_id/reject` | 2272 | Rejection entrypoint (WS-09) |
| POST | `/workflow-runs/start` | 2307 | Start a workflow run (primary API entry) |
| GET | `/workflow-runs/:workflow_run_id` | 2338 | Workflow status query |
| POST | `/workflow-runs/:workflow_run_id/resume-token` | 2349 | Issue resume token |
| POST | `/workflow-runs/:workflow_run_id/resume` | 2361 | Resume workflow from token |
| GET | `/workflow-runs/:workflow_run_id/validate-pack` | 2375 | Validate artifact pack |
| POST | `/workflow-runs/:workflow_run_id/archive-pack` | 2391 | Archive artifact pack to MinIO |
| GET | `/runs/:run_id/status` | 2407 | Run status query |
| GET | `/runs/:run_id/timeline` | 2442 | Execution timeline (WS-08) |
| GET | `/runs/:run_id/artifacts` | 2466 | Artifact listing (WS-08) |
| GET | `/approvals/pending` | 2486 | Pending approval queue |
| POST | `/chat` | 2640 | Primary vNext chat + coding dispatch entry |
| POST | `/traces` | 2644 | Trace recording |
| POST | `/traces/:trace_id/feedback` | 2659 | Trace feedback |

**Total canonical: 19**

---

## Deprecated Routes (to be removed in WS-14-03)

These routes are not in the North Star path. They have replacements and must be removed after deprecation headers are confirmed.

| Method | Path | Line | Reason | Replacement |
|--------|------|------|--------|-------------|
| POST | `/execute-tool` | 2242 | Pre-vNext ingress — handled by `/chat` + `runtime_dispatch` since vNext wiring | `POST /chat` |
| POST | `/debug/plan` | 2227 | Debug-only planning endpoint — not in North Star path, no contract | None (debug only, remove) |
| POST | `/workflows` | 2274 | Old workflow creation endpoint — replaced by `/workflow-runs/start` | `POST /workflow-runs/start` |
| GET | `/ui/approvals` | 2503 | HTML approval page — UI banned in current governance phase, no equivalent needed | `GET /approvals/pending` (JSON) |

**Total deprecated: 4**

---

## Uncertain Routes (need review)

None. All routes have been classified.

---

## Summary

| Category | Count |
|----------|-------|
| Canonical | 19 |
| Deprecated | 4 |
| Uncertain | 0 |
| **Total** | **23** |

---

## Deprecation Action Plan (WS-14-02 → WS-14-03)

### Step 1 — WS-14-02: Add headers (safe, immediate)
For each deprecated route, add before the response:
```js
res.set("X-Deprecated", "true");
console.warn(`[deprecated] ${req.method} ${req.path} called — use replacement`);
```

### Step 2 — Verify zero usage in integration tests
Run: `npm --prefix orchestrator test`
Confirm no test calls `/execute-tool`, `/debug/plan`, `/workflows` (POST), or `/ui/approvals`.

### Step 3 — WS-14-03: Remove routes
Delete the 4 deprecated route handlers from `index.js`.
Estimated line reduction: ~120–150 lines (contributes toward index.js budget goal of ≤800 lines).

### Step 4 — Update runbooks
If any runbook or canary references deprecated routes, update them.

---

## Notes

- `/vnext/route` and `/vnext/dispatch` are kept as canonical because they are used by the live validators and serve as contract inspection tools that support the North Star path.
- `/runtime/config` is kept as canonical for operational visibility.
- The `/traces` routes are kept as they support audit trail requirements (NFR-02 Traceability).
