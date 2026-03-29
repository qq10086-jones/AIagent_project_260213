# Nexus Beta Quality Memory

Date: 2026-03-29
Version: v1.1

## Facts

- Coding workflow order is:
  `pm_spec -> arch_design -> impl_be -> impl_fe -> smoke_test -> qa_verify -> release_pack -> deploy_preview`
- Final runnable backend root is `impl/be_changes/`.
- Frontend produces files under `impl/fe_changes/public/`.
- Before smoke or release run instructions are generated, frontend static files must be assembled into `impl/be_changes/public/`.

## Source Of Truth

- `release_pack` must inspect actual deliverables first:
  - `impl/be_changes/package.json`
  - `impl/be_changes/server.js`
  - `smoke/smoke_result.json` when present
- `impl/be_notes.md` is supporting context only.
- Never infer tech stack from the goal description alone.

## Backend Contract

- `impl_be` must always produce:
  - `impl/be_changes/server.js`
  - `impl/be_changes/package.json`
  - `impl/be_notes.md`
  - `handoff/be_to_fe.json`
- `server.js` must respect `process.env.PORT`.
- `server.js` must serve static files from `impl/be_changes/public/`.
- `GET /` must return `public/index.html`.

## Frontend Contract

- `impl_fe` must always produce:
  - `impl/fe_changes/public/index.html`
  - `impl/fe_changes/public/app.js`
  - `impl/fe_notes.md`
- Same-origin relative API paths only.
- Do not hardcode `localhost` in FE runtime calls.

## Smoke Test

- `smoke_test` writes `smoke/smoke_result.json` in all cases.
- Test port is fixed to `13099`.
- L1 means service boots and `GET /` returns HTML or HTTP 200.
- L2 means a primary API endpoint, when identifiable, returns a captured HTTP result.
- L1 failure is workflow-fatal.
- L2 failure can be recorded as partial evidence if L1 still passes.

## QA

- `qa_verify` must cite `smoke/smoke_result.json` when present.
- QA output file is `verify/qa_report.json`.
- QA should distinguish deterministic evidence from semantic checks.

## Release

- `release_pack` must always produce:
  - `release/release_notes.md`
  - `release/artifact_manifest.json`
  - `release/README.md`
  - `release/start.sh`
- `release/README.md` must include real commands for:
  - entering `impl/be_changes`
  - `npm install`
  - `node server.js`
  - preview/local access URL when known

## Discord Notifications

- Discord workflow runtime context is keyed by `workflow_run_id`.
- Store at least:
  - `channelId`
  - `lang`
  - `progressMessageId`
  - `runId`
- Runtime notifications currently send:
  - `step.started`
  - final workflow result message
- Started message should include current step and next step.
- Final result message should include `result_url` and a short run summary from release artifacts when available.
