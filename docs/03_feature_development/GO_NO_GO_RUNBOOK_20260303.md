# Coding Team Go/No-Go Runbook (v1.4.1)

## Purpose
Use a deterministic checklist before promoting a `coding_team_v0` run to production usage.

## Required Inputs
- `workflow_run_id` (preferred) or `run_id`
- Release artifacts under `artifacts/release/<run_id>/`

## Gate Criteria (Must All Pass)
1. Workflow status is `succeeded`.
2. Step success is complete (`6/6` for `coding_team_v0`).
3. Acceptance gate step (`qa_verify`) is `succeeded`.
4. Artifact pack validator passes.
5. Strict canary report exists and verdict is `pass`.
6. Strict canary `missing_artifacts_total=0`.

## Command
Run from `orchestrator/`:

```bash
npm run validate:go-nogo -- --workflow-run-id <workflow_run_id>
```

Alternative by `run_id`:

```bash
npm run validate:go-nogo -- --run-id <run_id>
```

## Output
- Console verdict: `GO` or `NO_GO`
- Persisted JSON:
  - `artifacts/release/<run_id>/qa/go_no_go_result.json`

## Decision Rule
- `GO`: all criteria pass.
- `NO_GO`: any criteria fails. Promotion is blocked until rerun or fix.

## Enforcement Status
- Enforced in workflow finalize path (`orchestrator/src/workflow_engine.js`):
  - release pack generation now writes `qa/go_no_go_result.json` automatically.
  - if verdict is `NO_GO`, workflow finalize is converted to failure (`ARTIFACT_INCOMPLETE` path).

## Notes
- Script is strict by default (`GONOGO_STRICT=true`).
- The script also writes machine-readable failure reasons for audit.
