# M2 E2E Live Evidence Runbook

## Purpose
- Capture the remaining live-service evidence required to move `WS-09` and `WS-10` from `conditional pass` toward `pass`.
- Keep execution inside the current allowed scope from `AI_Coding_Agent_System_Prompt.md`.

## Required Service State
- Local orchestrator is reachable at `http://localhost:3000`
- `GET /health` returns `ok`
- Approval token is available if non-default:
  - env: `APPROVAL_TOKEN`
  - or CLI flag: `--approval-token <token>`

## Validation Commands
- Full bundle:
```powershell
cmd /c npm --prefix orchestrator run validate:live_m2_e2e
```

- vNext entrypoint and approval evidence only:
```powershell
cmd /c npm --prefix orchestrator run validate:live_vnext_runtime
```

- Workflow runtime / timeline / artifacts evidence only:
```powershell
cmd /c npm --prefix orchestrator run validate:live_workflow_runtime
```

## Expected Evidence
- `validate:live_vnext_runtime`
  - direct chat bypass returns `mode=direct_reply`
  - direct chat run has zero created tasks
  - risky coding request returns `waiting_approval=true`
  - reject path closes run with `APPROVAL_REJECTED`
  - approve path moves task out of `waiting_approval`

- `validate:live_workflow_runtime`
  - workflow starts through `/workflow-runs/start`
  - terminal workflow state is queryable through `/workflow-runs/:workflow_run_id`
  - run timeline is queryable through `/runs/:run_id/timeline`
  - artifact roots are queryable through `/runs/:run_id/artifacts`

## Evidence Files
- `orchestrator/artifacts/canary/live_vnext_runtime/live_vnext_runtime_report.json`
- `orchestrator/artifacts/canary/live_workflow_runtime/live_workflow_runtime_report.json`

## Pass Criteria
- Both commands return exit code `0`
- Both reports show `"overall": "pass"`
- No missing runtime records for:
  - direct chat bypass
  - approval reject
  - approval approve
  - workflow start
  - workflow terminal state
  - workflow timeline
  - workflow artifacts

## Current Blocker
- As of `2026-03-06`, local service evidence is still blocked because `http://localhost:3000` is not reachable in the current environment.

## Governance Note
- Do not mark `WS-09` or `WS-10` complete until these live reports pass.
