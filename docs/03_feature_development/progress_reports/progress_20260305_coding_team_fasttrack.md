# OpenClaw Nexus Progress Update (AGENTS.md Style)

## Snapshot
- Date: `2026-03-05`
- Sprint context: `coding_team_v0` strict-mode stabilization
- North Star: one-command coding-team usability with strict artifact + go/no-go enforcement.
- Current status: fast-track path is online; both CRM and game mini inputs reached strict `GO` in latest validation batch.

## What Was Completed Today
- Canary timeout diagnostics enhanced:
  - `canary_coding_team` now emits `CANARY_TIMEOUT` with `current_step_index/current_step_id/current_task_id`.
  - Timeout failures are now directly actionable instead of generic `timeout waiting workflow`.
- Workflow fast-mode routing delivered:
  - `workflow_engine` now supports `input.fast_mode=true`.
  - `coding.delegate` steps now receive bounded `max_runtime_s` defaults by step:
    - `pm_spec=120`, `arch_design=180`, `impl_fe=240`, `impl_be=240`, `release_pack=120`.
  - Fast mode prompt suffix added to bias concise/required-artifact-first output.
- Canary fixtures moved to fast-track defaults:
  - `crm_mini.json` and `game_mini.json` switched to:
    - `provider=qwen`
    - `model=qwen-plus`
    - `fast_mode=true`
    - `max_runtime_s=180`

## Runtime Evidence (Today)
- CRM strict run passed:
  - command:
    - `npm.cmd --prefix orchestrator run canary:coding_team -- --n 1 --strict true --input crm_mini.json --timeout-sec 900`
  - result:
    - `workflow_run_id=9218e744-9c32-41ee-995a-fcf3deed759a`
    - `run_id=50c1b60f-dbc7-4ab1-a0e0-4f29df27299e`
    - `workflow_status=succeeded`, `go_no_go_verdict=GO`, `duration_s=412`
  - pack validation:
    - `GET /workflow-runs/9218e744-9c32-41ee-995a-fcf3deed759a/validate-pack -> validation.ok=true`
- Game strict run passed:
  - command:
    - `npm.cmd --prefix orchestrator run canary:coding_team -- --n 1 --strict true --input game_mini.json --timeout-sec 900`
  - result:
    - `workflow_status=succeeded`, `go_no_go_verdict=GO`
- Prior failure diagnosis retained as evidence:
  - timeout run reported `current_step_id=arch_design` (now surfaced by canary diagnostic fields).

## Current Gaps
- Consecutive-green target for production sign-off is still not met (`20` strict green not completed yet).
- Single-run wall-clock remains several minutes; acceptable for build tasks but still needs reliability hardening across batches.

## Next Priority (Mainline)
- Run dual-input strict canary batch (`crm_mini + game_mini`) with `n=5` to measure consecutive green trend.
- Expand to `n=20` once `n=5` is stable.
- Keep fast-mode defaults while tracking failure-code distribution and step-duration hotspots.

## Changed Files (Today)
- `orchestrator/scripts/canary_coding_team.js`
- `orchestrator/src/workflow_engine.js`
- `orchestrator/canary_inputs/crm_mini.json`
- `orchestrator/canary_inputs/game_mini.json`
- `docs/03_feature_development/progress_reports/progress_20260305_coding_team_fasttrack.md`

## Tracking Notes
- This update focuses on fast-track usability and observability for coding-team one-command execution.
- Existing unrelated git state was not reverted.
