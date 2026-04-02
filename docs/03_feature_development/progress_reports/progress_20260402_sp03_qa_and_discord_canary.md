# Progress Report - 2026-04-02 SP-03 QA Tightening and Discord Canary

## Summary

Today focused on restoring Nexus project quality after the workspace-reorg side quest and then pushing the v3.1 mainline forward. The result is a healthier `SP-03` path: the workplan contract is tighter, worker-coder fallback output is less likely to drift off-scope, and the Discord-entry workflow now completes successfully on repeated live runs.

The remaining gap is not workflow execution stability. It is quality consistency: the latest 2-run Discord validation finished with `2/2` succeeded workflows but only `1/2` `GO` outcomes because one QA artifact was still judged shallow.

## What Changed

### 1. Worker-coder and workspace-reorg regression repairs

- fixed stale workspace-path assertions in `worker-coder` tests after the workspace structure update
- aligned `entrypoint.sh`, scope guard behavior, and startup smoke expectations with the current `workspace/sandbox/...` layout
- restored a passing local `worker-coder` suite

### 2. SP-03 contract alignment

- updated the v3.1 tasklist to describe the real `plan/workplan.json` schema
- tightened `architect.system_spec.v2` in both registry copies so architect output must provide machine-readable workplan data
- confirmed `workflow_step_builder` injects structured workplan context into both `impl_be` and `impl_fe`

### 3. Validator and fallback hardening

- added validator logic to reject minimal CRM plans that silently add delete, pagination, responsive/mobile expansion, or unrelated backend scope
- tightened CRM fallback scaffolding so minimal-reviewable output stays within the requested surface
- added repair logic for malformed typed handoffs, especially `handoff/impl_to_qa.json` and `handoff/be_to_fe.json`

### 4. Plain-checkout project quality fixes

- added checked-in orchestrator config files so registry validation works without Docker stubs
- removed the Dockerfile placeholder-file workaround
- reduced generated-file review noise via `.gitignore`

## Verification

### Local verification

- `npm.cmd test` in `worker-coder`: passed
- `node --test orchestrator/test/coding_team_validators.test.js orchestrator/test/workflow_step_builder.context.integration.test.js`: passed
- `node orchestrator/scripts/validate_registry.js`: passed

### Discord-entry live validation

Validation command:

- `npm --prefix orchestrator run validate:discord_coding_supported_beta -- --base-url http://localhost:3000 --runs 2 --warmup 0 --concurrency 1 --timeout-sec 2700 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-total-p95-ms 5400000`

Results:

- suite verdict: `FAIL`
- reason: `go_rate 0.500 < 1.000`
- workflow success count: `2/2`
- `GO` count: `1/2`

Run details:

1. `a7fbf2b5-5db0-4cba-abd9-a84c8445d5be` / `ae540ae7-26fd-47d1-a9ec-a70ebb78bbae`
   - workflow: `succeeded`
   - preview: `http://localhost:46006`
   - fidelity: `artifact_complete_but_shallow`
   - root issue: QA artifact depth remained too weak

2. `ec5a4d18-2dea-4a45-889d-52312a863f55` / `817af805-cd4a-475b-ab76-2b721b25de60`
   - workflow: `succeeded`
   - preview: `http://localhost:46007`
   - fidelity: `demo_usable`
   - perceptual quality: `high`
   - preview validation: `preview_matched`

## Assessment

The main quest is still `SP-03`. It is not blocked, but it is not closed. The current codebase can produce a valid Discord-entry release, yet the live evidence still shows inconsistent QA depth across repeated runs.

The next milestone should be a clean repeated-canary result where the workflow still succeeds and the QA/release artifacts stay consistently above the `demo_usable` threshold.
