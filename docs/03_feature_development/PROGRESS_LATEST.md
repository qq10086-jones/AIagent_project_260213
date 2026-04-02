# Nexus Project Progress Report - 2026-04-03

## Current Status

The v3.1 mainline is still centered on:

`pm_spec -> arch_design -> impl_be -> impl_fe -> smoke_test -> qa_verify -> release_pack -> deploy_preview`

The project is now in a runnable Discord-entry beta state with stronger prompt contracts, stronger validator coverage, and plain-checkout registry validation working without Docker-only stub files.

As of `2026-04-03`, the latest locally validated state is no longer the earlier `2/2 workflow, 1/2 GO` canary. The mainline has now passed a fresh end-to-end Discord-supported beta validation after fixing two concrete worker-coder artifact-contract defects.

## What Changed Recently

### 1. SP-03 contract and validator tightening

- `architect.system_spec.v2` now explicitly requires `plan/workplan.json`.
- the structured workplan contract is injected into both `impl_be` and `impl_fe` as execution context.
- architect validation now rejects minimal CRM workplans that drift into out-of-scope features such as delete, pagination, mobile-first expansion, or unrelated auth/python scope.
- regression coverage was added for workplan injection and validator behavior.

### 2. Worker-coder repair and fallback hardening

- stale workspace-reorg test expectations were fixed in `worker-coder`.
- CRM scaffold repair now normalizes typed handoffs and repairs malformed `handoff/impl_to_qa.json` and `handoff/be_to_fe.json` outputs when the model emits the old shape.
- minimal CRM fallback output was narrowed so it no longer silently introduces extra search/delete scope.
- runtime backend manifest repair now infers missing Node dependencies such as `cors` from generated `server.js` and restores the correct ESM package shape for preview/smoke startup.
- successful delegation now re-applies step-contract artifact repair and validates typed handoffs before the result leaves `worker-coder`, so malformed `impl_be` handoffs are retried internally instead of failing later in orchestrator.

### 3. Project-quality cleanup

- the v3.1 tasklist was aligned to the real `SP-03` `workplan.json` contract.
- checked-in orchestrator config files now let `node orchestrator/scripts/validate_registry.js` pass from a plain checkout.
- `.gitignore` was updated to reduce generated-file noise from preview and quant report outputs.

## Latest Validated Outcome

### Discord-supported beta simulation

Previous state from `2026-04-02`:

Validated via:

- `npm --prefix orchestrator run validate:discord_coding_supported_beta -- --base-url http://localhost:3000 --runs 2 --warmup 0 --concurrency 1 --timeout-sec 2700 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-total-p95-ms 5400000`

Outcome:

- workflow success rate: `2/2`
- go rate: `1/2`
- verdict: `FAIL` at suite level because `go_rate 0.500 < 1.000`
- latest successful `GO` run:
  - `workflow_run_id = ec5a4d18-2dea-4a45-889d-52312a863f55`
  - `run_id = 817af805-cd4a-475b-ab76-2b721b25de60`
  - `preview_url = http://localhost:46007`
  - `product_fidelity = demo_usable`
  - `perceptual_quality = high`
- paired successful-but-warned run:
  - `workflow_run_id = a7fbf2b5-5db0-4cba-abd9-a84c8445d5be`
  - `run_id = ae540ae7-26fd-47d1-a9ec-a70ebb78bbae`
  - `preview_url = http://localhost:46006`
  - `product_fidelity = artifact_complete_but_shallow`

Primary evidence:

- validation report:
  `runtime/artifacts/orchestrator/validation/discord_coding_load_test/2026-04-02T08-35-41-916Z/discord_coding_load_test_report.json`
- successful `GO` run summary:
  `runtime/artifacts/release/817af805-cd4a-475b-ab76-2b721b25de60/summary/run_summary.md`
- successful `GO` product fidelity:
  `runtime/artifacts/release/817af805-cd4a-475b-ab76-2b721b25de60/qa/product_fidelity_report.json`
- successful `GO` preview deployment:
  `runtime/artifacts/release/817af805-cd4a-475b-ab76-2b721b25de60/preview/deployment_result.json`

Additional runtime evidence from the latest successful `GO` run:

- `strict_canary_verdict = PASS`
- `preview_validation = PREVIEW_MATCHED`
- `superpowers_detected_steps = 4`
- `superpowers_configured_steps = 4`
- `superpowers_available_steps = 4`
- `superpowers_steps_used = 4`
- `smoke_root_status = 200`
- `smoke_api_status = 401`

### Latest local recovery validation

Validated via:

- `npm.cmd --prefix orchestrator run validate:discord_coding_supported_beta -- --base-url http://localhost:3000 --runs 1 --warmup 0 --concurrency 1 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-total-p95-ms 3600000`

Outcome:

- workflow success rate: `1/1`
- go rate: `1/1`
- verdict: `PASS`
- latest successful `GO` run:
  - `workflow_run_id = 87261874-71d2-4197-812f-8e60df9439b1`
  - `run_id = fa3e3208-ed96-4b38-8fc2-1905f7418af1`
  - `preview_url = http://localhost:46007`
  - `product_fidelity = demo_usable`
  - `perceptual_quality = high`
  - `preview_validation = preview_matched`
  - `total_duration_ms = 1103588`

Primary evidence:

- validation report:
  `runtime/artifacts/orchestrator/validation/discord_coding_load_test/2026-04-02T15-44-30-951Z/discord_coding_load_test_report.json`
- `GO` result:
  `runtime/artifacts/release/fa3e3208-ed96-4b38-8fc2-1905f7418af1/qa/go_no_go_result.json`
- product fidelity:
  `runtime/artifacts/release/fa3e3208-ed96-4b38-8fc2-1905f7418af1/qa/product_fidelity_report.json`
- preview deployment:
  `runtime/artifacts/release/fa3e3208-ed96-4b38-8fc2-1905f7418af1/preview/deployment_result.json`

## QA Assessment

- Nexus critical-path code quality is materially better than before the recovery work: the release path runs end-to-end, preview deployment recovers cleanly, and typed handoff defects are now repaired and validated inside `worker-coder`.
- Nexus project quality is improved enough to claim a fresh local `PASS` for the Discord-supported beta path, but the broader `SP-03` closeout standard should still be multi-run consistency rather than a single clean rerun.
- The earlier shallow-QA signal from run `ae540ae7-26fd-47d1-a9ec-a70ebb78bbae` is still useful historical evidence, but it is no longer the latest blocker. The concrete blocker that was actually fixed was a combination of runtime package-manifest drift and success-path handoff-contract drift in `worker-coder`.

## Recommended Next Step

1. Re-run the Discord-supported beta canary in multi-run mode to confirm the new `PASS` result holds beyond a single recovery run.
2. If the multi-run canary stays clean, close `SP-03` and continue the v3.1 governance stream with `SCO-05 / GOV-01`.
