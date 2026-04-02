# Nexus Project Progress Report - 2026-04-02

## Current Status

The v3.1 mainline is still centered on:

`pm_spec -> arch_design -> impl_be -> impl_fe -> smoke_test -> qa_verify -> release_pack -> deploy_preview`

The project is now in a runnable Discord-entry beta state with stronger prompt contracts, stronger validator coverage, and plain-checkout registry validation working without Docker-only stub files.

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

### 3. Project-quality cleanup

- the v3.1 tasklist was aligned to the real `SP-03` `workplan.json` contract.
- checked-in orchestrator config files now let `node orchestrator/scripts/validate_registry.js` pass from a plain checkout.
- `.gitignore` was updated to reduce generated-file noise from preview and quant report outputs.

## Latest Validated Outcome

### Discord-supported beta simulation

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

## QA Assessment

- Nexus critical-path code quality is materially better than earlier in the day: the release path runs end-to-end and the validator/tests now enforce the intended minimal-reviewable scope much more tightly.
- Nexus project quality is improved but not fully closed: the suite-level canary is still not at `2/2 GO`, so `SP-03` is not closed yet.
- The main remaining quality signal is the shallow QA artifact from run `ae540ae7-26fd-47d1-a9ec-a70ebb78bbae`, not a workflow crash.

## Recommended Next Step

1. Finish `SP-03` by eliminating the shallow QA artifact path and getting a clean multi-run `GO` canary.
2. Continue the v3.1 governance stream with `SCO-05 / GOV-01`.
