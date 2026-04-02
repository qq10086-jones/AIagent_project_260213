# Progress Report - 2026-04-03 Nexus Worker-Coder Recovery and Canary Pass

## Summary

Today focused on the concrete blocker behind the long-running Nexus validation attempts. The key finding was that the mainline was not failing because the workflow could not finish. It was failing because `worker-coder` could still emit release artifacts that looked structurally complete at a coarse level while remaining invalid for preview startup or typed handoff validation.

Two real defects were fixed:

1. generated backend runtime manifests could omit dependencies required by the generated `server.js`
2. successful delegation did not re-apply step-contract repair or typed-handoff validation before returning control to orchestrator

After these fixes, the Discord-supported beta validation passed locally end-to-end.

## What Changed

### 1. Runtime package manifest repair

- `worker-coder/artifact_scaffold.js` now infers a Node runtime manifest from generated backend source
- the repair path restores missing dependencies such as `cors`
- the repair path also restores `type: "module"` and `main: "server.js"` when the generated backend is ESM-based

### 2. Success-path handoff enforcement

- `worker-coder/coding_service.js` now expands successful step scaffolding to include the full step-contract artifact set, not just the caller-supplied `expected_artifacts`
- successful delegation now validates both the step artifact contract and the typed handoff contract before returning
- malformed `impl_be -> impl_fe` handoffs are therefore repaired or retried inside `worker-coder`, instead of surfacing later as orchestrator-side `HANDOFF_TYPED_FIELDS_MISSING`

### 3. Regression coverage

- added regression coverage in `worker-coder/tests/artifact_scaffold.test.js`
- added coverage for step-contract artifact expansion in `worker-coder/tests/coding_service_pure.test.js`
- existing salvage-path coverage remained green

## Root Cause Details

### Defect A: preview/smoke startup drift

The failing release artifact contained a backend `server.js` that imported `cors`, while the paired `impl/be_changes/package.json` only declared `express`. That caused preview readiness and smoke startup to fail even though the workflow step itself appeared green.

### Defect B: success-path handoff drift

The failing `handoff/be_to_fe.json` used an old object-shaped schema for `api_contracts`, `shared_types`, and `scope_constraints`, and it omitted required top-level fields such as `from_step`, `to_step`, and `be_changes_path`. The repair logic already existed in scaffold code, but the successful delegation path was not forcing that handoff back through the full contract set.

## Verification

### Unit and regression checks

- `node worker-coder/tests/artifact_scaffold.test.js`: passed
- `node worker-coder/tests/coding_service_salvage.test.js`: passed
- `node worker-coder/tests/coding_service_pure.test.js`: passed

### Full Discord-supported beta validation

Command:

- `npm.cmd --prefix orchestrator run validate:discord_coding_supported_beta -- --base-url http://localhost:3000 --runs 1 --warmup 0 --concurrency 1 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-total-p95-ms 3600000`

Result:

- suite verdict: `PASS`
- workflow success count: `1/1`
- `GO` count: `1/1`
- workflow run: `87261874-71d2-4197-812f-8e60df9439b1`
- release run: `fa3e3208-ed96-4b38-8fc2-1905f7418af1`
- product fidelity: `demo_usable`
- perceptual quality: `high`
- preview validation: `preview_matched`
- total duration: `1103588 ms`

Primary evidence:

- `runtime/artifacts/orchestrator/validation/discord_coding_load_test/2026-04-02T15-44-30-951Z/discord_coding_load_test_report.json`
- `runtime/artifacts/release/fa3e3208-ed96-4b38-8fc2-1905f7418af1/qa/go_no_go_result.json`
- `runtime/artifacts/release/fa3e3208-ed96-4b38-8fc2-1905f7418af1/preview/deployment_result.json`

## Assessment

The long-run blocker was real and code-level, not just a flaky QA judgment. After the repair, Nexus is back in a locally validated end-to-end `PASS` state for the Discord-supported beta path.

The remaining work is no longer “find why the workflow cannot finish.” It is “prove the repaired path stays stable across repeated canary runs.”
