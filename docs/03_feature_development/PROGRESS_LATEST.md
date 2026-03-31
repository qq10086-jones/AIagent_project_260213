# Nexus Project Progress Report - 2026-04-01

## Current Status

The project has moved from design completion into runnable beta-path validation. The core Coding Team workflow is now wired as:

`pm_spec -> arch_design -> impl_be -> impl_fe -> smoke_test -> qa_verify -> release_pack -> deploy_preview`

The most important current milestone is that the Discord-supported beta path now completes successfully through the local live stack, with release artifacts, smoke evidence, preview routing, and quality classification all landing in the expected places.

## What Changed Recently

### 1. Quality-design closure

- `smoke_test` is now part of the formal workflow definition.
- `release_pack` now includes runtime evidence summaries in the release manifest and summary.
- workflow completion summaries now surface runtime evidence instead of relying only on `README` extraction.
- live validation scripts were tightened so they check smoke evidence and runtime evidence from actual artifacts.

### 2. Discord and preview-path stabilization

- Discord intake and workflow status messaging were cleaned up for the current workflow shape.
- preview routing now prefers run-scoped implementation output instead of shared sandbox fallback paths.
- the product-fidelity audit now evaluates the released frontend surface instead of being polluted by unrelated root files.

### 3. Worker-coder fallback hardening

- CRM fallback scaffolding now generates a minimal runnable backend and frontend instead of placeholders.
- backend fallback now serves `public/`, honors `process.env.PORT`, and exposes smokeable API routes.
- frontend fallback now produces a usable same-origin CRM UI for the preview and smoke path.
- scaffold repair logic now upgrades broken placeholder outputs into the new minimal runnable structure.

## Latest Validated Outcome

### Discord-supported beta simulation

Validated via:

- `npm --prefix orchestrator run validate:discord_coding_supported_beta -- --base-url http://localhost:3000 --runs 1 --warmup 0 --concurrency 1 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-total-p95-ms 3600000`

Outcome:

- workflow status: `succeeded`
- go/no-go: `GO`
- smoke root status: `200`
- smoke api status: `200`
- product fidelity: `demo_usable`
- perceptual quality: `high`

Primary evidence:

- latest validation report:
  `orchestrator/artifacts/validation/discord_coding_load_test/2026-03-31T16-57-51-312Z/discord_coding_load_test_report.json`
- successful release summary:
  `artifacts/release/960b2526-54a3-4313-a036-3c505b665578/summary/run_summary.md`
- successful smoke report:
  `artifacts/release/960b2526-54a3-4313-a036-3c505b665578/smoke/smoke_result.json`
- successful product fidelity report:
  `artifacts/release/960b2526-54a3-4313-a036-3c505b665578/qa/product_fidelity_report.json`

## Remaining Gap

The main remaining design gap is `superpowers` live evidence on the Discord-supported beta path. The chain is now green, but the successful run still reports:

- `superpowers_configured_steps = 0`
- `superpowers_available_steps = 0`

So the next closure item is not workflow correctness. It is runtime evidence visibility for the `superpowers` integration in the real Discord/live execution path.

## Recommended Next Step

1. Surface real `superpowers` plugin evidence into the successful live Discord run summary and manifest.
2. After that, run one more live Discord validation and confirm the final summary contains both:
   - smoke evidence
   - superpowers evidence
