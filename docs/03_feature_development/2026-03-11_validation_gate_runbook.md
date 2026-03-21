# Validation Gate Runbook

- Date: 2026-03-11
- Scope: repeatable release-gate execution for post-M9 mainline review

---

## 1. Purpose

This runbook defines the standard validation entrypoint for PM / Architect release review.

The gate bundles three checks:

- config preflight
- live vNext runtime validation
- live M9 workflow validation

---

## 2. Standard Command

Run from repo root:

```powershell
npm.cmd --prefix orchestrator run validate:next_stage_release_gate
```

Output summary artifact:

- `orchestrator/artifacts/validation/next_stage_release_gate/next_stage_release_gate_summary.json`

Expected child artifacts:

- `orchestrator/artifacts/canary/live_vnext_runtime/live_vnext_runtime_report.json`
- `orchestrator/artifacts/canary/live_m9_workflow/live_m9_workflow_report.json`

### Live Workflow Shortcuts

Use the timeout/failure-path canary when you want to prove timeout enforcement and failure closure:

```powershell
node orchestrator/scripts/live_validate_workflow_runtime.js --base-url http://localhost:3000 --input crm_mini.json --timeout-ms 480000
```

Use the default-timeout success canary when you want a true GO check for the current runtime config:

```powershell
node orchestrator/scripts/live_validate_workflow_runtime.js --base-url http://localhost:3000 --input crm_mini_default_timeout.json --timeout-ms 1500000
```

Interpretation:

- `crm_mini.json` intentionally sets `max_runtime_s=180` and may fail at `impl_be` or `impl_fe` by design.
- `crm_mini_default_timeout.json` does not override the runtime default and should be used for post-restart success validation.

---

## 3. When To Run

Run the full gate:

- before PM / Architect go-no-go review
- after runtime config, compose mount, or startup path changes
- after changes to workflow execution contracts, retry policy, or release-pack evidence

Run config-only mode:

```powershell
node orchestrator/scripts/validate_next_stage_release_gate.js --skip-live
```

Use config-only mode when:

- the local stack is not running
- only startup/config packaging changed
- you want a fast pre-check before full live validation

---

## 4. Pass Criteria

The run is considered pass only if:

- config preflight passes
- live vNext runtime validation passes
- live M9 workflow validation passes
- summary artifact reports `overall: pass`

---

## 5. Failure Handling

If the gate fails:

- use `failed_step` in the summary artifact as the first triage pointer
- read the child report referenced by that step's `report_path`
- do not advance release review until the failing gate is re-run successfully

---

## 6. Governance Notes

This gate is a mainline validation tool.

It does not authorize:

- cohort expansion
- router mode expansion
- new workflow types
- provider surface expansion
