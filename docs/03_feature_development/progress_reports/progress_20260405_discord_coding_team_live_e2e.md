# Discord Coding Team Live E2E - 2026-04-05

## Scope

Run a live HTTP entry test that simulates a Discord `/coder:` command entering Nexus, routes into `coding_team_v0`, and verifies the real workflow through release and preview deployment.

## Simulated Discord Command

```text
/coder: Build a minimal CRM web app with customer list, detail page, and add/edit form. Keep changes reviewable and include required artifacts.
```

## Execution Method

1. Probe the live Discord entry path:
   - `POST http://localhost:3000/vnext/dispatch`
2. Run the formal supported-beta validation suite:
   - `npm --prefix orchestrator run validate:discord_coding_supported_beta -- --base-url http://localhost:3000 --runs 1 --warmup 0 --concurrency 1 --strict false --min-workflow-success-rate 1.0 --min-go-rate 0.0 --max-total-p95-ms 3600000`
3. Query the live workflow state directly:
   - `GET http://localhost:3000/workflow-runs/<workflow_run_id>`
4. Cross-check simulated control-plane tests:
   - `node --test orchestrator/test/discord_entrypoint_workflow_e2e.integration.test.js orchestrator/test/discord_dispatch.integration.test.js`

## Primary Evidence

- Formal load-test JSON:
  - `runtime/artifacts/orchestrator/validation/discord_coding_load_test/2026-04-05T02-48-29-280Z/discord_coding_load_test_report.json`
- Formal load-test Markdown:
  - `runtime/artifacts/orchestrator/validation/discord_coding_load_test/2026-04-05T02-48-29-280Z/discord_coding_load_test_report.md`
- Workflow timeline:
  - `artifacts/runs/1aca8f74-57a5-46bd-a7b5-5a5673289045/timeline.md`
- Release artifact root:
  - `runtime/artifacts/release/1aca8f74-57a5-46bd-a7b5-5a5673289045/`

## Formal Load-Test Result

The official validator reported:

- `dispatch_status = 200`
- `dispatch_mode = progress_update`
- `workflow_id = coding_team_v0`
- `project_type = webapp_crm`
- `dispatch_latency_ms = 5948`
- `workflow_run_id = f5fbd6db-2f61-41b8-a036-3e3f6c56c840`
- `run_id = 1aca8f74-57a5-46bd-a7b5-5a5673289045`
- `timeout_ms = 1800000`
- `workflow_status_at_timeout = timeout`
- `total_duration_ms_at_timeout = 1810615`
- formal verdict: `FAIL`
- error code: `LOAD_TEST_TIMEOUT`

This report is factually correct for the validator window: the poller stopped after 30 minutes.

## Post-Timeout Live Reconciliation

The key point is that the workflow did **not** fail after that timeout. A later live query to the same workflow run showed:

- run status: `succeeded`
- `release_pack`: `succeeded`
- `deploy_preview`: `succeeded`
- preview URL: `http://localhost:46004`
- workflow `created_at`: `2026-04-05T02:48:33.267Z`
- workflow `updated_at`: `2026-04-05T03:24:37.029Z`
- actual end-to-end duration: about `2163762 ms` (`36.1 min`)

Important late-stage timings:

- `release_pack` started at `2026-04-05T03:17:58.116Z`
- `release_pack` ended at `2026-04-05T03:24:35.003Z`
- `deploy_preview` ended at `2026-04-05T03:24:36.664Z`

The run therefore exceeded the validator timeout by roughly `364 s` (`6.1 min`) and completed successfully afterward.

## What This Proves

- Discord-style `/coder:` intake is live in the current runtime.
- The request is normalized correctly and routed into `coding_team_v0`.
- The live workflow completed all major stages:
  - `pm_spec`
  - `arch_design`
  - `impl_be`
  - `impl_fe`
  - `smoke_test`
  - `qa_verify`
  - `release_pack`
  - `deploy_preview`
- Release artifacts and a preview deployment were both produced in the live path.

## Control-Plane Contrast

The simulated Discord control-plane tests also passed:

- `orchestrator/test/discord_entrypoint_workflow_e2e.integration.test.js`
- `orchestrator/test/discord_dispatch.integration.test.js`

This confirms the Discord intake and workflow dispatch layers were already healthy. The only issue exposed here was validator calibration, not routing correctness.

## Final Assessment

Current status is **live full-chain success with a timeout-calibration issue**.

- Entry success: `PASS`
- Workflow routing success: `PASS`
- Full workflow completion: `PASS`
- Release artifact generation: `PASS`
- Preview deployment: `PASS`
- Formal supported-beta validator verdict: `FAIL` because the polling window was too short

## Immediate Next Fix Target

Adjust the Discord coding load-test timeout for medium-complexity supported-beta scenarios so the validator does not report a false negative when the workflow legitimately completes after 30 minutes.

Minimum recommendation:

- keep the original 30-minute artifact as raw evidence
- rerun the supported-beta suite with `--timeout-sec 2400` or higher
- treat workflow run `f5fbd6db-2f61-41b8-a036-3e3f6c56c840` as the decisive proof that the live chain works end to end

## Manual Regression Follow-Up

After the initial full-chain success was confirmed, several manual live Discord-style regressions were run to identify what currently prevents a repeatable fresh pass on the coding-team path.

### Regression Evidence Roots

- `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T12-38-40+09-00/`
- `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T15-32-21+09-00/`
- `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T15-39-57+09-00/`
- `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T15-48-03+09-00/`
- `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T15-56-04+09-00/`
- `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T16-05-42+09-00/`

### What Was Fixed Along the Way

The manual regressions exposed and helped close a sequence of real contract issues:

- `impl_be -> impl_fe` typed handoff was missing `be_changes_path`
- `arch_design` was still bound to an older prompt script in the live runtime
- `architect_to_impl.json` initially omitted top-level `workplan`
- minimal-scope CRM plans still drifted into unrequested responsive/mobile scope
- `architect_to_impl.json` then still drifted in field shape, using object-rich `modules/interfaces/risks` and non-schema decision keys

These issues were corrected in the prompt registry, workflow binding, and regression tests before each subsequent live rerun.

### Current Frontier

The latest manual run is:

- workflow run: `278162b1-4a28-447f-813a-a4b8d8875866`
- run: `b67f8a47-1713-4024-b7bd-b23c98a7817e`
- evidence root: `runtime/artifacts/orchestrator/validation/manual_discord_live_e2e/2026-04-05T16-05-42+09-00/`

Its outcome is materially different from the earlier architecture failures:

- `pm_spec`: succeeded
- `arch_design`: succeeded
- `impl_be`: failed
- current error code: `OpenCode command failed`

This is an important milestone. It means the architecture-layer blockers are no longer the primary issue in the fresh live path. The new decisive blocker is backend execution stability inside `impl_be`.

### Remaining Issues

Current Nexus coding-team live-path issues are therefore:

- `R1`: the official supported-beta validator still needs a longer timeout to avoid false negatives on medium-complexity runs
- `R2`: a fresh repeatable live run has not yet completed beyond `impl_be` after the architecture-contract fixes
- `R3`: `impl_be` currently fails in the live runtime with `OpenCode command failed`; backend execution logs and worker stderr are the next required debugging target
- `R4`: because the newest fresh run stops at `impl_be`, there is not yet fresh post-fix evidence for `impl_fe`, `qa_verify`, `release_pack`, or `deploy_preview`

### Assessment Update

The correct current assessment is:

- historical live full-chain success: proven
- Discord intake and workflow dispatch: healthy
- architecture contract chain: materially hardened and now passing in fresh live runs
- current regression frontier: `impl_be`
- next decisive task: repair backend execution on the live path, then rerun to obtain fresh end-to-end evidence through release and preview
