# Progress Report: Discord Coding Team Generic-App E2E Validation

- Date: 2026-03-27
- Scope: Discord `/coder` intake, shared Coding Team workflow routing, stale-task cleanup, and real generic-app end-to-end validation

## Completed

1. Reproduced the real Discord coding failure for a non-CRM system request using a dedicated canary suite:
   - `orchestrator/canary_inputs/discord_doc_release_hub_suite_v1.json`
   - initial failure: `workflow 'coding_team_v0' project_type mismatch: expected 'generic_coding_task', got 'generic_app'`
2. Fixed shared-workflow routing so `coding_team_v0` can legally serve multiple coding project types:
   - `configs/registry/capability_registry.json`
   - `orchestrator/src/registry.js`
3. Updated PM-step prompt construction so non-CRM requests explicitly reject CRM template assumptions:
   - `orchestrator/src/domain/workflow_state.js`
4. Restarted orchestrator and coder worker, then re-ran the same request through the real dispatch path.
5. Confirmed the post-fix run entered Coding Team successfully with:
   - `workflow_run_id = ff2d77e5-5fe6-4f37-a280-687cadf90551`
   - `project_type = generic_app`
6. Cleaned stale coding backlog in Postgres and Redis so the active worker would stop consuming obsolete runs first:
   - stale tasks marked failed: `17`
   - stale workflow steps marked failed: `74`
   - stale workflow runs marked failed: `14`
   - stale top-level runs marked failed: `66`

## Validation Outcome

1. The real Discord -> dispatch -> Coding Team chain is now proven to admit `generic_app` requests.
2. The verified run progressed beyond routing and into execution stages:
   - `pm_spec`: succeeded
   - `arch_design`: succeeded
   - `impl_be`: failed
   - `impl_fe`: timed out in parallel branch handling
3. Artifact root for the validated run:
   - `artifacts/release/01bea846-5dc3-4721-9d66-ecd12707f08f`
4. Final workflow result:
   - `status = failed`
   - `error_code = HANDOFF_SECTIONS_MISSING`
   - failing step = `impl_be`

## Findings

1. The original project-type mismatch bug is fixed.
2. The next blocking issue is deeper in the coding pipeline:
   - `handoff/be_to_fe.json` was generated, but the validator still reported missing required sections:
   - `api_contracts`
   - `shared_types`
   - `scope_constraints`
3. Frontend implementation remains vulnerable to watchdog timeout under the current execution path.
4. Generated PM artifacts still show encoding degradation for Chinese source text, even when intake routing succeeds with the original request.

## Remaining Risks

1. Handoff generation and handoff validation are not aligned for backend-to-frontend contracts.
2. Generic-app execution still has timeout risk in implementation stages.
3. Non-ASCII request text can degrade inside generated artifacts, which can distort downstream execution quality.
4. Historical workspace-local prototype files under `sandbox/crm_site/` remain unrelated to this validation thread and were intentionally not included in the core fix scope.

## Recommended Next Tasks

1. Fix the `impl_be` handoff validator/parser mismatch against `handoff/be_to_fe.json`.
2. Investigate why `impl_fe` still hits watchdog timeout after the routing fix.
3. Repair encoding handling for non-ASCII Discord intake text through PM artifact generation.
4. Re-run the same Discord canary after the above fixes and compare artifact completeness and runtime.
