# 2026-03-10 Runtime Restart Open Issue

- Date: 2026-03-10
- Owner: QA / Architecture
- Severity: P0
- Status: Open

---

## Summary

The `post-M8 M7 Phase A` runtime/package work is complete, but final live confirmation is blocked by the currently running local orchestrator process on `localhost:3000`.

The active process appears to be a manually started `node src/index.js` instance with non-transparent startup environment. During this session, the runtime code was updated so advisory routing can resolve config files correctly under local startup, but the live 3000 process was not safely restarted onto that updated code.

As a result:

- design integration is complete
- runtime/code changes are complete
- tests are green
- Phase A live observability is **not yet fully confirmed**

---

## Evidence

Confirmed during investigation:

1. Latest workflow samples already carried valid cohort fields:
   - `workflow_id = coding_team_v0`
   - `project_type = webapp_crm`
   - `classifier_domain_lead = fe_led`
2. Prepared runtime configs are correct:
   - `configs/production_parallel_rollout.json`
   - `configs/m7_exposure_cohorts.json`
3. Local gate replay with the same values returns:
   - `routing_decision_source = dynamic_routing_advisory_only`
4. Updated tests passed:
   - `node --test orchestrator/test/parallel_rollout_gate.test.js orchestrator/test/routing_audit_log.test.js`
   - result: `28 / 28 PASS`

---

## Root Cause

The remaining issue is not the cohort definition itself.

The remaining issue is the local orchestrator runtime lifecycle:

- `localhost:3000` is served by a local `node src/index.js` process
- that process was started outside the current session with unclear env injection
- a direct restart attempt showed environment ambiguity and did not complete cleanly within this session

This means the updated advisory-routing code path is not yet guaranteed to be active on the live process currently serving requests.

---

## Impact

- Do not treat current `dynamic_routing_disabled` live rows as final architectural truth
- Do not widen Phase A scope
- Do not enter any `Phase B` discussion
- Pause new evidence collection until the process is restarted in a controlled way

---

## Required Recovery Actions

1. Stop the current local orchestrator process on `localhost:3000`
2. Restart it with explicit local runtime env:
   - `PGHOST`
   - `PGPORT`
   - `PGUSER`
   - `PGPASSWORD`
   - `PGDATABASE`
   - `REDIS_URL`
   - `WORKSPACE_ROOT`
   - `APPROVAL_TOKEN`
3. Re-run:
   - `node orchestrator/scripts/live_validate_vnext_runtime.js --base-url http://localhost:3000 --approval-token dev-approval-token --timeout-ms 180000`
4. Query latest routing rows and confirm new records contain:
   - `router_mode = dynamic_routing_advisory`
   - `dynamic_routing_enabled = true`
   - `routing_decision_source = dynamic_routing_advisory_only`
5. Resume Phase A evidence collection only after step 4 succeeds

---

## Exit Condition

This issue may be closed only when a restarted local orchestrator process produces fresh `routing_decision_log` rows with `dynamic_routing_advisory_only` under the approved Phase A cohort.
