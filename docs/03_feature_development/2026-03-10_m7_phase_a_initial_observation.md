# M7 Phase A Initial Observation

- Date: 2026-03-10
- Scope: Post-M8 M7 `Phase A: advisory-only`
- Prepared by: QA / Architecture

---

## 1. Current State

Phase A advisory-only was applied with:

- `master_enabled=true`
- `dynamic_routing_enabled=true`
- `router_mode=dynamic_routing_advisory`
- `m7_exposure_cohorts.json.runtime_controls.cohort_enabled=true`

Validation immediately after enablement:

- `live_validate_vnext_runtime.js` -> `PASS`
- `canary_m7_phase_a_advisory.js` -> `PASS`
- `run_m7_dynamic_routing_trial.js` -> `live_trial`

---

## 2. Initial Sample Collection

Controlled advisory-only traffic injection was started through:

- `node orchestrator/scripts/inject_live_traffic.js --count 12 --interval-ms 5000 --approval-safe`
- `node orchestrator/scripts/inject_live_traffic.js --count 3 --interval-ms 5000 --approval-safe`

Observed result from injector:

- first run reached `9/12` successful requests before tool timeout interrupted the terminal session
- second run completed `3/3`
- all observed requests returned `200`

---

## 3. Observation Report

Primary artifact:

- `orchestrator/artifacts/m7_phase_a/phase_a_initial_observation_20260310.json`

Current summary from the latest generated report:

- `routing_samples = 15`
- `workflow_run_samples = 15`
- `gated_parallel_allowed = 12`
- `forced_sequential = 3`
- `forced_sequential_ratio = 0.20`
- `execution_dispatch p50 = 6419ms`
- `execution_dispatch p95 = 10802ms`

Evaluation flags:

- `enough_routing_samples = false`
- `execution_dispatch_observed = true`
- `forced_sequential_ratio_within_limit = true`

---

## 4. Key Finding

Current routing decision sources are still:

- `dynamic_routing_disabled = 15`

Interpretation:

- the prepared Phase A config is correct
- live runtime validation passed earlier
- but the current collected requests did not yet produce `dynamic_routing_advisory_only` records on the local 3000 process
- subsequent investigation showed the running local orchestrator process was likely started through a manual path with ambiguous config/env loading, so final live confirmation is still pending a controlled restart

Investigation result:

1. workflow_id / project_type / classifier_domain_lead in DB samples were valid for the approved cohort
2. runtime code was updated to support fallback config lookup for local `node src/index.js` startup
3. final proof is still pending because the existing local 3000 process was not safely restarted onto the updated code during this session

This means the immediate next engineering task is:

- verify why live requests are not generating advisory-only decision sources
- adjust request shaping or runtime intake mapping before treating Phase A as fully observable

---

## 5. Reporting Window Note

The database timestamp window was initially offset relative to the local execution clock.

The reporting script has now been corrected to anchor the time window to the latest database sample timestamp.

Current report window:

- `anchor_mode = latest_db`
- `anchor_time = 2026-03-09T15:53:56.535Z`
- `since_minutes = 20`

---

## 6. Recommended Next Step

Do not widen scope yet. Do not treat Phase A observability as complete yet.

Next actions:

1. restart the local orchestrator process with explicit runtime env
2. re-run `live_validate_vnext_runtime.js`
3. verify that new `routing_decision_log` rows show `dynamic_routing_advisory_only`
4. only then resume advisory-only sample collection under the same narrow cohort
