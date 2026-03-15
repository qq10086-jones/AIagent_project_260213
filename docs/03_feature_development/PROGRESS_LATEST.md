# Feature Progress - Latest Snapshot

**Last updated:** 2026-03-15 (M10 load test PASS + code quality hardening complete; worker-coder test suite 18/18; orchestrator 150/150)
**Author:** PM / Architecture Review

---

## Execution Evidence

- **30-minute accelerated validation completed:** controlled live injection finished and compressed-go/no-go evidence package produced.
- **Observed routing samples:** `89`
- **Observed workflow samples:** `89`
- **Parallel admission:** `71` gated-parallel-allowed
- **Forced sequential:** `18`
- **Forced sequential ratio:** `20.2%`
- **Execution dispatch latency:** `P50 6548ms`, `P95 10834ms`
- **Live runtime validation:** `PASS` on 2026-03-10
- **Real local LLM validation:** `PASS` against local `deepseek-r1:32b` on 2026-03-09
- **M7 Phase A advisory-only code/config package:** completed on 2026-03-10
- **Phase A live runtime validation after enablement:** `PASS` on 2026-03-10
- **M9 runtime config preflight:** `PASS` on 2026-03-11
- **M9 real live workflow validation:** `PASS` on 2026-03-11
- **Next-stage release gate (config-only):** `PASS` on 2026-03-11
- **Next-stage release gate (full live):** `PASS` on 2026-03-11
- **Runtime boot source validation:** `PASS` on 2026-03-11
- **Brain gateway typed contract handlers/tests:** landed on 2026-03-11
- **Worker-coder structural decomposition:** completed on 2026-03-11 (`coding_service.js` reduced to ~705 lines, now ~991 lines after step_artifact_contract extraction)
- **Worker lifecycle single-finalization guard:** landed on 2026-03-11 (`task_lifecycle.js` + targeted tests)
- **Stable local OpenCode/Ollama execution lane:** `PASS` on 2026-03-12 (`opencode + ollama/glm-4.7-flash:latest`)
- **Authoritative stable-lane full four-case cohort:** `PASS` on 2026-03-12 (`4 pass / 0 fail / 0 partial`)
- **M10 load test (stable_local_lane, 5 runs):** `PASS / GO verdict` on 2026-03-15 — run a48a521e: 6/6 steps, 23 minutes, `strict_canary_report` PASS, `go_no_go_result = GO`
- **Post-M10 code quality hardening (QA/Architect audit P0–P2 + round 2):** completed on 2026-03-15

---

## Milestone Summary

| Milestone | Description | Status | Date |
|-----------|-------------|--------|------|
| M2 | Core Orchestration | **CLOSED** | 2026-03-07 |
| M3 | vNext Service Layer | **CLOSED** | 2026-03-07 |
| M4 | LLM Dispatcher + Role Policy | **CLOSED** | 2026-03-08 |
| M5 | Workflow DAG Engine | **CLOSED** | 2026-03-08 |
| M6 | Parallel Rollout Readiness | **GO_LIMITED_EXPOSURE** | 2026-03-09 |
| M7 | Limited Dynamic Routing v1 | **CLOSED - ACCEPTED WITH DEVIATION** | 2026-03-09 |
| M8 | Staging Evidence and Live Routing Validation | **CLOSED** | 2026-03-09 |
| M9 | Coding Precision & Sandbox Guardrails | **CLOSED** | 2026-03-12 |
| M10 | Load Test + Execution Lane Validation | **CLOSED** | 2026-03-15 |

---

## Active Design Authority

**Active milestone:** M10 CLOSED. No active milestone. Next step is M11 scoping or continued stabilization.

Current governance state:
- M10 load test produced a PASS / GO verdict on 2026-03-15 using `stable_local_lane` (opencode + ollama/glm-4.7-flash:latest).
- Post-M10 QA/architect audit identified and resolved 13 issues across two rounds (P0–P2 each round).
- All known code quality debts in the critical execution path are now addressed.
- `configs/production_parallel_rollout.json`: `master_enabled=true`, `dynamic_routing_enabled=true`, `router_mode=dynamic_routing_enforced`.
- Qwen/opencode direct coding lane (`alibaba-coding-plan/qwen3-coder-plus`) remains unresolved — `opencode` built-in provider path rejects the DashScope credential. Tracked as known risk, not a current blocker.

Governing documents:

| Document | Path |
|----------|------|
| Governance v3 | `docs/01_design/system/260309/260309_1048/OpenClaw_Execution_Governance_Scope_Control_v3.md` |
| Architect Contract | `docs/01_design/system/260307/Architect_Engineer_Role_Contract.md` |
| M10 Draft Task List | `docs/03_feature_development/2026-03-12_m10_draft_tasklist.md` |
| M10 Load Test Spec | `docs/03_feature_development/2026-03-12_m10_load_test_spec.md` |
| M10 Phase B Sign-off | `docs/governance/2026-03-12_m10_phase_b_limited_enforced_signoff.md` |

---

## M10 Status - CLOSED

### What was done (2026-03-13 → 2026-03-15)

**Round 1 (QA/Architect audit — first pass, 6 issues):**

| # | Fix | Commit |
|---|-----|--------|
| 1 | Hardcoded `TASK_RUNNING_TIMEOUT_SEC = "900"` in `src/index.js` overrode config file value (1800). Changed default to `""` so runtime config takes effect. | `d365036` |
| 2 | `stream_batch_size` was hardcoded `5` in two places in `worker.js`. Replaced with runtime-configurable `CODER_STREAM_BATCH_SIZE` (default 1) to prevent concurrent LLM saturation on single-GPU setups. | `d365036` |
| 3 | `coding_service.js` contained 100+ lines of dynamic import plumbing (`resolveOrchestratorModule`, etc.) duplicating orchestrator logic inline. Extracted into `worker-coder/step_artifact_contract.js` (121 lines). | `d365036` |
| 4 | `salvageWorkflowArtifactFailure` was private, had zero test coverage despite being the most complex failure path. Exported it and added `coding_service_salvage.test.js` (5 cases, real filesystem + real validators). | `e0dcab6` |
| 5 | `artifact_scaffold.js` validated `plan/interfaces.md` only for CRM-specific headings. Generalized to `["interfaces"]` only; added heading checks for `plan/spec.md`, `plan/arch.md`, and handoff JSON schemas. | `d365036` |
| 6 | `worker-coder/tests/opencode_adapter.test.js` and `provider_preflight.test.js` had gaps after recent adapter changes. Updated to cover new error-code paths and ollama model ref allowance. | `25b0860` |

**Round 1 follow-up (opencode lane triage hardening):**
- Removed rejection block in `opencode_adapter.js` for `ollama/*` model refs (was incorrectly blocking local lane).
- Added early-return fast-path in `mapErrorCode` for successful proc.
- Fixed `production_parallel_rollout.json` comment that contradicted actual config values (dynamic_routing was true but comment said HOLD).

**Round 2 (QA/Architect audit — deeper pass, 7 issues → P0–P2 addressed):**

| Priority | Fix | Commit |
|----------|-----|--------|
| P0 | `step_artifact_contract.js` tests (`step_artifact_contract.test.js`, 18 assertions): all 6 step IDs, null/undefined inputs, fresh-object-per-call guarantee. | `e0dcab6` |
| P0 | `startup_smoke.test.js` assertions: added `coding_service must import from step_artifact_contract` and `step_artifact_contract.js must exist`. | `e0dcab6` |
| P1 | `workflow_step_builder.js` `wall_clock_timeout_s` clamp boundary tests (3 cases): config below `max_runtime_s`, production scenario, no-config default. Orchestrator tests: 147 → 150. | `36cb2a7` |
| P1 | 16 `context_budget_*.json` golden files updated after new boundary tests triggered re-generation. | `148c166` |
| P2 | `payloadToAdapterRequest` and `requiresScopedTargetPaths` exported and unit-tested (`coding_service_pure.test.js`, 44 assertions): full round-trip, defaults, type coercions, edge cases. | `9c68bbf` |
| P3 deferred | `createStepBuilder` inner closure (`buildStepPayload`) — 4 captured services make extraction noisy with no real testability gain. Deferred indefinitely. |

**Runtime config changes landed:**
- `global_task_timeout_ms: 1800000` (was 900000)
- `wall_clock_timeout_s_default: 900`
- `running_timeout_sec: 1800`
- `stream_batch_size: 1` (single GPU, prevents concurrent LLM saturation)
- `execution_lane_default: stable_local_lane`
- `runtimeByStep` per-step max_runtime_s: `pm_spec 120→360`, `arch_design 180→480`, `impl_fe 240→360`

**.gitignore entries added:**
- `orchestrator/artifacts/validation/m10_load_test/`
- `orchestrator/orchestrator/`
- `sandbox/crm_site/workflow_impl_*.js`
- `**/opencode.json`

---

## Current Verification Status

```text
npm --prefix orchestrator test                                -> 150 / 150 PASS  (2026-03-15)
npm --prefix worker-coder test                               -> 18 / 18 PASS   (2026-03-15)
coding_team_e2e canary                                       -> PASS (happy_path 6/6 + be_failure correct failure)
M10 load test (stable_local_lane, run a48a521e)              -> PASS / GO verdict (6/6 steps, 23 min) (2026-03-15)
canary:m10_phase_b_enforced                                  -> PASS 3 / 3 (2026-03-12)
canary:m10_phase_b_parallel_isolation                        -> PASS (2026-03-12)
canary:m10_observability_correlation                         -> PASS (2026-03-12)
worker-coder/tests/step_artifact_contract.test.js            -> PASS 18 assertions (2026-03-15)
worker-coder/tests/coding_service_salvage.test.js            -> PASS 5 cases (2026-03-15)
worker-coder/tests/coding_service_pure.test.js               -> PASS 44 assertions (2026-03-15)
orchestrator/test/workflow_step_builder.context.integration  -> PASS 150 / 150 (2026-03-15)
```

Historical validations (still valid):
```text
node --test orchestrator/test/*.test.js                       -> 127 / 127 PASS  (2026-03-09)
pytest brain/tests/                                           ->  11 /  11 PASS  (2026-03-09)
canary_m7_phase_a_advisory.js                                 -> PASS 4 / 4 (2026-03-10)
canary_m9_coding_guardrails.js                                -> PASS (2026-03-10)
worker-coder canary:m9_autofix_retry                          -> PASS (2026-03-10)
validate:worker_coding_cohort_execute (full four-case cohort) -> PASS 4 / 4 (2026-03-11)
validate:worker_coding_cohort_execute (promote mode)          -> PASS 4 / 4 (2026-03-12)
```

---

## Blocking Points

No active P0 blockers.

Recently cleared:

| Block | Status | Resolution |
|-------|--------|------------|
| Hardcoded `TASK_RUNNING_TIMEOUT_SEC = "900"` in index.js overrode config file | RESOLVED | Changed default to `""` in env destructuring |
| `stream_batch_size` hardcoded 5 caused concurrent LLM saturation on single GPU | RESOLVED | Configurable via `CODER_STREAM_BATCH_SIZE` / runtime config, default 1 |
| `coding_service.js` contained inlined orchestrator dynamic import plumbing | RESOLVED | Extracted to `step_artifact_contract.js` |
| `ollama/*` model refs rejected by opencode_adapter.js | RESOLVED | Removed rejection block |

Known non-blocking issues:

| Issue | Severity | Status |
|-------|----------|--------|
| Qwen via `opencode alibaba-coding-plan` provider path fails auth despite valid DashScope key | Medium | Ongoing — stable_local_lane (ollama) is primary lane; cloud lane (`stable_cloud_lane`) defined but untested end-to-end |
| `createStepBuilder` inner closure (`buildStepPayload`) cannot be unit-tested independently | Low | Deferred — integration coverage via orchestrator tests + e2e canary is sufficient |

---

## TODO (Next Steps)

### If continuing stabilization (no new milestone)

| # | Task | Priority |
|---|------|----------|
| S1 | Resolve Qwen/opencode auth — test `stable_cloud_lane` end-to-end with a real `alibaba-coding-plan` API key | P1 |
| S2 | Run M10 load test with `stream_batch_size > 1` to establish multi-workflow throughput baseline | P2 |
| S3 | Expand `context_budget_*.json` golden coverage — current goldens reflect test-generated runs, not real LLM-driven runs | P3 |

### If starting M11

Recommended M11 scope candidates (not decided):
1. **Multi-workflow concurrency hardening** — validate `stream_batch_size=2+` with real GPU, measure queue depth, latency, and OOM risk
2. **Qwen cloud lane activation** — resolve `alibaba-coding-plan` provider auth and promote `stable_cloud_lane` to production-ready
3. **Cohort expansion** — widen beyond `coding_team_v0 / webapp_crm / fe_led` to a second project type
4. **Failure injection suite** — systematic T-33 failure injection (timeout, OOM, bad output) against the validated pipeline

---

## Known Risks

| ID | Risk | Severity | Status |
|----|------|----------|--------|
| R-13 | Classifier misrouting under production traffic | High | Mitigated |
| R-14 | Dynamic routing overriding static safety guardrails | High | Mitigated |
| R-15 | Routing source ambiguity in audit trail | High | Mitigated |
| R-16 | `model_tier` drift causing unstable execution choices | High | Mitigated |
| R-17 | No fast rollback path if M7 behaves badly | High | Mitigated |
| R-18 | Limited exposure sample bias | Medium | Mitigated |
| R-19 | Classifier unavailable during decision path | High | Mitigated |
| R-NEW-01 | `brain/` still has direct DB coupling without API boundary | High | Mitigated |
| R-NEW-02 | Workflow engine complexity regresses upward again | Medium | Mitigated |
| R-NEW-03 | Production/staging config drift | Medium | Resolved |
| R-NEW-04 | Local manual startup path bypasses intended config roots / env injection | High | Resolved |
| R-NEW-05 | Qwen via opencode alibaba-coding-plan path fails auth | Medium | Ongoing / non-blocking |
| R-NEW-06 | Single-GPU constraint limits multi-workflow concurrency | Medium | Known / stream_batch_size=1 is mitigation |

---

## Key Artifact Index

| Artifact | Path | Purpose |
|----------|------|---------|
| Root rollout config | `configs/production_parallel_rollout.json` | Prepared runtime gate state |
| Root M7 cohort config | `configs/m7_exposure_cohorts.json` | Phase A cohort restriction |
| Runtime defaults | `configs/runtime/runtime_defaults.json` | All runtime tuning parameters |
| Exposure policy | `configs/parallel_exposure_policy.json` | Allowed M6 exposure cohorts |
| M10 load test artifacts | `orchestrator/artifacts/canary/coding_team_e2e/` | Latest canary run evidence |
| M10 load test spec | `docs/03_feature_development/2026-03-12_m10_load_test_spec.md` | Quantified test specification |
| M10 Phase B Sign-off | `docs/governance/2026-03-12_m10_phase_b_limited_enforced_signoff.md` | Governance approval |
| M6 accelerated validation report | `orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json` | Measured compressed evidence |
| Phase A enabled trial result | `orchestrator/artifacts/m7_trial/preflight_result_20260310_phase_a_enabled.json` | Post-enable cohort-scoped trial evidence |

---

## Code Metrics (2026-03-15)

| Module | File | Lines |
|--------|------|-------|
| worker-coder | `coding_service.js` | ~991 |
| worker-coder | `step_artifact_contract.js` | 121 (extracted) |
| orchestrator | `src/index.js` | ~546 |
| orchestrator | `src/workflow_engine.js` | ~431 |
| orchestrator | `src/domain/workflow_step_builder.js` | 626 |

Test counts:
- Orchestrator: **150 / 150**
- worker-coder: **18 / 18**
  - New in this session: `step_artifact_contract.test.js` (18), `coding_service_salvage.test.js` (5 cases), `coding_service_pure.test.js` (44 assertions)
