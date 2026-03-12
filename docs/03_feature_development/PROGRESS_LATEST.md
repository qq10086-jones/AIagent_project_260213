# Feature Progress - Latest Snapshot

**Last updated:** 2026-03-13 (stable local OpenCode/Ollama execution lane validated; authoritative worker-coding full-slice revalidation restored to pass; Qwen/OpenCode remains isolated triage)
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
- **Current coding focus:** M9 closeout tasks after core guardrails landed
- **M9 runtime config preflight:** `PASS` on 2026-03-11
- **M9 real live workflow validation:** `PASS` on 2026-03-11
- **Next-stage release gate (config-only):** `PASS` on 2026-03-11
- **Next-stage release gate (full live):** `PASS` on 2026-03-11
- **Runtime boot source validation:** `PASS` on 2026-03-11
- **Brain gateway typed contract handlers/tests:** landed on 2026-03-11
- **Worker-coder structural decomposition:** completed on 2026-03-11 (`coding_service.js` reduced to ~705 lines)
- **Worker lifecycle single-finalization guard:** landed on 2026-03-11 (`task_lifecycle.js` + targeted tests)
- **Worker-coding task contract v1 landing:** in progress on 2026-03-11 (`task_class` / `context_envelope` / `failure_attribution` compatible fields landed)
- **Worker-coding contract authority assets:** landed on 2026-03-11 (schema + beta template registry + validation command)
- **Worker-coding cohort task matrix:** landed on 2026-03-11 (initial four-class beta validation set defined)
- **Worker-coding cohort result format:** landed on 2026-03-11 (schema + validation command ready for first cohort run)
- **Worker-coding cohort execution plan:** landed on 2026-03-11 (machine-readable four-task cohort plan validated)
- **Worker-coding contract consistency hardening:** landed on 2026-03-11 (shared task-class authority + template/task-class mismatch guard)
- **Worker-coding first multi-class cohort cycle:** completed on 2026-03-11 (`4/4` partial, no fail; current gap is verification tier, not workflow closure)
- **Worker-coding result-quality hardening v1:** landed on 2026-03-11 (`verification_plan` execution + achieved-tier evidence persisted)
- **Worker-coding repo-aware verification source:** landed on 2026-03-11 (`sandbox/crm_site` package scripts now back live cohort verification tiers)
- **Worker-coding live cohort truthful-fail signal:** recorded on 2026-03-11 (`0 pass / 4 fail / 0 partial` after real verification enforcement)
- **Worker-coding cohort structural recovery:** completed on 2026-03-11 (runtime source drift, contract-field drop, and single-file scope fallback repaired)
- **Worker-coding full four-case cohort rerun:** `PASS` on 2026-03-11 (`4 pass / 0 fail / 0 partial`)
- **Worker-coding execution isolation phase 1 scaffold:** landed on 2026-03-11 (isolated workspace manifests behind feature flag; main execution path unchanged)
- **Worker-coding execution isolation phase 2 shadow execution:** landed on 2026-03-11 (`delegate + static checks + verification` can run in isolated workspace without touching main workspace)
- **Worker-coding execution isolation phase 3 promotion gate:** landed on 2026-03-11 (promotion preflight and explicit `promote` mode added; main workspace writes remain opt-in)
- **Worker-coding live shadow-mode validation:** confirmed on 2026-03-12 (step-level isolation evidence remains good; orchestrator finalization now fails closed instead of leaving runs stuck in `running`)
- **Worker-coding live shadow-mode debug cohort:** `PASS` on 2026-03-11 (`2 pass / 0 fail / 0 partial` after runner terminal-state inference hardening)
- **Worker-coding live shadow-mode full cohort:** `PASS` on 2026-03-11 (`4 pass / 0 fail / 0 partial`)
- **Orchestrator result-consumer pending recovery:** landed on 2026-03-12 (`XAUTOCLAIM`-based stale result recovery + fail-loud loop logging)
- **Post-restart live shadow-mode debug cohort:** `PASS` on 2026-03-11T23:20Z (`2 pass / 0 fail / 0 partial`, pending backlog cleared)
- **Stable local OpenCode/Ollama execution lane:** `PASS` on 2026-03-12 (`opencode + ollama/glm-4.7-flash:latest` live probe and cohort evidence now usable as the known-good coding lane)
- **Authoritative stable-lane debug cohort:** `PASS` on 2026-03-12 (`2 pass / 0 fail / 0 partial`)
- **Authoritative stable-lane full four-case cohort:** `PASS` on 2026-03-12 (`4 pass / 0 fail / 0 partial`)

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

---

## Active Design Authority

**Active milestone:** M10 is in active development (Phase 2).

Current governance state:
- M6 evidence has been strengthened by compressed accelerated validation and is ready for next-stage review.
- M7 Phase A design, scripts, and config package are complete.
- M9 coding execution hardening is CLOSED.
- M10 Phase 1 (Execution Promotion Engine) is CLOSED. Atomic patch application, baseline drift detection, and rollback journaling are landed and validated.
- M10 Phase 2 (Dynamic Routing Enforcement) is IN PROGRESS. `router_mode` is now set to `dynamic_routing_enforced` in production config.
- Phase B limited enforced entry is explicitly approved for the narrow `coding_team_v0 / webapp_crm / fe_led` cohort.
- Canary A validation for enforced mode PASSED (3/3).
- Canary B validation for enforced mode PASSED (FE + BE queued in parallel with disjoint `target_paths`, routing audit log persisted, and `policy_evaluation` waterfall stage recorded).
- Canary C validation for enforced mode PASSED (full `fe_safe` DAG completed with `qa_verify`, `release_pack`, `GO` go/no-go, and release-pack artifacts).
- Observability correlation canary PASSED (`routing_decision_log`, `policy_evaluation`, and derived `branch_completion_*` stages now query cleanly by the same parent `run_id`).
- M10 quantified load-test specification recorded with fixed task mix, latency assumptions, `XAUTOCLAIM` parameters, and pass/fail thresholds.
- Stable mainline coding execution is now restored on `opencode + local ollama/glm-4.7-flash:latest`.
- OpenCode/Qwen runtime triage remains explicit and isolated: the direct DashScope compatible-mode endpoint is valid, but current `opencode` built-in `alibaba-coding-plan` provider path still does not accept the same credential path for coding execution.
- next-stage mainline execution has delivered a passing release gate, completed startup-path hardening, completed brain gateway contract hardening, completed worker-coder decomposition, and restored truthful worker-coding cohort readiness evidence.

Governing documents:

| Document | Path |
|----------|------|
| Governance v3 | `docs/01_design/system/260309/260309_1048/OpenClaw_Execution_Governance_Scope_Control_v3.md` |
| Architect Contract | `docs/01_design/system/260307/Architect_Engineer_Role_Contract.md` |
| M8 Engineering Task List v1 | `docs/01_design/system/260309/260309_M8/OpenClaw_Nexus_Engineering_Task_List_M8_v1.md` |
| M9 Engineering Task List v1 | `docs/01_design/system/260310/OpenClaw_Nexus_Engineering_Task_List_M9_v1.md` |
| M10 Draft Task List | `docs/03_feature_development/2026-03-12_m10_draft_tasklist.md` |
| M10 Load Test Spec | `docs/03_feature_development/2026-03-12_m10_load_test_spec.md` |
| M10 Phase B Sign-off | `docs/governance/2026-03-12_m10_phase_b_limited_enforced_signoff.md` |

---

## M6 Status - GO_LIMITED_EXPOSURE

Upgraded from `STAY_GATED` via M8 Go/No-Go approval on 2026-03-09.

**Active production config:** `configs/production_parallel_rollout.json`
- `master_enabled: true`
- `dynamic_routing_enabled: true`
- `router_mode: dynamic_routing_enforced`

Current interpretation:
- M6 production gate remains active as the safety baseline.
- Dynamic routing has now advanced from Phase A advisory-only to Phase B limited enforced execution for the narrow approved cohort.
- 2026-03-09 accelerated validation remains the advisory evidence base that justified controlled enablement.
- Current focus is no longer canary admission proof; it is executing quantified load validation and then targeted failure injection without widening cohort scope.
- `T-32` is presently split into two tracks: mainline workflow-quality/load validation under the known-good `stable_local_lane`, and separate remediation of `Qwen on opencode` runtime/provider integration.

Key artifacts:
- Runtime gate: `orchestrator/src/domain/parallel_rollout_gate.js`
- Eligibility policy: `configs/parallel_exposure_policy.json`
- Circuit breaker: `orchestrator/src/domain/circuit_breaker_service.js`
- Accelerated validation plan: `docs/03_feature_development/2026-03-09_30min_accelerated_validation_plan.md`
- Accelerated validation report: `orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json`
- Go/No-Go conclusion: `docs/governance/m6_accelerated_validation_go_no_go_2026-03-09.md`

---

## M7 Status - CLOSED (ACCEPTED WITH DEVIATION)

Original M7 implementation work remains complete and accepted with deviation. Live-trial evidence requirements were later closed in M8. Dynamic routing infrastructure is implemented, but production enablement is still deferred.

Current governance state:
- Original M7 implementation remains closed and accepted.
- Post-M8 controlled enablement plan has been integrated into design and runtime.
- Phase A advisory evidence collection is complete enough for the current limited cohort.
- Phase B limited enforced execution is now explicitly approved and active for the narrow cohort through M10 governance.
- Next governance step is not broader rollout; it is executing `T-32` against the recorded M10 load-test spec, then `T-33` failure injection, while keeping cohort scope fixed.
- Today's runtime finding does not reopen M7/M10 governance. The mainline conclusion is that `stable_local_lane` is now a working execution baseline, while `Qwen via current opencode provider chain` remains unresolved as a parallel triage item.

| Workstream | Status | Key Files |
|------------|--------|-----------|
| WS-27 Design Delta | DONE | `OpenClaw_Nexus_Design_Document_v4.md` |
| WS-28 Brain Router Classification | DONE | `src/vnext/brain_router_classifier.js`, `contracts/routing_decision.schema.json` |
| WS-29 Adaptive Runtime Integration | DONE | `src/domain/parallel_rollout_gate.js` |
| WS-30 Observability / Auditability | DONE | `routing_audit_log.js`, `waterfall_trace_service.js`, `routing_evaluation_report.js` |
| WS-31 Limited Dynamic Exposure | DONE (deviation accepted) | Live trial executed in M8 staging and later reinforced by accelerated validation |
| WS-32 Closure Package | DONE | `docs/governance/m7_go_no_go.md`, `m7_closure_note.md` |

---

## M8 Status - CLOSED

| Phase | Status | Key Output |
|-------|--------|------------|
| Phase 0: Technical Debt | DONE | `brain` pytest passed; workflow engine budget held |
| Phase 1: Live Trial | DONE | Staging/live routing validation completed |
| Phase 2: Evidence Review | DONE | Counterfactual report, drill validation, classifier availability evidence |
| Phase 3: Closure / Decisions | DONE | M6 approved for limited exposure; M7 stayed on hold in production |

Governance:
- `docs/governance/m8_go_no_go.md`

---

## M9 Status - CLOSED

Current interpretation:
- M9 is focused on `worker-coder` execution quality, not on routing/governance expansion.
- The core coding hardening slice is now landed in code and validated locally plus through passing live cohort evidence.
- Scope now covers context grounding, write-scope enforcement, scoped file-delta capture, fast static checks, verification execution, coding failure memory, hardened execution contracts, bounded retry controls, durable memory write-back, release-pack evidence visibility, and truthful cohort result attribution.
- Runtime/compose preflight validation is now added and passing for orchestrator startup-critical config.
- `C-BUG-01` is resolved and fully proven by the 4/4 cohort run post-result-consumer recovery.

Completed so far:
- Task-scoped context packet generation for coding steps
- Lightweight repo map generation and artifact persistence
- Context packet / repo map injection into coding execution payloads
- `target_paths`-based write guardrails in `worker-coder`
- Scoped snapshot-based `files_changed` recovery without full-repo `git status`
- Fast static checks before success return (`node --check`, JSON parse, `python -m py_compile`)
- `verification_command` execution support with structured verification logs
- Append-only coding failure memory under run-scoped artifacts
- Coding failure memory copied into orchestrator durable memory roots on workflow closure
- Hardened execution contract block injected into coding prompts to constrain scope/output format
- Bounded auto-fix loop controls (`max_attempts`, `same_error_repeat_limit`, `wall_clock_timeout_s`) in `worker-coder`
- Terminal `final_failure_summary` emitted for bounded retry failures
- Orchestrator-side payload productization for `verification_command` and retry controls
- Release-pack evidence contract extended with verification, retry, final-failure, prompt-contract, and failure-memory visibility
- Default coding runtime compatibility updated toward `provider=opencode`, `model=alibaba-coding-plan/qwen3-coder-plus`, while preserving legacy `qwen3-coder-plus-2025-07-22` normalization in the adapter
- Local canary coverage for M9 coding guardrails (`canary:m9_coding_guardrails`)
- Worker-level auto-fix retry canary passes in sandbox via inline mocked provider (`worker-coder canary:m9_autofix_retry`)
- `brain` fact polling is decoupled from direct PostgreSQL access through orchestrator HTTP gateway endpoints
- Live stack validation completed for the new M9/brain boundary path after container refresh and config mount fix
- full workflow-level live validation is now passing with contract-valid deterministic provider fixtures for PM/architect/implementation/release steps
- full four-case worker-coding cohort is now passing with truthful verification-tier evidence (`4/4 pass`), and the 2026-03-12 authoritative rerun is now locked to `execution_lane=stable_local_lane`, `model_provider=opencode`, `model_name=ollama/glm-4.7-flash:latest`

Current closeout blocker:
- None. M9 is officially closed.

---

## Current Verification Status

```text
node --test orchestrator/test/*.test.js                       -> 127 / 127 PASS  (2026-03-09)
pytest brain/tests/                                           ->  11 /  11 PASS  (2026-03-09)
run_m7_dynamic_routing_trial.js                               -> PASS preflight / governed evaluation
run_m7_dynamic_routing_trial.js --drill-unavailable           -> PASS forced_sequential fallback
live_local_llm_dispatcher (deepseek-r1:32b, real local call)  -> PASS (2026-03-09)
live_validate_vnext_runtime.js                                -> PASS (2026-03-10)
compressed_validation_report                                  -> PASS 89 routing / 71 parallel / 18 sequential
canary_m7_phase_a_advisory.js                                 -> PASS 4 / 4 (2026-03-10)
run_m7_dynamic_routing_trial.js (Phase A enabled)             -> PASS live_trial, cohort_cases=10, agreement=0.94
canary_m9_coding_guardrails.js                                -> PASS local end-to-end evidence contract / retry visibility (2026-03-10)
worker-coder canary:m9_autofix_retry                          -> PASS inline mocked first-fail / retry-success path (2026-03-10)
pytest brain/tests/test_supervisor_routing.py                 -> 12 / 12 PASS after HTTP fact-gateway decoupling (2026-03-10)
validate:live_vnext_runtime                                   -> PASS after orchestrator/brain container refresh (2026-03-10)
validate:config_preflight                                     -> PASS (2026-03-11)
validate:next_stage_release_gate -- --skip-live               -> PASS (2026-03-11)
validate:next_stage_release_gate                              -> PASS (2026-03-11)
validate:runtime_boot_sources                                 -> PASS (2026-03-11)
curl /brain/facts/latest + synthetic DB fact                  -> PASS live HTTP gateway lookup (2026-03-10)
POST brain /run                                               -> PASS on refreshed brain container (2026-03-10)
validate:live_m9_workflow                                     -> PASS succeeded workflow + release-pack evidence check (2026-03-11)
brain_gateway.integration.test.js                             -> PASS (2026-03-11)
worker-coder test:adapter + prompt/retry/failure tests        -> PASS (2026-03-11)
worker-coder/tests/scoped_delta.test.js                       -> PASS after single-file fallback hardening (2026-03-11)
worker-coder/tests/isolation_workspace.test.js                -> PASS after isolation scaffold landing (2026-03-11)
worker-coder/tests/isolation_delegate_shadow.test.js          -> PASS (shadow-mode failure leaves main workspace unchanged) (2026-03-11)
worker-coder/tests/promotion_workspace.test.js                -> PASS (`shadow` no-apply + `promote` apply + out-of-scope block) (2026-03-11)
validate:worker_coding_cohort_execute (debug FE+BE)           -> PASS 2 / 2 (2026-03-11)
validate:worker_coding_cohort_execute (debug FE+BE, shadow mode) -> PASS 2 / 2 (2026-03-11)
validate:worker_coding_cohort_execute (BE only)               -> PASS 1 / 1 (2026-03-11)
validate:worker_coding_cohort_execute (full four-case cohort) -> PASS 4 / 4 (2026-03-11)
validate:worker_coding_cohort_execute (full four-case cohort, shadow mode) -> PASS 4 / 4 (2026-03-11)
validate:worker_coding_cohort_execute (full four-case cohort, promote mode) -> PASS 4 / 4 (2026-03-12)
worker-coder/tests/promotion_workspace.test.js (atomic patch apply and conflict detector) -> PASS 1 / 1 (2026-03-12)
worker-coder unified diff generation for `promotion_request.diff` -> PASS (2026-03-12)
canary:m10_phase_b_enforced                                  -> PASS 3 / 3 (2026-03-12)
canary:m10_phase_b_parallel_isolation                        -> PASS (enforced FE+BE parallel queue + disjoint target_paths) (2026-03-12)
canary:m10_observability_correlation                        -> PASS (run_id-correlated routing + waterfall + branch timelines) (2026-03-12)
```

---

## Blocking Points

Active P0 blocker:

| Block | Status | Resolution Path |
|-------|--------|-----------------|
| Local orchestrator process still running old code / unknown startup env | RESOLVED | Migrated fully to containerized `docker-compose` stack. Fixed git lock bottlenecks in worker and mapped Qwen models correctly. Phase A live validation now generates `dynamic_routing_advisory_only` records successfully. |

Recently cleared:

| Block | Status | Resolution |
|-------|--------|------------|
| Containerized orchestrator startup gap | RESOLVED | `infra/docker-compose.yml` updated with `contracts` mount |
| Live runtime validation script drift | RESOLVED | Approval trigger and workflow assertions updated; timeout logic fixed to handle `succeeded` status correctly. |
| Long-cycle evidence dependency | RESOLVED (compressed substitute) | 30-minute accelerated validation package produced |
| Phase A runtime semantics mismatch | RESOLVED | advisory/enforced router_mode and cohort-first runtime behavior implemented |

---

## TODO (Ordered by Priority)

### P0

| # | Task | Owner | Status |
|---|------|-------|--------|
| T-A1 | Safely restart local orchestrator onto updated M7 advisory runtime | You | DONE |
| T-A2 | Re-run Phase A live validation and confirm `dynamic_routing_advisory_only` appears in logs | You | DONE |
| T-A3 | Resume advisory-only evidence collection after runtime confirmation | You | DONE |
| T-A4 | Keep cohort narrow and rollback path ready | You | IN EFFECT |

### P1

| # | Task | Dependency | Goal |
|---|------|------------|------|
| T-B1 | Productize accelerated evidence workflow | T-A1 | Make Phase A evidence collection repeatable, not one-off |
| T-B2 | Decide whether Phase B should remain blocked or enter review | T-A2 | DONE - limited enforced entry approved on 2026-03-12 |
| T-B3 | Draft `brain/` API boundary decoupling design | None | DONE for current endpoint surface; next step is broader endpoint expansion only if needed |
| T-B4 | Extend M9 validation from local canary to worker/live integration | None | DONE in local mocked-worker scope; next step is full live runtime confidence |

### P2

| # | Task | Dependency |
|---|------|------------|
| T-C1 | Consider expanding cohort beyond current guarded FE-led scope | T-B2 |
| T-C2 | Evaluate whether `model_tier` should influence runtime execution policy | T-B2 |
| T-C3 | Expand M9 evidence and memory contracts into durable orchestration memory | T-B4 |

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

---

## Key Artifact Index

| Artifact | Path | Purpose |
|----------|------|---------|
| Root rollout config | `configs/production_parallel_rollout.json` | Prepared runtime gate state for local/root-based startup |
| Root M7 cohort config | `configs/m7_exposure_cohorts.json` | Prepared Phase A cohort restriction for local/root-based startup |
| Orchestrator rollout config | `orchestrator/configs/production_parallel_rollout.json` | Compatibility copy; should match root governance config |
| Orchestrator M7 cohort config | `orchestrator/configs/m7_exposure_cohorts.json` | Compatibility copy; should match root governance config |
| Exposure policy | `configs/parallel_exposure_policy.json` | Allowed M6 exposure cohorts |
| M6 accelerated validation plan | `docs/03_feature_development/2026-03-09_30min_accelerated_validation_plan.md` | Compressed evidence plan |
| M6 accelerated validation report | `orchestrator/artifacts/m6_trial/accelerated_validation_report_30m.json` | Measured compressed evidence |
| M6 accelerated validation Go/No-Go | `docs/governance/m6_accelerated_validation_go_no_go_2026-03-09.md` | Formal next-stage recommendation |
| Post-M8 M7 controlled enablement plan | `docs/governance/post_m8_m7_controlled_enablement_plan_2026-03-10.md` | Proposed M7 production enablement path |
| Phase A advisory preflight | `orchestrator/artifacts/m7_phase_a/phase_a_advisory_preflight.json` | Pre-enable readiness artifact |
| Phase A advisory canary | `orchestrator/artifacts/canary/m7_phase_a_advisory/canary_m7_phase_a_advisory.json` | Runtime behavior verification for advisory mode |
| Phase A enabled trial result | `orchestrator/artifacts/m7_trial/preflight_result_20260310_phase_a_enabled.json` | Post-enable cohort-scoped trial evidence |
| Phase A initial observation | `docs/03_feature_development/2026-03-10_m7_phase_a_initial_observation.md` | First evidence-collection snapshot after enablement |
| Phase A initial observation artifact | `orchestrator/artifacts/m7_phase_a/phase_a_initial_observation_20260310.json` | Initial advisory-only observation report |
| QA accelerated validation task list | `docs/03_feature_development/2026-03-09_qa_accelerated_validation_tasklist.md` | Execution checklist |
| QA test summary | `docs/03_feature_development/2026-03-09_qa_test_summary.md` | Consolidated verification evidence |
| Live runtime report | `orchestrator/artifacts/canary/live_vnext_runtime/live_vnext_runtime_report.json` | Online runtime validation |
| Local LLM dispatch evidence | `orchestrator/artifacts/canary/live_local_llm_dispatcher/live_local_llm_dispatcher_20260309.json` | Real local model invocation evidence |

---

## Code and Configuration Changes Since Last Snapshot

### Core Runtime and Validation
- `orchestrator/scripts/live_validate_vnext_runtime.js` updated to match current approval/workflow entry behavior.
- `orchestrator/scripts/inject_live_traffic.js` parameterized for compressed validation execution.
- `orchestrator/scripts/generate_accelerated_validation_report.js` added to summarize routing and execution evidence.
- `orchestrator/src/domain/parallel_rollout_gate.js` extended for `dynamic_routing_advisory`, `dynamic_routing_enforced`, and cohort-first gating.
- `orchestrator/scripts/preflight_m7_phase_a_advisory.js` added for Phase A readiness checks.
- `orchestrator/scripts/canary_m7_phase_a_advisory.js` added for advisory-only runtime verification.
- `orchestrator/scripts/set_m7_phase_a_advisory.js` and `rollback_m7_phase_a_advisory.js` added for reversible config control.
- `orchestrator/scripts/generate_accelerated_validation_report.js` updated to anchor report windows to latest database timestamps.
- `orchestrator/src/domain/parallel_rollout_gate.js` and `orchestrator/src/domain/routing_audit_log.js` updated with config path fallback so `node src/index.js` local startup can resolve root configs instead of relying on `/workspace`.
- `orchestrator/scripts/validate_next_stage_release_gate.js` added as a single next-stage release-gate entrypoint with machine-readable summary output.
- `orchestrator/scripts/validate_runtime_boot_sources.js` added to verify startup-path config sources and compose mounts.
- `orchestrator/src/vnext/brain_gateway.js` added to isolate brain gateway handlers from `index.js`.

### Infrastructure
- `infra/docker-compose.yml` updated so containerized `orchestrator` mounts `contracts` correctly.
- `infra/docker-compose.yml` updated so governance runtime files mount from root `configs/`.
- `infra/docker-compose.yml` updated to inject `ALIBABA_CODING_PLAN_API_KEY` into the coding runtime for provider triage.

### Coding / M9
- `orchestrator/src/domain/repo_context_service.js` added for task-scoped context packet and repo map generation.
- `orchestrator/src/domain/workflow_step_builder.js` updated to inject coding context and persist `context_packet` / `repo_map` into execution requests.
- `orchestrator/src/domain/workflow_step_artifacts.js` updated to write context artifacts under release packs.
- `orchestrator/src/domain/workflow_artifact_pack.js` updated to archive/index context artifacts.
- `worker-coder/coding_service.js` updated with `target_paths` write guardrails, scoped snapshot diff recovery, and fast static checks.
- `worker-coder/worker.js` updated to pass `target_paths` into patch/delegate operations.
- `worker-coder/worker.js` import path fixed after decomposition so live container startup matches current source layout.
- `worker-coder/scope_guard.js` extracted from `coding_service.js` and covered by targeted tests.
- `worker-coder/artifact_scaffold.js` extracted from `coding_service.js` to isolate scaffold/template/repair behavior with targeted tests.
- `worker-coder/scoped_delta.js` extracted from `coding_service.js` to isolate scoped snapshot, diff summary, and deterministic implementation-delta recovery.
- `worker-coder/static_checks.js` extracted from `coding_service.js` to isolate fast static-check execution, severity shaping, and timeout clamping.
- `configs/runtime/runtime_defaults.json`, `infra/docker-compose.yml`, and `orchestrator/src/index.js` aligned to `opencode + qwen3-coder-plus-2025-07-22`.
- `docs/03_feature_development/2026-03-10_opencode_qwen_runtime_note.md` added as the current coding runtime note.
- `worker-coder/prompt_contract.js`, `worker-coder/verification_runner.js`, `worker-coder/retry_policy.js`, and `worker-coder/failure_memory.js` extracted from `coding_service.js` with targeted tests.
- `worker-coder/tests/artifact_scaffold.test.js` added to verify scaffold creation and repair logic.
- `worker-coder/tests/scoped_delta.test.js` added to verify scoped snapshot, diff accounting, and fallback stub recovery.
- `worker-coder/tests/static_checks.test.js` added to verify static-check execution and timeout guard behavior.
- `worker-coder/task_lifecycle.js` added to enforce single-finalization semantics across success, failure, timeout, and ack paths.
- `worker-coder/tests/task_lifecycle.test.js` added to verify timeout cannot double-write result/fact/ack after late completion.
- `worker-coder/git_side_effects.js` added to replace shell-based auto-commit with structured git execution and artifacted outcomes.
- `worker-coder/tests/git_side_effects.test.js` added to verify structured git arguments and auto-commit evidence output.
- `worker-coder/task_contract.js` added to normalize `task_class` / `context_envelope` metadata and derive `failure_attribution`.
- `worker-coder/tests/task_contract.test.js` added to verify task-contract normalization and failure-attribution mapping.
- `worker-coder/coding_service.js`, `worker-coder/worker.js`, and `worker-coder/failure_memory.js` updated so `coding.delegate` can carry compatible task-contract metadata and persist it in diagnostics/failure memory.
- `docs/03_feature_development/2026-03-11_worker_coding_task_contract_note.md` added as the first `WC-NEXT-01` contract landing note.
- `orchestrator/contracts/worker_coding_task_contract.schema.json` and `orchestrator/contracts/worker_coding_beta_template_registry.schema.json` added as authority schemas for next-stage worker-coding governance.
- `configs/registry/worker_coding_beta_templates.json` added with initial templates for `fe_create`, `fe_modify`, `be_create`, and `bug_fix`.
- `orchestrator/scripts/validate_worker_coding_contract.js` and `npm.cmd --prefix orchestrator run validate:worker_coding_contract` added and passing.
- `orchestrator/src/worker_coding_templates.js` added so worker-coding beta templates are loaded at orchestration time, not inferred inside the worker.
- `orchestrator/src/domain/workflow_step_builder.js` updated to inject template defaults into `coding.delegate` payloads before execution.
- `orchestrator/test/worker_coding_templates.test.js` added to verify template-driven `task_class` / `context_envelope` injection.
- `docs/03_feature_development/2026-03-11_worker_coding_cohort_task_matrix.md` added to define the first multi-class worker-coding validation cohort (`fe_create`, `fe_modify`, `be_create`, `bug_fix`).
- `orchestrator/contracts/worker_coding_cohort_result.schema.json` added as the authority schema for cohort validation result artifacts.
- `docs/03_feature_development/2026-03-11_worker_coding_cohort_result_format.md` added to define the machine-readable cohort result format.
- `orchestrator/scripts/validate_worker_coding_cohort_result_format.js` and `npm.cmd --prefix orchestrator run validate:worker_coding_cohort_result` added and passing.
- `orchestrator/contracts/worker_coding_cohort_plan.schema.json` added as the authority schema for cohort execution plans.
- `configs/registry/worker_coding_cohort_plan_v1.json` added as the first machine-readable worker-coding cohort plan.
- `orchestrator/scripts/validate_worker_coding_cohort_plan.js` and `npm.cmd --prefix orchestrator run validate:worker_coding_cohort_plan` added and passing.
- `shared/worker_coding_contract.mjs` added as the shared authority source for allowed worker-coding task classes.
- `orchestrator/src/worker_coding_templates.js` hardened so `beta_template_id` and `task_class` mismatch now fails fast instead of silently polluting cohort data.
- `orchestrator/scripts/validate_worker_coding_contract.js` and `orchestrator/scripts/validate_worker_coding_cohort_plan.js` updated to consume the shared task-class authority rather than duplicating local enums.
- `orchestrator/scripts/run_worker_coding_cohort.js` and `npm.cmd --prefix orchestrator run validate:worker_coding_cohort_execute` added to execute the first controlled worker-coding cohort and emit machine-readable cohort result artifacts.
- first cohort artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-17-51-917Z/worker_coding_cohort_result.json`; observed result was `4 partial / 0 fail / 0 pass`, with all four runs closing workflow successfully but only achieving `syntax_check` against higher declared verification tiers.
- `orchestrator/src/domain/workflow_step_builder.js` now translates template-declared verification tiers into structured `verification_plan` payloads, resolving package-script-backed tiers where available and preserving unresolved tiers explicitly.
- `worker-coder/verification_runner.js`, `worker-coder/coding_service.js`, `worker-coder/worker.js`, and `worker-coder/prompt_contract.js` now support ordered verification-plan execution while remaining compatible with legacy single-command verification.
- second cohort artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-26-43-490Z/worker_coding_cohort_result.json`; result remained `4 partial / 0 fail / 0 pass`, confirming the current blocker is verification-depth availability rather than workflow instability.
- `configs/registry/worker_coding_task_classes.json` is now the container-safe authority source for worker-coding task classes; orchestrator and worker loaders now resolve from config-visible paths instead of container-fragile relative imports.
- `sandbox/crm_site/package.json` and `sandbox/crm_site/scripts/verify_crm_site.mjs` added as the first repo-aware verification source for cohort validation (`lint`, `typecheck`, `test`, `build` all locally pass).
- latest cohort artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-57-01-157Z/worker_coding_cohort_result.json`; result is now `4 fail / 0 partial / 0 pass`, which is a more truthful signal after live verification enforcement:
  - `fe_create`, `fe_modify`, `bug_fix`: `verification_failure`
  - `be_create`: `coding_logic_failure`
- `infra/docker-compose.yml` updated so `worker-coder` now mounts live repo source into `/app`, removing stale-image runtime drift during live validation.
- `orchestrator/src/coding_executor.js` updated so `verification_plan`, `task_class`, `beta_template_id`, and `context_envelope` are preserved in coding-executor requests.
- `orchestrator/scripts/run_worker_coding_cohort.js` updated to isolate non-focused implementation steps for debug cohorts, classify scope-guard failures correctly, and treat achieved verification supersets as pass.
- `worker-coder/scoped_delta.js` and `worker-coder/coding_service.js` updated so single-file targets no longer fabricate out-of-scope fallback stub files when no implementation delta is produced.
- `worker-coder/isolation_workspace.js` landed to create isolated workspace manifests and scoped shadow copies behind `CODER_ISOLATION_MODE=scaffold|shadow`; current runtime remains main-workspace execution until promotion logic is implemented.
- `worker-coder/coding_service.js`, `worker-coder/coding_executor_runtime.js`, and `worker-coder/adapters/opencode_adapter.js` now support shadow-mode isolated execution so delegate/static-check/verification failures do not mutate the main workspace before promotion support exists.
- `worker-coder/promotion_workspace.js` now adds promotion preflight plus explicit `promote` mode, validating changed files against `target_paths` before any main-workspace write is applied.
- live shadow-mode debug cohort evidence shows `isolation_mode=shadow` and `promotion.applied=false` in worker step diagnostics while artifact roots remain stable.
- `orchestrator/src/workflow_engine.js` now fails workflow finalization closed when artifact-pack generation throws, preventing the prior `all steps succeeded but run stayed running` state from persisting silently.
- `orchestrator/test/workflow_finalization.test.js` added to lock both terminal success closeout and finalization-failure closeout semantics.
- `orchestrator/scripts/run_worker_coding_cohort.js` still contains temporary terminal-state inference for reporting continuity, but that path is no longer the desired steady-state evidence source after workflow-engine hardening.
- `orchestrator/src/vnext/result_consumer.js` now reclaims stale pending result-stream messages before reading new ones and logs per-message failures instead of silently sleeping the whole loop.
- `orchestrator/src/data/task_repository.js` now only marks tasks `running` when they are still `queued` or `waiting_approval`, preventing recovered stale `claimed` messages from regressing terminal task state.
- `orchestrator/test/result_consumer_recovery.test.js` added to lock stale-result recovery and fresh-result processing behavior.
- shadow-mode debug cohort artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T14-02-24-449Z/worker_coding_cohort_result.json`; result is `2 pass / 0 fail / 0 partial`.
- shadow-mode full cohort artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T14-14-49-306Z/worker_coding_cohort_result.json`; result is `4 pass / 0 fail / 0 partial`, showing the recovered baseline is preserved under isolated `shadow` execution.
- final recovery artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T12-56-17-290Z/worker_coding_cohort_result.json`; result is now `4 pass / 0 fail / 0 partial`, restoring worker-coding cohort readiness evidence under truthful verification enforcement.
- post-restart debug shadow artifact recorded at `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T23-20-45-381Z/worker_coding_cohort_result.json`; result is `2 pass / 0 fail / 0 partial`, confirming pending backlog recovery after orchestrator restart.
- `worker-coder/tests/startup_smoke.test.js` added to check worker entrypoint import wiring and key module syntax guards before container startup.
- `worker-coder/adapters/opencode_adapter.js` live-validation mock outputs aligned with current implementation-step handoff and schema governance so full live gate now passes.
- `worker-coder/adapters/opencode_adapter.js` now normalizes legacy `qwen3-coder-plus-2025-07-22` references to `alibaba-coding-plan/qwen3-coder-plus` for opencode compatibility probing.
- `worker-coder/tests/opencode_adapter.test.js` updated to lock the new Qwen normalization path.
- `worker-coder/scripts/qwen_compatible_probe.mjs` added to directly validate `QWEN_BASE_URL + QWEN_API_KEY` against the OpenAI-compatible DashScope endpoint from the live worker container.
- live runtime triage completed on 2026-03-12 with these authoritative outcomes:
  - direct `QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1` call with `model=qwen3-coder-plus-2025-07-22` -> `PASS (HTTP 200)`
  - `opencode models alibaba-coding-plan` after `ALIBABA_CODING_PLAN_API_KEY` injection -> provider visible with `qwen3-coder-plus`
  - `opencode run ... --model alibaba-coding-plan/qwen3-coder-plus` -> `FAIL (invalid access token or token expired)`
  - conclusion: the current DashScope credential is valid for direct compatible-mode calls but does not authenticate successfully through `opencode`'s built-in `alibaba-coding-plan` provider path

### Governance and Documentation
- Accelerated validation plan, QA summary, Go/No-Go package, and post-M8 M7 controlled enablement plan added.
- `docs/03_feature_development/2026-03-11_validation_gate_runbook.md`, `2026-03-11_runtime_startup_path_note.md`, and `2026-03-11_brain_gateway_contract_note.md` added for current mainline execution governance.

