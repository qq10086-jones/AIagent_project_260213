# Feature Progress - Latest Snapshot

**Last updated:** 2026-03-11 (M9 closeout validation follow-up)
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
- **Phase A live runtime validation after enablement:** `PASS` before observability-gap investigation
- **Current coding focus:** M9 closeout tasks after core guardrails landed
- **M9 runtime config preflight:** `PASS` on 2026-03-11
- **M9 real live workflow validation:** `PASS` on 2026-03-11

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
| M9 | Coding Precision & Sandbox Guardrails | **GO WITH CONDITIONS** | 2026-03-11 |

---

## Active Design Authority

**Active milestone:** M9 is in closeout.

Current governance state:
- M6 evidence has been strengthened by compressed accelerated validation and is ready for next-stage review.
- M7 Phase A design, scripts, and config package are complete.
- M9 coding execution hardening is landed in code, and closeout work is focused on validation depth plus structural cleanup.

Governing documents:

| Document | Path |
|----------|------|
| Governance v3 | `docs/01_design/system/260309/260309_1048/OpenClaw_Execution_Governance_Scope_Control_v3.md` |
| Architect Contract | `docs/01_design/system/260307/Architect_Engineer_Role_Contract.md` |
| M8 Engineering Task List v1 | `docs/01_design/system/260309/260309_M8/OpenClaw_Nexus_Engineering_Task_List_M8_v1.md` |
| M9 Engineering Task List v1 | `docs/01_design/system/260310/OpenClaw_Nexus_Engineering_Task_List_M9_v1.md` |

---

## M6 Status - GO_LIMITED_EXPOSURE

Upgraded from `STAY_GATED` via M8 Go/No-Go approval on 2026-03-09.

**Active production config:** `orchestrator/configs/production_parallel_rollout.json`
- `master_enabled: true`
- `dynamic_routing_enabled: true` in prepared runtime config
- `router_mode: dynamic_routing_advisory` in prepared runtime config

Current interpretation:
- M6 production gate is active.
- Phase A advisory-only runtime package is prepared and validated offline / canary-level.
- 2026-03-09 accelerated validation produced enough compressed evidence to enter Phase A without waiting for a 1-2 week natural observation window.
- Live advisory-only observability is not yet fully confirmed because the currently running local orchestrator process still needs a controlled restart onto the updated code.

Key artifacts:
- Runtime gate: `orchestrator/src/domain/parallel_rollout_gate.js`
- Eligibility policy: `orchestrator/configs/parallel_exposure_policy.json`
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
- Phase A activation package exists, but live observation is paused until the local orchestrator process is restarted on the new code path.
- Next governance step is not Phase B review yet; it is completing that restart and confirming `dynamic_routing_advisory_only` records appear in `routing_decision_log`.

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

## M9 Status - GO WITH CONDITIONS

Current interpretation:
- M9 is focused on `worker-coder` execution quality, not on routing/governance expansion.
- The core coding hardening slice is now landed in code and validated locally with passing canaries/tests.
- Scope now covers context grounding, write-scope enforcement, scoped file-delta capture, fast static checks, verification execution, coding failure memory, hardened execution contracts, bounded retry controls, durable memory write-back, and release-pack evidence visibility.
- Runtime/compose preflight validation is now added and passing for orchestrator startup-critical config.

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
- Default coding runtime aligned to `provider=opencode`, `model=qwen3-coder-plus-2025-07-22`
- Local canary coverage for M9 coding guardrails (`canary:m9_coding_guardrails`)
- Worker-level auto-fix retry canary passes in sandbox via inline mocked provider (`worker-coder canary:m9_autofix_retry`)
- `brain` fact polling is decoupled from direct PostgreSQL access through orchestrator HTTP gateway endpoints
- Live stack validation completed for the new M9/brain boundary path after container refresh and config mount fix

Not yet done:
- broader M9 release-quality validation against richer end-to-end business scenarios
- typed brain gateway expansion beyond latest-fact lookup / event ingestion if future workflows need more surface area

Current closeout blocker:
- full workflow-level live validation is now passing with contract-valid deterministic provider fixtures for PM/architect/implementation/release steps
- remaining closeout work is structural follow-up, not live workflow correctness

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
curl /brain/facts/latest + synthetic DB fact                  -> PASS live HTTP gateway lookup (2026-03-10)
POST brain /run                                               -> PASS on refreshed brain container (2026-03-10)
validate:live_m9_workflow                                     -> PASS succeeded workflow + release-pack evidence check (2026-03-11)
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
| T-A3 | Resume advisory-only evidence collection after runtime confirmation | You | IN PROGRESS |
| T-A4 | Keep cohort narrow and rollback path ready | You | IN EFFECT |

### P1

| # | Task | Dependency | Goal |
|---|------|------------|------|
| T-B1 | Productize accelerated evidence workflow | T-A1 | Make Phase A evidence collection repeatable, not one-off |
| T-B2 | Decide whether Phase B should remain blocked or enter review | T-A2 | Formalize enforced-mode entry based on advisory evidence |
| T-B3 | Draft `brain/` API boundary decoupling design | None | Reduce direct DB coupling risk before broader expansion |
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
| R-NEW-04 | Local manual startup path bypasses intended config roots / env injection | High | Open |

---

## Key Artifact Index

| Artifact | Path | Purpose |
|----------|------|---------|
| Root rollout config | `configs/production_parallel_rollout.json` | Prepared runtime gate state for local/root-based startup |
| Root M7 cohort config | `configs/m7_exposure_cohorts.json` | Prepared Phase A cohort restriction for local/root-based startup |
| Orchestrator rollout config | `orchestrator/configs/production_parallel_rollout.json` | Container-oriented rollout config copy |
| Orchestrator M7 cohort config | `orchestrator/configs/m7_exposure_cohorts.json` | Container-oriented cohort config copy |
| Exposure policy | `orchestrator/configs/parallel_exposure_policy.json` | Allowed M6 exposure cohorts |
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

### Infrastructure
- `infra/docker-compose.yml` updated so containerized `orchestrator` mounts `contracts` correctly.

### Coding / M9
- `orchestrator/src/domain/repo_context_service.js` added for task-scoped context packet and repo map generation.
- `orchestrator/src/domain/workflow_step_builder.js` updated to inject coding context and persist `context_packet` / `repo_map` into execution requests.
- `orchestrator/src/domain/workflow_step_artifacts.js` updated to write context artifacts under release packs.
- `orchestrator/src/domain/workflow_artifact_pack.js` updated to archive/index context artifacts.
- `worker-coder/coding_service.js` updated with `target_paths` write guardrails, scoped snapshot diff recovery, and fast static checks.
- `worker-coder/worker.js` updated to pass `target_paths` into patch/delegate operations.
- `configs/runtime/runtime_defaults.json`, `infra/docker-compose.yml`, and `orchestrator/src/index.js` aligned to `opencode + qwen3-coder-plus-2025-07-22`.
- `docs/03_feature_development/2026-03-10_opencode_qwen_runtime_note.md` added as the current coding runtime note.

### Governance and Documentation
- Accelerated validation plan, QA summary, Go/No-Go package, and post-M8 M7 controlled enablement plan added.
