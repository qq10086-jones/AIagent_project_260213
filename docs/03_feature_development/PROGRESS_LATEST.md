# Feature Progress - Latest Snapshot

**Last updated:** 2026-03-15 (T-33 failure injection complete; worker-coder 19/19; orchestrator 150/150)
**Author:** PM / Architecture Review

---

## Execution Evidence

- **M10 load test (stable_local_lane):** `PASS / GO` — run a48a521e, 6/6 steps, 23 min (2026-03-15)
- **Post-M10 code quality hardening (2 audit rounds):** complete (2026-03-15)
- **T-33 failure injection suite:** complete, 7 failure paths covered (2026-03-15)

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

**No active milestone.** System is in stabilization / T-33 hardening phase.

Current governance state:
- `production_parallel_rollout.json`: `master_enabled=true`, `dynamic_routing_enabled=true`, `router_mode=dynamic_routing_enforced`
- Default execution lane: `stable_local_lane` (opencode + ollama/glm-4.7-flash:latest)
- `stable_codex_lane` defined in runtime config (provider=codex, model=codex-mini-latest) — deferred until OPENAI_API_KEY is available
- `stable_cloud_lane` defined but not end-to-end tested (Qwen alibaba-coding-plan auth unresolved)

---

## What Was Done (2026-03-13 → 2026-03-15)

### Round 1 — QA/Architect audit fixes (6 issues)

| Fix | Commit |
|-----|--------|
| `TASK_RUNNING_TIMEOUT_SEC = "900"` default overrode config 1800s → changed to `""` | `d365036` |
| `stream_batch_size` hardcoded 5 in worker.js → configurable (default 1) | `d365036` |
| `coding_service.js` inline dynamic import plumbing → extracted to `step_artifact_contract.js` | `d365036` |
| `salvageWorkflowArtifactFailure` private/untested → exported + `coding_service_salvage.test.js` (5 cases) | `e0dcab6` |
| `artifact_scaffold.js` CRM-specific heading check → generalized | `d365036` |
| `opencode_adapter.js` blocked `ollama/*` refs → removed; added `mapErrorCode` fast-path | `25b0860` |

### Round 2 — Deeper audit fixes (P0–P2)

| Fix | Commit |
|-----|--------|
| `step_artifact_contract.test.js`: 18 assertions, 6 step IDs, fresh-object guarantee | `e0dcab6` |
| `startup_smoke.test.js`: import wiring + file existence assertions | `e0dcab6` |
| `wall_clock_timeout_s` clamp boundary tests (3 cases); orchestrator 147→150 | `36cb2a7` |
| 16 `context_budget_*.json` golden files updated | `148c166` |
| `payloadToAdapterRequest` + `requiresScopedTargetPaths` exported + `coding_service_pure.test.js` (44 assertions) | `9c68bbf` |

### Runtime config changes

| Parameter | Before | After |
|-----------|--------|-------|
| `global_task_timeout_ms` | 900000 | 1800000 |
| `wall_clock_timeout_s_default` | 300 | 900 |
| `running_timeout_sec` | 900 (hardcoded) | 1800 (from config) |
| `stream_batch_size` | 5 (hardcoded) | 1 (configurable) |
| `runtimeByStep.pm_spec` | 120s | 360s |
| `runtimeByStep.arch_design` | 180s | 480s |
| `runtimeByStep.impl_fe` | 240s | 360s |

### T-33 Failure Injection

**worker-coder** (`delegate_failure_injection.test.js`, 7 tests):

| # | Scenario | Error Code | Commits |
|---|----------|-----------|---------|
| 1 | Provider exits 1 twice, max_attempts=2 | `E_DELEGATE_FAILED`, `terminal_reason=attempt_budget_exhausted` | `34c77cc` |
| 2 | Same error repeats, `same_error_repeat_limit=1` | stops after attempt 1, not max_attempts=3 | `34c77cc` |
| 3 | `impl_be` + empty `target_paths` | `E_UNAUTHORIZED_WRITE` pre-execution | `34c77cc` |
| 4 | `impl_be` + `.git/hooks` target path | `E_UNAUTHORIZED_WRITE` pre-execution | `34c77cc` |
| 5 | Command writes JS syntax error, exits 0 | `E_STATIC_CHECK_FAILED` | `34c77cc` |
| 6 | Command writes valid file, verification exits 1 | `E_VERIFICATION_FAILED` | `d712550` |
| 7 | `provider="gpt-99-turbo"` (unsupported) | `E_PROVIDER_UNAVAILABLE` pre-execution | `d712550` |

**orchestrator** (`canary_coding_team_e2e.js`, case 3 added):

| Case | Scenario | Assert |
|------|----------|--------|
| qa_failure | PM/arch/BE/FE succeed, `qa_verify` returns `status=failed` | `workflow=failed`, `release_pack` NOT dispatched (5 tasks total), `workflow.failed` event emitted |

### Infra additions
- `stable_codex_lane` defined in `runtime_defaults.json` (deferred)
- `validateCodexLane` added to `provider_preflight.js` + 2 preflight tests

---

## Current Verification Status

```
npm --prefix orchestrator test    → 150 / 150 PASS  (2026-03-15)
npm --prefix worker-coder test    →  19 /  19 PASS  (2026-03-15)
node scripts/canary_coding_team_e2e.js
  happy_path (6/6 steps)          → PASS
  be_failure (artifacts missing)  → PASS
  qa_failure (QA blocks release)  → PASS
M10 load test (stable_local_lane) → PASS / GO verdict (2026-03-15)
```

---

## Test Coverage Map

### worker-coder (19 tests)

| File | What it covers |
|------|---------------|
| `codex_adapter.test.js` | Codex CLI invocation, auth check, error mapping |
| `opencode_adapter.test.js` | OpenCode CLI invocation, ollama model refs, error mapping |
| `coding_executor_runtime.test.js` | Lane resolution, provider fallback, adapter dispatch |
| `verification_command.test.js` | Safe command validation, redaction |
| `prompt_contract.test.js` | Prompt contract artifact writing |
| `retry_policy.test.js` | Retry decision, same-error gate, final failure summary |
| `failure_memory.test.js` | Failure memory persistence |
| `scope_guard.test.js` | Target path validation, protected roots, scope check |
| `artifact_scaffold.test.js` | Stub creation, heading validation, handoff schemas |
| `scoped_delta.test.js` | Snapshot diff, filesChanged, delta recovery |
| `static_checks.test.js` | node --check, python compile, timeout guard |
| `task_contract.test.js` | task_class normalization, failure attribution |
| `task_lifecycle.test.js` | Single-finalization semantics across all terminal paths |
| `git_side_effects.test.js` | Auto-commit evidence, structured git arguments |
| `step_artifact_contract.test.js` | getWorkflowStepHandoff for all 6 step IDs |
| `coding_service_salvage.test.js` | salvageWorkflowArtifactFailure gate/happy paths |
| `coding_service_pure.test.js` | payloadToAdapterRequest, requiresScopedTargetPaths |
| `delegate_failure_injection.test.js` | **T-33**: 7 failure injection scenarios (see above) |
| `startup_smoke.test.js` | Import wiring, file existence, syntax checks |

### Untested paths (documented, not worth single-testing)

| Path | Reason deferred |
|------|----------------|
| `E_WALL_CLOCK_TIMEOUT` | Minimum clamp 60s — needs real elapsed time; covered by salvage tests conceptually |
| `E_PROMOTION_FAILED` | Only fires when `CODER_ISOLATION_MODE=promote` and promotion fails; covered in `promotion_workspace.test.js` |
| `createStepBuilder` inner `buildStepPayload` | Captures 4 services; extracting it adds noise with no testability gain |

---

## Blocking Points

None active.

### Recently resolved

| Block | Resolution |
|-------|-----------|
| Hardcoded `TASK_RUNNING_TIMEOUT_SEC="900"` overrode config | Changed default to `""` |
| `stream_batch_size` hardcoded 5 caused GPU saturation | Configurable, default 1 |
| `ollama/*` model refs rejected by opencode_adapter | Removed rejection block |
| Qwen/opencode `alibaba-coding-plan` auth fails | Ongoing non-blocker — stable_local_lane is primary |

### Known non-blocking issues

| Issue | Severity |
|-------|----------|
| Qwen via `opencode alibaba-coding-plan` auth fails | Medium — stable_local_lane is proven |
| Codex lane deferred (no OPENAI_API_KEY) | Low — lane defined, activate when key available |
| `createStepBuilder` inner closure not independently testable | Low — integration coverage sufficient |

---

## TODO — Next Steps (ordered by value)

### Immediate (next session)

| # | Task | Value |
|---|------|-------|
| N1 | **End-to-end live run** — restart docker stack, run one full `coding_team_v0` workflow on `stable_local_lane`, confirm the hardened timeout chain works end-to-end with the new config | High — validates all config changes together in the real runtime |
| N2 | **Multi-concurrent baseline** — run 2 workflows in parallel with `stream_batch_size=1` (current), measure wall time vs sequential; then try `stream_batch_size=2` and observe GPU queue behaviour | Medium — establishes throughput ceiling before any horizontal scaling |

### Near-term (M11 candidates)

| # | Task | Scope |
|---|------|-------|
| M11-A | Qwen cloud lane activation — resolve `alibaba-coding-plan` provider auth and promote `stable_cloud_lane` to production-ready | Medium |
| M11-B | Cohort expansion — add a second project type beyond `webapp_crm` | Medium |
| M11-C | Watchdog + DLQ observability — add alerting when tasks hit the DLQ (`stream:task:dlq`) | Low-medium |
| M11-D | `E_WALL_CLOCK_TIMEOUT` integration test — add a canary that uses a short `max_runtime_s` + verifies the timeout chain fires correctly with a slow mock command | Low |

---

## Key Runtime Parameters (current)

```json
{
  "execution_lane_default":    "stable_local_lane",
  "provider_default":          "opencode",
  "model_default":             "ollama/glm-4.7-flash:latest",
  "global_task_timeout_ms":    1800000,
  "wall_clock_timeout_s_default": 900,
  "running_timeout_sec":       1800,
  "stream_batch_size":         1
}
```

---

## Code Metrics (2026-03-15)

| Module | Lines | Notes |
|--------|-------|-------|
| `worker-coder/coding_service.js` | ~991 | Core delegate loop |
| `worker-coder/step_artifact_contract.js` | 121 | Extracted from coding_service |
| `orchestrator/src/index.js` | ~546 | HTTP router |
| `orchestrator/src/workflow_engine.js` | ~431 | Workflow engine |
| `orchestrator/src/domain/workflow_step_builder.js` | 626 | Step builder + timeout logic |
