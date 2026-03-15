# Feature Progress - Latest Snapshot

**Last updated:** 2026-03-16 (MiniMax mixed-lane parallel evidence captured; N2 remains in progress)
**Author:** PM / Architecture Review

---

## Execution Evidence

- **M10 load test (stable_local_lane):** `PASS / GO` - run a48a521e, 6/6 steps, 23 min (2026-03-15)
- **N1 live runtime validation, failure-path canary:** `PASS AS EXPECTED` - run `919efc21-32dc-4733-8fd2-421f0c76168e` failed at `impl_be` because the validation input explicitly set `max_runtime_s=180` (2026-03-15)
- **N1 live runtime validation, default-timeout canary:** `PASS / GO` - run `1b71e15d-df8a-48b0-8557-3b0e0ef7d4b4`, `coding_team_v0` completed `6/6` on `stable_local_lane` after restart with default runtime timeout behavior (2026-03-15)
- **N2 concurrent baseline, initial run:** `FAIL` - one workflow failed at `arch_design` with `ARCH_REQUIRED_SECTIONS_MISSING`; a second workflow remained stuck at `arch_design` under 2-workflow concurrency (2026-03-15)
- **N2 concurrent baseline, rerun after arch prompt hardening:** `PARTIAL IMPROVEMENT / NOT GO` - both workflows advanced past `arch_design` and reached `impl_be`, but neither reached terminal state within the observation window; primary bottleneck moved from `arch_design` to `impl_be` (2026-03-15)
- **M11-A MiniMax cloud lane wiring:** `CONFIGURED` - `stable_cloud_lane` now points to `opencode + minimax/MiniMax-M2.5`, `MINIMAX_API_KEY` is injected into `worker-coder`, and preflight/adapter coverage was updated (2026-03-16)
- **M11-A MiniMax cloud lane smoke:** `PASS / GO` - after switching `opencode.json` to the official MiniMax provider shape and forcing full-file fallback on `stable_cloud_lane` impl steps, live workflow `8418008d-a2d2-4317-bac3-879e81016f0a` / `87658374-e4b8-43eb-9491-842e7510bef0` completed `6/6` on MiniMax (2026-03-16)
- **M11-A mixed-lane parallel baseline:** `PARTIAL PASS` - concurrent `stable_local_lane + stable_cloud_lane` run showed `stable_cloud_lane` succeeded `6/6` (`e29d41ce-9704-4656-84d8-5515edc19d48` / `f507a1be-dda6-4534-9edc-bf35e081c0d0`) while `stable_local_lane` still failed at `arch_design` with `ARCH_REQUIRED_SECTIONS_MISSING` (`b819029c-4a5d-4cf0-ba20-adaa0e279cff` / `2ea12610-7d5d-49b9-bce5-157610413160`) (2026-03-16)
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
- `stable_codex_lane` defined in runtime config (provider=codex, model=codex-mini-latest)  Edeferred until OPENAI_API_KEY is available
- `stable_cloud_lane` now targets `opencode + minimax/MiniMax-M2.5`; official MiniMax provider shape validated and a full `coding_team_v0` live run completed `6/6`

---

## What Was Done (2026-03-13 ↁE2026-03-15)

### Round 1  EQA/Architect audit fixes (6 issues)

| Fix | Commit |
|-----|--------|
| `TASK_RUNNING_TIMEOUT_SEC = "900"` default overrode config 1800s ↁEchanged to `""` | `d365036` |
| `stream_batch_size` hardcoded 5 in worker.js ↁEconfigurable (default 1) | `d365036` |
| `coding_service.js` inline dynamic import plumbing ↁEextracted to `step_artifact_contract.js` | `d365036` |
| `salvageWorkflowArtifactFailure` private/untested ↁEexported + `coding_service_salvage.test.js` (5 cases) | `e0dcab6` |
| `artifact_scaffold.js` CRM-specific heading check ↁEgeneralized | `d365036` |
| `opencode_adapter.js` blocked `ollama/*` refs ↁEremoved; added `mapErrorCode` fast-path | `25b0860` |

### Round 2  EDeeper audit fixes (P0–P2)

| Fix | Commit |
|-----|--------|
| `step_artifact_contract.test.js`: 18 assertions, 6 step IDs, fresh-object guarantee | `e0dcab6` |
| `startup_smoke.test.js`: import wiring + file existence assertions | `e0dcab6` |
| `wall_clock_timeout_s` clamp boundary tests (3 cases); orchestrator 147ↁE50 | `36cb2a7` |
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
npm --prefix orchestrator test    - 150 / 150 PASS  (2026-03-15)
npm --prefix worker-coder test    -  19 /  19 PASS  (2026-03-15)
node scripts/canary_coding_team_e2e.js
  happy_path (6/6 steps)          - PASS
  be_failure (artifacts missing)  - PASS
  qa_failure (QA blocks release)  - PASS
M10 load test (stable_local_lane) - PASS / GO verdict (2026-03-15)
node orchestrator/scripts/live_validate_workflow_runtime.js --base-url http://localhost:3000 --input crm_mini.json --timeout-ms 480000
  failure-path canary (`max_runtime_s=180`) - PASS AS EXPECTED
node orchestrator/scripts/live_validate_workflow_runtime.js --base-url http://localhost:3000 --input crm_mini_default_timeout.json --timeout-ms 1500000
  default-timeout live canary                - PASS / GO
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
| `E_WALL_CLOCK_TIMEOUT` | Minimum clamp 60s  Eneeds real elapsed time; covered by salvage tests conceptually |
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
| Qwen/opencode `alibaba-coding-plan` auth fails | Ongoing non-blocker  Estable_local_lane is primary |

### Known non-blocking issues

| Issue | Severity |
|-------|----------|
| Qwen via `opencode alibaba-coding-plan` auth fails | Medium - stable_local_lane is proven |
| Codex lane deferred (no OPENAI_API_KEY) | Low - lane defined, activate when key available |
| `createStepBuilder` inner closure not independently testable | Low - integration coverage sufficient |
| `crm_mini.json` forces `max_runtime_s=180` and should be used only for timeout-path validation | Low - use `crm_mini_default_timeout.json` for default-timeout GO validation |

---

## TODO  ENext Steps (ordered by value)

### Immediate (next session)

| # | Task | Value |
|---|------|-------|
| N1 | **End-to-end live run** - restart docker stack, run one full `coding_team_v0` workflow on `stable_local_lane`, confirm the hardened timeout chain works end-to-end with the new config | **Completed 2026-03-15** - restart evidence, timeout/failure-path evidence, and default-timeout success evidence captured |
| N2 | **Multi-concurrent baseline** - run 2 workflows in parallel with `stream_batch_size=1` (current), measure wall time vs sequential; then try `stream_batch_size=2` and observe GPU queue behaviour | **In progress (2026-03-16)** - `arch_design` concurrency failure mitigated, but `impl_be` is now the active bottleneck; do not raise `stream_batch_size` yet |

### Near-term (M11 candidates)

| # | Task | Scope |
|---|------|-------|
| M11-A | MiniMax cloud lane activation - official `opencode` provider shape validated; mixed-lane evidence shows MiniMax can carry real load in parallel, so decide whether to promote `stable_cloud_lane` as explicit overflow/default-for-heavy-tasks lane | Medium |
| M11-B | Cohort expansion  Eadd a second project type beyond `webapp_crm` | Medium |
| M11-C | Watchdog + DLQ observability  Eadd alerting when tasks hit the DLQ (`stream:task:dlq`) | Low-medium |
| M11-D | `E_WALL_CLOCK_TIMEOUT` integration test  Eadd a canary that uses a short `max_runtime_s` + verifies the timeout chain fires correctly with a slow mock command | Low |

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




