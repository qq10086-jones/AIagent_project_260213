# Feature Progress - Latest Snapshot

**Last updated:** 2026-03-16 (MiniMax M2.5 fully activated; Discord Live Progress Bar landed; M12 local-preview design approved)
**Author:** PM / Architecture Review

---

## M11/M12 UX & Local Preview (Current Focus)

- **MiniMax M2.5 Cloud Lane (M11-A):** `COMPLETE` - `stable_cloud_lane` now targets MiniMax M2.5 via OpenCode. `MINIMAX_API_KEY` injected and verified with a 6/6 step successful workflow run (`15952cc0-3b6a-471f-86e3-00c018fa7b91`).
- **Discord Live Progress Bar (M12 Phase A):** `LANDED` - Real-time, debounced progress tracking implemented (`discord_progress_manager.js`). Workflow engine now emits fine-grained step events to Discord messages.
- **M12 Local Preview Pivot:** `APPROVED` - Live Preview primary path changed from Render-first to localhost-first. Render is now a deferred public-preview enhancement.
- **Brain Model Upgrade:** `COMPLETE` - Brain upgraded to `qwen3.5-397b-a17b`. Intent detection hardened for Chinese natural language via regex refactoring and `coding.delegate` schema registration.
- **Project Renaming:** All internal references and Discord branding updated to **Nexus**.

---

## Execution Evidence

- **M11-A MiniMax cloud lane smoke:** `PASS / GO` - full workflow `15952cc0-3b6a-471f-86e3-00c018fa7b91` completed `6/6` on MiniMax M2.5 (2026-03-16).
- **M12 Phase A (UX) verification:** `PASS` - Discord message updates dynamically with heartbeat and step status; debouncing (3s) verified against rate limits (2026-03-16).
- **M12 Phase B design decision:** `LOCKED` - `deploy_preview` should first launch local ephemeral previews on `localhost`, not depend on paid cloud infrastructure (2026-03-16).
- **Chinese Intent Routing:** `FIXED` - Correctly routes "做个网站" to `coding_team_v0` after removing `\b` boundary restrictions and updating `AGENT_TOOLS_SCHEMA` (2026-03-16).
- **Environment Sanitation:** `COMPLETE` - Cleared persistent Shell environment leaks (`QWEN_MODEL`, `CODER_PROVIDER`) that caused config drift (2026-03-16).

---

## Milestone Summary

| Milestone | Description | Status | Date |
|-----------|-------------|--------|------|
| M10 | Load Test + Execution Lane Validation | **CLOSED** | 2026-03-15 |
| M11 | Cloud Lane Scaling (MiniMax) | **CLOSED** | 2026-03-16 |
| M12 | Discord UX & Live Preview | **IN PROGRESS** | 2026-03-16 |

---

## Active Design Authority

**M12: Discord UX & Live Preview (v2.2)**
- **Approved Design:** Local ephemeral preview runtime on `localhost` + static dependency scanning for eligibility.
- **Current State:** Phase A (Progress Bar) completed. Phase B (Local Preview Launcher) pending.

Current governance state:
- `production_parallel_rollout.json`: `router_mode=dynamic_routing_enforced`
- Default execution lane: `stable_cloud_lane` (MiniMax M2.5) - **Promoted from local lane for Alpha testing.**
- Brain Model: `qwen3.5-397b-a17b`

---

## TODO — Next Steps (ordered by value)

### Immediate (M12 Phase B/C)

| # | Task | Value |
|---|------|-------|
| WS-40 | **Local Preview Launcher** - Implement localhost preview boot, port allocation, and TTL cleanup for `ops.deploy_preview`. | High |
| WS-41 | **Workflow Extension** - Keep `deploy_preview` in `coding_team_v0` and update Arch Design prompts for deterministic local boot metadata. | High |
| WS-39 | **Persistent Progress State** - Move Discord message mapping from in-memory Map to Redis/SQLite for restart resilience. | Medium |

### Near-term

| # | Task | Scope |
|---|------|-------|
| WS-40-03 | **Static Dependency Scanner** - Implement `package.json`/`requirements.txt` scanner to block DB-heavy previews. | Medium |
| WS-42-03 | **Preview Runtime Registry** - Persist local preview pid/port/expiry metadata for restart-safe cleanup. | Medium |
| WS-42-04 | **Cloud Upgrade Path** - Revisit Render only after localhost preview is stable. | Low |
| N2 | **Concurrency Tuning** - Address `impl_be` bottleneck to allow `stream_batch_size > 1`. | Medium |

---

## Key Runtime Parameters (current)

```json
{
  "execution_lane_default":    "stable_cloud_lane",
  "provider_default":          "opencode",
  "model_default":             "minimax/MiniMax-M2.5",
  "qwen_model":                "qwen3.5-397b-a17b",
  "stream_batch_size":         1
}
```

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




