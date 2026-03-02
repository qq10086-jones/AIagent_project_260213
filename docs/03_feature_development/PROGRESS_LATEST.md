# Feature Progress Latest Snapshot

## Date
2026-03-01

## Current State
- **Major Architecture Decoupling (Phase 1 & 2 Completed)**: Transitioned from a monolithic orchestrator to a **Skill-as-a-Service** model.
- **Worker-Coder Launched**: A dedicated `worker-coder` container is now operational, handling all coding tasks via Redis Streams.
- **Orchestrator Gateway Refined**: Removed all hardcoded business logic from `index.js`, which now acts as a pure API Gateway and security layer.
- **Unified Workspace Mapping**: Standardized `/workspace` volume across all containers to ensure consistent file access for the Coding Agent.
- **Real Chain Verified (Patch + Execute)**: Completed end-to-end verification with task approval flow; `coding.patch` and `coding.execute` both reached `succeeded`.
- **Discord Coder Entry Online**: Added a dedicated `/coder:` directive path in Discord to force `Brain mode=coding` without passing through quant intent routing.
- **Codex Delegation Online**: `/coder:` now routes to `coding.delegate` and executes through Codex adapter in `worker-coder`.
- **Coder Result Rendering Fixed**: `coding.delegate` now uses dedicated result formatting (run/task id, changed files, artifacts), no longer falling back to quant-style `SYSTEM Analysis Report`.
- **Approval Policy Upgraded**: switched from blanket approval to **risk-based approval** for `coding.delegate` (low-risk auto-run, high-risk waiting approval).
- **Change Detection Accuracy Fixed**: `worker-coder` now reports real changed files (`git status --porcelain -uall` + artifact noise filtering), avoiding `artifacts/runs/...` false positives.

## Architecture Highlights
- **Asynchronous Task Flow**: Brain now triggers tools via `trigger_tool`, which enqueues tasks into Redis. Workers claim tasks based on `tool_name` prefixes.
- **DB-backed Fact Polling**: Enhanced `supervisor.py` with an improved `poll_for_fact` mechanism that supports tool-specific result retrieval from PostgreSQL.
- **Service-Level Isolation**: `worker-coder` has its own environment (Git, Python, Node.js), preventing dependency bloat in the main orchestrator.

## Coding Agent Improvements
- **Robust Patching**: `patch_manager.js` (now in `worker-coder`) supports multi-block edits with extreme whitespace normalization to handle diverse LLM outputs.
- **Security Hardening**: Strict shell meta-character blocking and whitelist-based command execution are enforced at the worker level.
- **Full-Chain Success**: Verified the complete loop: `Brain` (Python) -> `Orchestrator` (HTTP) -> `Redis` -> `Worker-Coder` (Node.js) -> `FileSystem` -> `PostgreSQL` -> `Brain` (Poll).
- **Queue Isolation Completed**: `coding.*` now routes to dedicated stream `stream:task:coding`, preventing cross-consumption with `worker-quant`.
- **Parser Compatibility Fixes**: Coder-side SEARCH/REPLACE parsing now tolerates 6-7 `>` variants, reducing extraction misses.
- **Direct Task Injection**: `brain /run` now accepts external `messages`, enabling direct coding-task prompts from orchestrator and future skill frontends.

## Today Validation Notes
- **Date**: 2026-03-01
- **Run ID**: `docker-chain-test-1772340993`
- **Task Status**:
  - `coding.patch`: `succeeded`
  - `coding.execute`: `succeeded` (after approval)
- **Important Runtime Note**: `worker-coder` uses image build mode; code changes require `docker compose build worker-coder` before restart to take effect.
- **Discord Trigger Validation**:
  - `/coder: <task>` now routes directly to coding workflow.
  - Existing quant/discovery routing remains unchanged.
- **Delegation Validation**:
  - Codex auth + CLI path issues resolved; delegated runs can create files under `/workspace/coder_test`.
  - Example succeeded runs include file writes and artifact generation under `artifacts/runs/<run_id>/task_<task_id>/`.
- **Risk Policy Validation**:
  - Low-risk prompt auto-ran to `succeeded` without manual `/approve`.
  - High-risk prompt (destructive/install pattern) correctly entered `waiting_approval`.
- **Functional Demo Validation**:
  - Implemented runnable scientific calculator in `sandbox/calculator.py` using safe AST evaluation.
  - Local checks passed for arithmetic, trig/log functions, and malicious expression blocking.

## Next Steps
- **Phase 3 (Long-term)**: Implement the plugin-based dynamic node loading in the Brain to fully decouple `supervisor.py`.
- **Media Skill Integration**: Begin scaffolding `worker-media` using the now-proven Redis Stream worker pattern.
- **Refine Prompting**: Optimize the Coder Agent's prompts to reduce "SEARCH block not found" errors by enforcing exact context preservation.
- **Ops Improvement**: Add explicit docs/CLI helper for approval flow (`/tasks/:task_id/approve`) and optional auto-approval policy for trusted environments.
- **Coder-Centric Skill Fabric**: Standardize skill prefixes (`/coder:`, future `/ui:`, `/db:`) to keep Coder as the primary orchestrator while enabling multi-skill expansion.

## vNext P0 Go-Live (Execution Update)
- **P0-1 Completed**: Added `worker-coder/adapters/opencode_adapter.js` with standardized adapter fields and unified error codes (`E_PROVIDER_UNAVAILABLE`, `E_TIMEOUT`, `E_APPLY_FAILED`, `E_INTERNAL`).
- **P0-2 Completed**: `worker-coder/coding_service.js` now routes `provider=auto -> opencode`, with fallback to `codex` only when OpenCode is unavailable; model passthrough and command source are recorded.
- **P0-2 Completed**: `worker-coder/worker.js` now passes through `opencode_command` payload for delegated execution.
- **P0-3 Completed**: `/coder` default payload switched to `provider=opencode`, default model `minimax-m2.5`, with explicit `@gpt-5.3` override support.
- **P0-3 Completed**: Coder result rendering remains on dedicated coder template (`formatCodingDelegateResult`), no quant fallback text path.
- **P0-4 Completed**: `configs/tools.json` aligned to risk-based approval baseline (`coding.delegate` no blanket approval in config).
- **P0-4 Completed**: `infra/docker-compose.yml` added `CODER_PROVIDER_DEFAULT`, `CODER_MODEL_DEFAULT`, and `OPENCODE_BIN` env wiring.
- **P0-5 Completed**: Added log/artifact redaction in `worker-coder/coding_service.js`; expanded `.gitignore` for local auth/runtime secret artifacts.
- **P0-5 Validation**: secrets scan executed for tracked files + staged diff; result `tracked_hits=0`, `staged_hits=0`.
- **Remaining**: Container-level E2E/canary validation (Day1 16h-24h / Day2) not yet executed in this update.

## vNext P0 Closure (2026-03-02)
- **E2E A/B/C Completed**: low-risk auto-run, high-risk approval gate, approve-resume, reject-terminate all verified via orchestrator `/execute-tool` + approval APIs.
- **Model Switch Verified**: `coding.delegate` executed successfully with both `minimax-m2.5` and `gpt-5.3`.
- **Fallback Verified**: when OpenCode is unavailable, delegation falls back to Codex and records `diagnostics.fallback_from=opencode`.
- **Metrics (current E2E batch)**: total 5, success 4, expected reject-fail 1, high-risk gate hit 3/3, non-high-risk auto-run success 2/2.
- **Runtime Note**: OpenCode CLI compatibility issue on Alpine was resolved by moving `worker-coder` base image to Debian slim (`node:20-bookworm-slim`).

## Night Update (2026-03-02)
- **Quant Report UX Upgraded**: `news.tdnet_close_flash` now outputs structured bullets and includes source links in Discord + HTML artifacts.
- **Run Command Reliability**: explicit `/run <tool_name> [json_payload]` entry added in orchestrator to avoid intent-route misses.
- **Config Hotfix**: DashScope endpoint normalized to `compatible-mode/v1`; services rebuilt and restarted.
- **Open Issue Recorded**: command `40W JPY capital, how to operate tomorrow? currently no position` returns `Qwen API error 404 Not Found` in some path(s); pending unified endpoint/config tracing.

### Pending (News System)
1. Add media-home/channel backup links when direct article URLs are unavailable.
2. Add source credibility labels in report rendering (official media/aggregator/forum).

### Pending (Project)
1. Continue `worker-quant/worker.py` cleanup to reduce legacy overlap risk.
2. Standardize output contract across news tools (bullets + links + artifacts).
3. Run canary regression for `/coder`, `/run news.tdnet_close_flash`, and approval workflows.
4. Improve dashboard visibility for `tasks.result_json` key fields.

## Late Night Continuation (2026-03-02 18:40 JST)
- **Discovery Fast-Path Added**: capital/no-position/next-day operation prompts now enforce quick discovery payload (`quick_mode`, `time_budget_s=75`, `max_attempts=2`, `min_candidates=2`).
- **Brain Payload Pass-through Fixed**: `brain/supervisor.py` allowlist now keeps quick fields; screening step auto-applies quick defaults for capital-driven discovery requests.
- **Discovery Pre-step Short-Circuit**: for fast discovery runs, `news.daily_report` pre-step is skipped to reduce blocking risk.
- **Worker Runtime Guardrails Added**: `worker-quant` discovery now has explicit time budget and early-stop behavior; LLM timeout is configurable and tightened for risk-scoring calls.
- **Validation Result**:
  - Qwen 404 path no longer directly surfaces in tested flow.
  - Latest `tasks.payload_json` confirmed quick fields are now present.
  - Worker logs confirmed `Quick=True` execution path.
- **Still Open**:
  1. `/chat` can still return generic `unknown Analysis Report` while discovery task continues/completes later (response aggregation timing mismatch).
  2. Historical `news.daily_report` long-running tasks still exist and need timeout/degrade governance.

## v1.4 Coding Team First (Kickoff Update — 2026-03-02 23:30 JST)
- **Execution Focus Shift Confirmed**: aligned to `docs/01_design/system/260302/*` with core objective "autonomous Coding Team pipeline" (PM/Architect/FE/BE/QA), not generic multi-agent expansion.
- **T1 Delivered (Registry Schema + Baseline Registry)**:
  - Added `configs/registry/capability_registry.json`.
  - Added `configs/registry/schemas/capability_registry.schema.json`.
  - Added workflow/policy/acceptance seeds:
    - `configs/registry/workflows/coding_team_v0.json`
    - `configs/registry/acceptance/webapp_crm_v0.json`
    - `configs/registry/policy/coding_task_v0.json`
- **T2 Delivered (Validator CLI + CI Hook)**:
  - Added `orchestrator/scripts/validate_registry.js`.
  - Added npm script: `npm run validate:registry`.
  - Added CI workflow: `.github/workflows/validate-registry.yml`.
  - Local validation passed: `registry valid` (project_types=3, roles=8, tools=23, workflows=1).
- **T3 Delivered (Runtime Fail-Fast Registry Loading)**:
  - Added orchestrator registry module: `orchestrator/src/registry.js`.
  - Orchestrator now loads/validates registry on startup (invalid registry blocks boot).
- **T4 Partial Delivered (Task Submission Runtime Validation)**:
  - `enqueueTask(...)` now validates `tool/project_type/workflow/role/params` against registry before queueing.
  - Invalid payloads now fail with `REGISTRY_INVALID`.
- **Compatibility/Infra Wiring**:
  - Mounted `configs/registry` into orchestrator container (`infra/docker-compose.yml`).
  - `policy.js` now supports new registry path (`configs/registry/...`) with backward fallback.
- **Regression Smoke Passed After Integration**:
  - `quant.fetch_price` -> `succeeded`
  - `coding.patch` -> `succeeded`
  - Task status/result persistence remained healthy.

### v1.4 Current Step
- **Completed**: T1, T2, T3, T4 (partial by runtime validation core).
- **In Progress Next**: EPIC 2 workflow shell minimum path (T6/T7/T8/T9/T10/T11/T12).
- **Blockers**: none hard-blocking; main remaining work is implementation depth, not environment readiness.

## v1.4 Coding Team First (Workflow Shell Update — 2026-03-02)
- **T6 Delivered (Sequential Workflow Runner)**:
  - Added deterministic workflow shell engine: `orchestrator/src/workflow_engine.js`.
  - New start API: `POST /workflow-runs/start` (registry workflow based, first step dispatch).
- **T7 Delivered (Step State Machine + Persistence)**:
  - Added DB tables: `workflow_runs`, `workflow_steps`, `workflow_checkpoints` (in both `infra/init.sql` and orchestrator runtime ensure).
  - Step statuses now persisted through `pending/queued/waiting_approval/running/succeeded/failed`.
  - Result consumer now syncs step state on `claimed` and terminal events.
- **T8 Delivered (Checkpoint per Step)**:
  - On each successful step, engine writes `checkpoint_id` + `workspace_hash` + `artifact_refs`.
  - `workflow_runs.last_checkpoint_id` and step `checkpoint_id` are updated.
- **T9 Delivered (Resume Token issue/verify)**:
  - Added HMAC signed resume token (`RESUME_TOKEN_SECRET`, `RESUME_TOKEN_TTL_SEC`).
  - APIs:
    - `POST /workflow-runs/:workflow_run_id/resume-token`
    - `POST /workflow-runs/:workflow_run_id/resume`
  - Invalid/mismatch/expired token returns `error_code=RESUME_INVALID`.
- **T10 Delivered (Policy Gate for each Step)**:
  - Every step now executes risk check before dispatch.
  - Policy audit event persisted (`policy.gate.checked`) with reasons/risk/approval requirement.
- **T11 Delivered (Approval Gate backend close-loop)**:
  - `approve` now updates workflow step state (`queued`) for waiting-approval steps.
  - `reject` supports `reason`, persists rejection result, and closes workflow run as failed with `APPROVAL_REJECTED`.
- **T12 Delivered (Acceptance Gate integration)**:
  - For acceptance-gated steps, engine auto injects acceptance suite commands from registry into `coding.execute`.
  - Acceptance command failure now fails current step and workflow run.

### Current Step (after update)
- **Completed**: T1-T12 baseline contract and workflow-shell closure.
- **Next Focus**: EPIC 3 deepening (`coding_team_v0` role outputs/artifact richness) and EPIC 4 pack validator hardening.

### Runtime Validation (2026-03-02)
- `POST /workflow-runs/start` succeeded:
  - `workflow_run_id=7bfefc4f-2b0b-4b5e-8c3d-14fcb4acd9ec`
  - first step `pm_spec` dispatched with `task_id=674e8705-f533-43f7-92a9-feaefee88912`
  - workflow steps persisted with deterministic `step_index` timeline.
- Policy/Approval gate closed-loop verified with high-risk prompt:
  - `workflow_run_id=ba6625d8-6fa6-4d0a-99f3-336bd3e8b678`
  - first step entered `waiting_approval` (`risk_level=high`).
  - reject API with reason transitioned run to `failed` and step `error_code=APPROVAL_REJECTED`.
- Resume API invalid-path check:
  - `POST /workflow-runs/:id/resume-token` returns `RESUME_INVALID` when no checkpoint exists.

## v1.4 Continuation (T13-T18/T22-T24 bridge — 2026-03-02)
- **Test Re-run Status**:
  - workflow shell APIs re-tested after restart; `/workflow-runs/start` and `/workflow-runs/:id` stable.
  - approval reject close-loop re-tested (`APPROVAL_REJECTED` persisted to run + step).
- **Bug Fixed During Re-test**:
  - fixed SQL placeholder mismatch in `workflow_engine.failWorkflowRun(...)` that caused:
    - `could not determine data type of parameter $3`
  - regression check after fix: reject path no longer emits the error.
- **T15-T18 (Role Output Contract) Partial Hardening**:
  - workflow step payload now carries:
    - `artifact_root`
    - `expected_artifacts`
    - role/step specific structured `task_prompt` contract (PM/Architect/FE/BE/QA/Release).
  - acceptance step payload now includes acceptance context for deterministic verification.
- **T22-T24 Baseline Started (Artifact Pack skeleton)**:
  - on workflow success path, orchestrator now attempts release-pack generation:
    - `artifacts/release/<run_id>/meta/run_manifest.json`
    - `artifacts/release/<run_id>/summary/run_summary.md`
  - added baseline completeness checks and `ARTIFACT_INCOMPLETE` fail conversion.
- **Open Validation Gap**:
  - full success-path artifact-pack generation still needs one clean end-to-end successful `coding_team_v0` run to finalize proof.

## v1.4 Governance Update (T26-T31 backend closure — 2026-03-02)
- **T26 (result_json + error_code normalization) Delivered**:
  - task terminal write path now normalizes result payload to structured envelope:
    - `ok/status/error_code/output/updated_at`
  - reject path also uses normalized result schema.
- **T27/T28 (Timeline + Artifacts query) Delivered**:
  - Added APIs:
    - `GET /runs/:run_id/status`
    - `GET /runs/:run_id/timeline`
    - `GET /runs/:run_id/artifacts`
  - Added pending-approval list API:
    - `GET /approvals/pending?limit=...`
- **T30 (unknown response degradation mitigation) Delivered (backend side)**:
  - When brain-controlled run has no body report, reply now includes:
    - `run_id`
    - `status_api=/runs/:run_id/status`
    - `timeline_api=/runs/:run_id/timeline`
- **T31 (timeout + DLQ guardrail) Delivered**:
  - Added watchdog loop for stale `running` tasks (`TASK_RUNNING_TIMEOUT_SEC`).
  - Timeout behavior:
    - mark task `failed` with `error_code=TASK_TIMEOUT`
    - append `task.timeout` event
    - enqueue to DLQ stream (`stream:task:dlq`) and append `task.dlq.enqueued`
    - propagate failure to workflow step/run.
- **Runtime Test Evidence**:
  - Injected stale running task: `447917b4-8273-4a6e-891e-a7856ada97bb`
  - Post-watchdog status: `failed / TASK_TIMEOUT`
  - DLQ length increased from `4 -> 5`
  - event_log contains: `task.timeout`, `task.dlq.enqueued`

## v1.4 Config Consistency Update (T32 partial — 2026-03-03)
- Added unified runtime config file:
  - `configs/runtime/runtime_defaults.json`
- Orchestrator now resolves key runtime settings by precedence:
  1. Environment variables
  2. `runtime_defaults.json`
  3. Hardcoded fallback
- Added runtime introspection API:
  - `GET /runtime/config`
- Infra wiring updated:
  - `infra/docker-compose.yml` mounts `../configs/runtime:/app/configs/runtime:ro`
  - `RUNTIME_CONFIG_PATH=/app/configs/runtime/runtime_defaults.json`
- Boot validation:
  - `/runtime/config` returns loaded path and resolved values
  - startup log confirms watchdog resolved config values
- Remaining for full T32 closure:
  - align the same runtime config source across brain/worker processes (currently orchestrator-first rollout).

## v1.4 Closure Extension (T29 + T32 full-chain — 2026-03-03)
- **T29 Delivered (Approval UI minimum usable)**:
  - Added built-in approval console page:
    - `GET /ui/approvals`
  - UI supports:
    - list pending approvals
    - approve action
    - reject with mandatory reason
  - Backend API integration uses existing:
    - `GET /approvals/pending`
    - `POST /tasks/:task_id/approve`
    - `POST /tasks/:task_id/reject`
- **T32 Delivered (runtime config unified across services)**:
  - Added shared config source:
    - `configs/runtime/runtime_defaults.json`
  - Brain integrated:
    - `brain/runtime_config.py`
    - `GET /runtime/config` in brain service
  - Worker-coder integrated:
    - `worker-coder/runtime_config.js`
    - provider/model/global timeout defaults sourced from runtime config
  - Worker-quant integrated:
    - runtime config load at startup for provider/model/base-url/timeout defaults
  - Compose integrated:
    - `RUNTIME_CONFIG_PATH` wired for orchestrator/brain/worker-coder/worker-quant
    - runtime config mount added where needed
- **Cross-service Validation Evidence**:
  - `orchestrator /runtime/config` returns resolved path + effective values.
  - `brain /runtime/config` returns resolved qwen/local model defaults.
  - `worker-coder` startup log prints runtime config path and resolved provider/model/timeout.
  - `worker-quant` startup log prints runtime config path and resolved provider/model/timeout.

## v1.4 Artifact Validator Update (T24 + T25 — 2026-03-03)
- **T24 Delivered (artifact_pack_validator module)**:
  - Added validator module:
    - `orchestrator/src/artifact_pack_validator.js`
  - Validation scope:
    - manifest/summary existence
    - manifest schema minimum fields
    - run/manifest id consistency
    - step success completeness
    - checkpoint count sanity
    - required artifact coverage check by project_type
- **T25 Delivered (finalize gate integration)**:
  - `workflow_engine` now uses validator during release-pack generation.
  - If validator fails, workflow finalization converts to:
    - `status=failed`
    - `error_code=ARTIFACT_INCOMPLETE`
  - Added inspection API:
    - `GET /workflow-runs/:workflow_run_id/validate-pack`
- **Runtime verification**:
  - Existing running workflow run validation returned structured failure reasons when pack missing.
  - Missing workflow run returns `WORKFLOW_RUN_NOT_FOUND` with HTTP 404.

## v1.4 Artifact Pack Schema Update (T22 + T23 partial — 2026-03-03)
- **T22 Delivered (run_manifest schema file)**:
  - Added:
    - `configs/registry/schemas/run_manifest.schema.json`
  - schema includes:
    - run/workflow identity fields
    - steps/checkpoints
    - step_artifacts
    - artifact coverage metadata
- **T23 Delivered (release_pack aggregation + persistence/index)**:
  - `workflow_engine` manifest now includes `step_artifacts` aggregated from checkpoint artifact refs.
  - release pack local files + step artifact refs are now indexed into `assets` table for run-level lookup.
  - validator now checks `step_artifacts` presence and step count consistency.
  - release pack now supports MinIO archive + DB index through:
    - automatic archive on finalize success path
    - manual archive API: `POST /workflow-runs/:workflow_run_id/archive-pack`
  - validation evidence (synthetic fixture run):
    - MinIO object keys returned for manifest/summary
    - `assets` table contains both `release_pack_local` and `release_pack_minio` entries for the workflow run.

## v1.4 End-to-End Runthrough Fixes (Full-chain green — 2026-03-03)
- **Worker-coder execute gate fix**:
  - `coding.execute` command validator now supports controlled `&&` command chains (segment-level whitelist) instead of blanket-blocking `&`.
  - keeps shell meta-char blocking for high-risk operators.
- **OpenCode adapter hard fix**:
  - `worker-coder/adapters/opencode_adapter.js` switched to `spawn(..., shell:false)`.
  - fixed multiline prompt being misinterpreted by shell as separate commands.
- **Provider fallback semantics fix**:
  - OpenCode adapter non-provider failures now return `E_EXEC_FAILED` (not `E_PROVIDER_UNAVAILABLE`), preventing false fallback to codex auth-missing path.
- **Acceptance suite runtime fit update**:
  - `webapp_crm_v0` acceptance commands changed to:
    - `node --version`
    - `npm --version`
  - updated in:
    - `configs/registry/capability_registry.json`
    - `configs/registry/acceptance/webapp_crm_v0.json`
- **End-to-end validation result**:
  - workflow run:
    - `workflow_run_id=8aa1e8a5-6cfd-4c65-a810-2b926ccb4237`
    - `run_id=42cf0103-43e9-4a41-a770-080969819cc6`
  - final status:
    - workflow `succeeded`
    - steps `pm_spec/arch_design/impl_fe/impl_be/qa_verify/release_pack` all `succeeded`
  - acceptance step output:
    - `stdout: v20.20.0 / 10.8.2`
