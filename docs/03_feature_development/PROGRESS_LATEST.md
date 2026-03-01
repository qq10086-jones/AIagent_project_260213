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
