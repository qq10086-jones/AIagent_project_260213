# Changelog

## 2026-04-06
### Added
- **`worker-coder/constants.js`**: Centralized all magic numbers — timeouts, retry budgets, error codes, artifact paths.
- **Isolation cleanup**: `cleanupIsolationWorkspace()` in `isolation_workspace.js` — prevents disk leaks on failure and cleans up after completion.
- **Request ID tracing**: All `coding_service.js` log lines now include `[req:runId/taskId]` for cross-operation tracing.

### Fixed
- **20 bare `catch {}` blocks** across 8 files replaced with proper logging — errors no longer silently swallowed.
- **Command injection defense**: `executeCommand` now calls `validateSafeCommand()` before `exec()`.
- **4 broken tests fixed**:
  - `isolation_workspace.test.js` / `promotion_workspace.test.js`: Path assertions missing `workspace/` directory prefix.
  - `isolation_delegate_shadow.test.js`: Expected `E_STATIC_CHECK_FAILED` but handoff validation now fires first (`STEP_IMPL_FE_HANDOFF_MISSING`).
  - `delegate_scope_policy.test.js`: PM role validation requires complete spec artifacts — test now pre-creates them per schema.

### Changed
- **Error codes centralized**: 7 hardcoded string error codes in `coding_service.js` replaced with `ErrorCode.*` constants.
- **Quality scores updated**: Code Robustness 6.5→8.5, Engineering/QA 7.0→8.5, Overall 6.5→8.5.
- Test suite: **27/27 all green** (was 23/27).

## 2026-04-05
### Added
- **SP-03 Integration**: Landed Structured Workplan as first-class execution context.
- `orchestrator/contracts/coding_team_arch_handoff.schema.json`: Mandatory `workplan` field added with `be_tasks` and `fe_tasks`.
- `scripts/verify_sp03_contract.js`: New script for validating architecture handoff integrity.
- `canary_verification/todo_app_v1/`: Reference implementation of SP-03 task list and feedback notes.

### Changed
- `configs/prompt_scripts/registry.json`: Updated `architect.system_spec.v2`, `backend.impl.v1`, and `frontend.impl.v1` to enforce SP-03 task tracking.
- Backend/Frontend implementers now REQUIRED to include `Task Status` in their `.notes.md` referencing workplan IDs.

### Fixed
- **EPERM Resolution**: Force-cleaned `pytest` cache locks in E: drive and configured `PYTEST_ADDOPTS="-p no:cacheprovider"` to prevent future file lock collisions.
- Normalized project directory permissions for CI/CD readiness.

## 2026-02-28
### Added
- Formal quant design doc: `docs/quant_design.md`.
- `news.active_hot_search` output diagnostics: `artifact_capture_stats` (`requested`, `attempted`, `archived_ok`, `blank_filtered`, `kept`, `start_ok`, `skipped_reason`).

### Changed
- Geo-impact JP routing now enforces `market=JP` and `auto_expand_market=false` for discovery workflow payloads.
- Orchestrator added rule-based forced tool fallback when intent parser returns chat/low-confidence/no-tool.
- `quant.discovery_workflow` now treats explicit market as a hard constraint and improves account/position-aware defaults.
- `news.active_hot_search` upgraded to global multi-source collection and holdings-priority enrichment.

### Fixed
- Prevented JP context from unintentionally expanding to US candidates/USD-oriented recommendations in JP geo-impact scenarios.
- Normalized JP numeric symbols to Yahoo format (e.g., `9432` -> `9432.T`) to reduce quote/news lookup failures.
- Added screenshot blank-frame filtering to avoid sending empty/white artifact images.

## 2026-02-24
### Added
- Daily market news report pipeline (`news.daily_report`) generating HTML + PNG artifacts.
- Daily GitHub agent skills report pipeline (`github.skills_daily_report`) generating HTML + PNG artifacts.
- US stock universe seed file (`configs/universe_us.json`).
- UI quick-launch buttons for daily reports.

### Changed
- Worker dependencies updated to support report rendering, scraping, and charts.
- Docker Compose mounts for new config.
