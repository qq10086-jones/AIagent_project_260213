# Changelog

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
