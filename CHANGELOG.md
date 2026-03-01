# Changelog

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
