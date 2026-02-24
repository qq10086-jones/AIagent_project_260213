# Changelog

## 2026-02-24
### Added
- Daily market news report pipeline (`news.daily_report`) generating HTML + PNG artifacts.
- Daily GitHub agent skills report pipeline (`github.skills_daily_report`) generating HTML + PNG artifacts.
- US stock universe seed file (`configs/universe_us.json`).
- UI quick-launch buttons for daily reports.

### Changed
- Worker dependencies updated to support report rendering, scraping, and charts.
- Docker Compose mounts for new config.

