# Feature Progress Latest Snapshot

## Date
2026-03-01

## Current State
- Docs structure has been reorganized into `design / patch / feature_progress / archive`.
- Latest design baselines are centralized under `docs/01_design/`.
- Historical and multi-version documents are preserved under `docs/90_archive/`.

## Quant & Routing Highlights (Latest)
- Geo-impact JP query routing is workflow-safe with explicit market lock (`JP`, no auto expansion).
- `news.active_hot_search` supports global source mix and holdings-priority enrichment.
- JP numeric symbol normalization (e.g., `9432` -> `9432.T`) is active.
- Blank screenshot artifacts are filtered; capture diagnostics are exposed in output.

## Documentation Hygiene
- Active docs path references were repaired to new locations.
- One legacy mojibake-heavy progress file was archived and replaced by a placeholder.
- Navigation indexes were added:
  - `docs/README.md`
  - `docs/00_overview/docs_migration_map_20260301.md`
  - `docs/02_patch/PATCH_INDEX.md`
  - `docs/03_feature_development/FEATURE_PROGRESS_INDEX.md`

## Next Suggested Maintenance
- If needed, reconstruct archived legacy raw reports from git history into cleaned markdown snapshots.
