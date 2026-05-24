# ADR-0003: Decision Log Infrastructure As Shared Foundation

## Status

Accepted 2026-05-23.

## Context

Governance §8.6 mandates a feedback log for every instrument-level prediction with the following fields: `prediction_id`, `symbol`, `trade_date`, `decision_cutoff`, `input_snapshot_id`, `model_version`, probability or score status, and buy/sell/hold outputs. §9.4 mandates calibration before any score may be labeled as a win rate. §10 names Decision Logging as gate 3 — the first gate that must be passed before any output can leave `uncalibrated_research_score` / `insufficient_calibration` status.

The codebase has two parallel decision-producing paths:

1. **Attribution path**: `src/hot_theme_rotator/attribution/baseline_decision_score.py` produces `SymbolDecisionScore` and `IntegratedDecisionScore`. Already carries `model_version` (constant `"baseline-v0"`) and a deterministic `snapshot_id` pattern (`pit-{symbol}-{trade_date}-{sha256_16hex}`) via `attribution/point_in_time_adapter.py`.
2. **Opportunity path**: `src/hot_theme_rotator/opportunity/opportunity_scanner.py` produces `OpportunityCandidate` and `OpportunityScanResult`. Carries `score_status` and `decision_cutoff` but does **not** carry `model_version` or `prediction_id`.

Neither path persists its predictions today. As entries begin to fire — the P6-04 diagnostic currently shows 0 of 38 sample signals as entries due to `ENTRY_SCORE_TOO_LOW` — every prediction will be produced and lost. That violates the spirit of §8.6 and blocks any future calibration sample.

P7-06, P8-05, P9-01, P9-02, P9-03 were originally written as five separate task entries. They describe the same underlying subsystem from three perspectives:

- attribution (P7-06)
- opportunity ladder (P8-05)
- generic automation gates 3 / 4 / 5 (P9-01 / P9-02 / P9-03)

Five independent implementations would produce five schemas, five writers, and five storage formats, contradicting Rule 2's single-source principle and producing fragmented feedback data that a single calibration engine cannot consume.

## Decision

Introduce a single Decision Log subsystem at `src/hot_theme_rotator/decision_log/` that:

1. Defines one `PredictionRecord` schema covering both attribution and opportunity prediction shapes. Domain-specific fields live under an explicit `extra: dict[str, Any]` field with documented keys per path.
2. Provides one JSONL writer (`append_prediction`) and one reader (`read_predictions`) that fail closed on missing required fields and on duplicate `prediction_id`. **Point-in-time `available_ts > decision_cutoff` enforcement is upstream** of the writer — `opportunity.scan_opportunities` raises `OpportunityValidationError` and `attribution.build_symbol_snapshot` raises `AttributionValidationError` before any `PredictionRecord` is constructed. The writer therefore does not need to re-check PIT; it trusts that records reaching it have already passed upstream PIT validation. Direct callers who bypass the scanner / adapter and construct `PredictionRecord` themselves are responsible for the PIT contract.
3. Owns the storage convention at `reports/predictions/` mirroring the existing `reports/{daily, backtests, paper}` pattern. One JSONL file per trade date.
4. Generates `prediction_id` deterministically as `sha256(input_snapshot_id || model_version || decision_cutoff || symbol)[:16]`. Two identical predictions produce the same id, enforcing reproducibility and idempotent writes.

`opportunity_scanner` and `baseline_decision_score` both adopt the `model_version` discipline and call `append_prediction` for every emitted prediction. `realtime_opportunity_panel` is the first integration target; attribution path integration follows in P7-06.

P9-01 covers the infrastructure (schema + writer + first integration). P9-02 covers outcome join on top of the same storage. P9-03 covers calibration math. P7-06 narrows to attribution-specific calibration evaluation; P8-05 narrows to opportunity-ladder-specific calibration evaluation. None of P7-06 / P8-05 implement their own storage.

## Consequences

Positive:

- Single source of truth for §8.6 compliance.
- Calibration in P9-03 consumes a single dataset format, not five.
- Reuses the existing `snapshot_id` SHA-256 pattern from `point_in_time_adapter.py` and the `model_version` constant pattern from `baseline_decision_score.py`.
- The §10 ordering stays intact: gate 3 (logging) before gate 4 (outcomes) before gate 5 (calibration).
- Idempotent `prediction_id` means re-running the same scanner produces no duplicate log rows.

Negative:

- Adds a new module that both opportunity and attribution paths must call into. Both paths accept this coupling.
- DESIGN.md §6, §7 and FOLDER_MAP.md require updates (handled in P9-01 implementation cycle).

Risks:

- If `PredictionRecord` is too rigid for both paths, future extension friction. Mitigated by the `extra: dict` field with documented per-path keys.
- If JSONL is later replaced by SQLite, all callers will need updating. Mitigated by hiding storage details behind the writer interface; only `append_prediction` / `read_predictions` are public.
- If a caller forgets to call `append_prediction`, §8.6 silently regresses. Mitigated by integration tests at the panel level that assert the JSONL file gained the expected rows.

## Alternatives Considered

- **Per-path storage** (one JSONL for attribution, one for opportunity). Rejected: forces P9-03 to merge formats and forces P9-02 to scan two roots.
- **SQLite from day one**. Rejected: JSONL is simpler, append-only, human-inspectable, and consistent with existing `reports/` pattern. SQLite remains an option behind the writer interface if scale demands it.
- **Defer to per-task implementation in P7-06 / P8-05**. Rejected: creates the five-implementation fragmentation problem and violates Rule 2's single-source principle.

## Out of Scope

- Outcome joining (P9-02).
- Calibration math (P9-03).
- Human-readable alerts (P9-04).
- Live broker execution gate (P9-06).
- Migration from JSONL to SQLite.

## References

- `docs/02_GOVERNANCE.md` §8.6, §9.4, §10.
- `docs/00_DESIGN.md` §6.9 (added by P9-01), §7 (updated by P9-01).
- `docs/03_FOLDER_MAP.md` (updated by P9-01).
- `src/hot_theme_rotator/attribution/point_in_time_adapter.py` — `snapshot_id` pattern reused.
- `src/hot_theme_rotator/attribution/baseline_decision_score.py` — `model_version` constant reused.
- `reports/backtests/no_trade_diagnostics_2026-05-21.md` — current zero-entry blocker context.
