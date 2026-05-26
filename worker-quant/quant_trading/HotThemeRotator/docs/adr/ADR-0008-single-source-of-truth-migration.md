# ADR-0008: Single Source of Truth Migration (HotThemeRotator)

## Status

Proposed 2026-05-26. Cutover day T to be confirmed by user (recommended end of W2, 2026-06-08, after P10-21 implementation lands). On T this ADR becomes Accepted and ADR-0005 is marked Superseded by ADR-0008.

## Context

ADR-0005 (2026-05-24) established HotThemeRotator as a read-only consumer of `Project_optimized/japan_market.db`. `data/position_adapter.py` reads the live `etf_buyhold` strategy positions and `account_snapshots` from the sibling project, never writing back.

Two developments since make ADR-0005 insufficient as a long-term arrangement:

1. **User stated direction (2026-05-26)**: HotThemeRotator becomes the sole project the user interacts with day-to-day. Project_optimized continues running for live execution infrastructure but should not be edited daily by the user.
2. **Position update friction**: the user has manual fills to record (e.g., 2026-05-25 SELL 1306.T 400 @ ¥417.6). The current architecture forces the user into Project_optimized's `import_fills.py` flow, which is designed for machine-decision pipelines (`decision_runs → orders → fills`), not for manual single-trade entry. The friction violates the "operator-friendly" requirement set by the user.

A solution that lets HotThemeRotator accept fills directly requires resolving the dual-source-of-truth problem: if both `Project_optimized.positions` and a HotThemeRotator-local table claim ownership of live positions, they will drift the first time either side is updated.

## Decision

Migrate HotThemeRotator from read-only consumer to single source of truth for the user's live portfolio, via a one-shot cutover.

### 1. Cutover model (one-shot, atomic)

On cutover day T:

a. Snapshot Project_optimized's `etf_buyhold` positions and `account_snapshots` at end-of-day T.
b. Translate the snapshot into a sequence of `migration` journal entries (one `cash_event` for cash opening balance + one fill-equivalent entry per held symbol carrying opening qty and `avg_cost`).
c. Append these `migration` entries to HotThemeRotator's portfolio journal as the only journal content on T.
d. Disable Project_optimized's portfolio-writing daemons (manual stop or via flag).
e. `data/position_adapter.py` is removed from dashboard runtime path; the file is preserved for one cycle in case rollback is needed, then deleted in W3.

After T: all portfolio events flow only to HotThemeRotator. Project_optimized's `positions` and `account_snapshots` tables are frozen and visible only as historical archive.

### 2. Schema (implemented by P10-21)

HotThemeRotator owns a single append-only journal:

- File convention: `reports/portfolio/journal/{trade_date}.jsonl`
- Entry types:
  - `fill`: trade event with side / qty / price / fee.
  - `cash_event`: non-trade cash flow with reason enum (per Rule 14.7).
  - `corporate_action`: split, merger, ticker change. Out of scope for P10-21; placeholder reserved.
- All entries carry: `entry_id`, `ts` (JST ISO), `source` (Rule 14.3 enum), `note`.
- `positions` and `cash_balance` are derived views over the journal, never persisted (Rule 14.1).
- `entry_id` is deterministic: `sha256(ts|symbol|side|qty|price|source|note)` truncated to 16 hex chars. Duplicate IDs are rejected at write time.

### 3. Boundary contracts

- ADR-0005 is superseded on T. Its negative consequence "tight coupling to Project_optimized schema" is resolved by removing that read.
- Rule 3 advice-only is preserved — manual fill entry is the user telling HotThemeRotator what they did externally via their broker, not HotThemeRotator placing a trade.
- §10 gate 8 (live trading) is not relaxed. Manual fills logged here reflect trades the user already executed.
- Section 14 of `02_GOVERNANCE.md` (this ADR's operational counterpart) becomes binding on T.

### 4. Rollback plan

If post-cutover defects make HotThemeRotator unusable for portfolio entry within T+7 days:

a. Re-enable Project_optimized's portfolio daemons.
b. Replay any HotThemeRotator manual fills entered post-T into Project_optimized's `fills` table.
c. Revert this ADR's status to "Proposed", restore ADR-0005 as authoritative.

After T+7 days, rollback is no longer a routine option — drift in HotThemeRotator's journal exceeds what manual replay can sync.

## Consequences

Positive:

- One place to record fills, one place to check portfolio truth.
- Manual entry UI (P10-23) becomes the user's daily interaction surface, replacing CSV-and-CLI pipelines for everyday operations.
- Calibration outcome verification (P9-02 / Rule 14.6) gets clean source attribution: `manual` vs `paper` vs `migration` is explicit.
- Project_optimized is freed from being the user's daily editing target; it continues running as historical data infrastructure.

Negative:

- One-shot migration risk: if T-day snapshot is wrong, all subsequent derived state is wrong. Mitigated by replaying `position_adapter.load_portfolio_state()` and diffing NAV against expected.
- Historical positions before T are accessible only via the frozen Project_optimized DB, not via HotThemeRotator's journal. The journal starts at T.
- Removing `position_adapter.py` from runtime breaks any code path that imports it. P10-22 must audit and remove imports.

Risks and mitigations:

- **Risk**: T-day snapshot picks up a stale or partial position. **Mitigation**: T-day migration script reads Project_optimized's `positions` after `build_positions.py` has run for T; verify by replaying `position_adapter.load_portfolio_state()` and comparing NAV against the broker statement.
- **Risk**: User enters a manual fill on T while migration is in progress. **Mitigation**: cockpit UI shows "migration in progress, manual entry disabled" banner for T's session until P10-22 emits a `migration_complete` marker.
- **Risk**: Project_optimized's `etf_buyhold` strategy receives a new fill via `import_fills.py` after T. **Mitigation**: rename or comment-disable Project_optimized's daily portfolio jobs on T; document the freeze in Project_optimized's own README.
- **Risk**: P10-21 ships with a bug that produces non-deterministic derived positions. **Mitigation**: Rule 14.1 mandates determinism tests; CI must guard.

## Alternatives Considered

- **Keep ADR-0005, add a write-back path from HotThemeRotator manual entry into `Project_optimized.fills`**: rejected. Two-system writing creates dual writers; Project_optimized's `decision_runs → orders → fills` schema requires synthetic run_ids and orders to make manual fills fit, which is more complexity than just owning the state.
- **Dual-source with HotThemeRotator authoritative for manual entry and Project_optimized authoritative for paper/algo**: rejected. Requires merge logic for derived `positions`, opens reconciliation drift, defeats the "one truth" goal.
- **Postpone manual entry until Project_optimized retires entirely**: rejected. User has fills to enter now (2026-05-25 SELL 1306.T 400 @ ¥417.6); waiting is not operator-friendly.
- **Replicate Project_optimized's full `decision_runs → orders → fills` schema in HotThemeRotator**: rejected. That schema serves machine pipelines; manual entry needs ~5 fields, not three coupled tables.

## Out of Scope

- Schema for paper-trading fills (P9-05 still owns).
- Schema for corporate actions (splits, mergers, ticker changes). Future work; placeholder entry type reserved.
- Multi-strategy portfolio decomposition. `etf_buyhold` is the only live strategy currently; multi-strategy support deferred until justified.
- Broker integration / auto-execution. Rule 3 + §10 gate 8 forbid.

## References

- `docs/adr/ADR-0005-project-optimized-data-dependency.md` (the ADR being superseded on T).
- `docs/02_GOVERNANCE.md` Section 14 (operational rules; binding on T).
- `docs/02_GOVERNANCE.md` Rule 3 / Rule 8.6 / Rule 9.4 / §10 gate 8 (preserved).
- `docs/01_TASKS.md` P10-21 (journal schema + record_fill), P10-22 (migration snapshot script), P10-23 (manual entry UI).
- `Project_optimized/japan_market.db` (frozen on T).
- `src/hot_theme_rotator/data/position_adapter.py` (runtime-removed on T+).
