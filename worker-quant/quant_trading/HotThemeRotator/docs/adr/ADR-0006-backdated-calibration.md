# ADR-0006: Backdated Calibration as One-Time Bootstrap for Win Rate Display

## Status

Accepted 2026-05-25. User explicitly chose Option C after reviewing Options A (合规推进 wait ~5 months) and B (deep-dive first, calibration later).

## Context

User vision (recorded 2026-05-25): build a personal quantitative advisory system that can answer "在 X 价位买的胜率" / "在 Y 价位卖的胜率" for any ticker, and proactively push high-win-rate strategies.

Hard constraint from existing governance:

- Rule 8.3 forbids labeling any number as `buy_win_prob` / `sell_win_prob` / `hold_win_prob` unless produced by a calibrated historical model with explicit `model_version`.
- Rule 9.4 forbids labeling any number as a true win rate until ≥100 paired (PredictionRecord, OutcomeRecord) samples exist.
- P9-01 / P9-02 / P9-03 (calibration math layer) are done, but `reports/predictions/` is empty in production. No daily scheduler has ever run the scanner to persist forward predictions. `reports/outcomes/` is empty in lockstep.

Naive forward path (Option A): start a daily scheduler today, accumulate ≥100 samples, then unlock win-rate display.

- Best case: 10 candidates/day × 10 days = 100 samples in ~2 weeks after T+5 outcome join window completes — so roughly 3 weeks to first calibration report.
- Realistic case: less than 10 candidates/day pass scanner filters, so 4-8 weeks.
- User's vision requires win-rate-style output to feel useful at all, not after 4-8 weeks.

Bootstrap alternative (Option C, this ADR): generate `PredictionRecord` instances retroactively from the historical artifacts that already exist on disk —

- `Project_optimized/selected_tickers.json` daily snapshots (when archived)
- `Project_optimized/japan_market.db.factor_signals` historical rows
- `Project_optimized/japan_market.db.daily_prices` for outcome join
- already-implemented `opportunity_scanner` recomputed against historical inputs

Each retroactive prediction is joined against truly-future OHLC (from the perspective of the synthesized `decision_cutoff`) via the existing `P9-02 outcome_join` path, producing real `OutcomeRecord` instances. The math layer (`P9-03 calibration`) does not need to change — it already accepts any pair stream.

## Decision

A one-time bootstrap calibration tool is permitted under strict conditions. The tool synthesizes `PredictionRecord` instances from historical snapshots, joins them against forward-in-historical-time OHLC, and feeds the resulting paired stream into the existing calibration reporter. The synthesized predictions are explicitly flagged so they can never be confused with live forward predictions and can never inform a new trade.

### Required flags on every backdated prediction

- `PredictionRecord.extra["backdated"] = True`
- `PredictionRecord.extra["live"] = False`
- `PredictionRecord.model_version` carries the suffix `-backdated` (e.g. `opportunity-v0-backdated`) so any downstream consumer that filters by model_version sees the bootstrap nature explicitly.
- Generation tool is the dedicated `tools/backdated_calibration_bootstrap.py` — no other module is allowed to emit backdated predictions.

### Required temporal invariants (PIT relaxation, not abandonment)

For each synthesized prediction with `decision_cutoff = D`:

- Every input feature must have `available_ts < D`. The relaxation is that the operator is generating the record after wall-clock T >> D, not that the inputs themselves can leak the future.
- The matched outcome window uses `daily_prices` bars with `asof > D` only. No same-day or earlier bars.
- The synthesized `decision_cutoff` must strictly predate the earliest outcome bar by at least 1 trading day.

### Calibration report changes

- `CalibrationReport` gains an `evidence_origin` field with values `"live"`, `"bootstrap"`, or `"mixed"`.
- A report whose `evidence_origin == "bootstrap"` may still reach `status="calibrated"` if `sample_count >= min_samples_required` AND `brier_score` / `log_loss` are present.
- UI surface (calibration badge in dashboard) must visually distinguish bootstrap-derived calibration from live-derived calibration. A bootstrap-only badge says e.g. "校准样本 (历史回填) · 100/100" with an explicit "bootstrap" pill, never plain "100/100 校准完成".
- Once live samples reach `min_samples_required`, the dashboard MUST switch to displaying live calibration only. The bootstrap report is retained on disk but not displayed.

### Sunset clause

Bootstrap calibration display is automatically retired the first day live evidence_origin reaches `min_samples_required`. The bootstrap tool itself may continue to exist for audit but produces no UI effect after that day. This sunset is a hard governance commitment, not an option.

## Consequences

Positive:

- First real Brier / log loss / calibration bins available within 1-2 days of running the bootstrap, instead of 4-8 weeks.
- User's Pull-mode vision (give a ticker → see a real win-rate-anchored ladder) can be tested with real evidence shape, not placeholder.
- Push-mode vision (system surfaces high-win-rate strategies) can start being meaningful immediately.
- The bootstrap pathway exercises the entire P9-01 → P9-02 → P9-03 pipeline against real data volumes early, surfacing fail-closed / schema / scaling issues that would otherwise be discovered slowly over weeks.

Negative:

- Strict §8.2 PIT semantics are softened: while inputs still respect `available_ts < decision_cutoff`, the human operator runs the generator with full knowledge of what happened after, which creates subtle ways to bias prediction generation (model_version choice, feature set choice, threshold choice).
- Two parallel calibration regimes (bootstrap + live) exist during the transition window, increasing surface area for "which number is real?" confusion. Mitigated by the explicit `evidence_origin` field and the visual badge separation.
- Risk that bootstrap "feels good enough" and the team stops investing in the live forward scheduler. Mitigated by the sunset clause being mandatory: once live ≥ threshold, bootstrap display dies.

## Risks and Mitigations

- **Risk**: Operator unconsciously selects historical windows / model variants that look favorable, biasing calibration.
  - **Mitigation**: bootstrap tool must process a contiguous historical window (no cherry-picking by date) and emit a `bootstrap_provenance.json` recording: window start/end, total snapshots considered, snapshots excluded with reason, model_version used, scanner config hash.
- **Risk**: User confuses bootstrap "win rate" with live forward "win rate" and trades on it.
  - **Mitigation**: dashboard calibration badge separates origins; advisory copy (`docs/02_GOVERNANCE.md` Rule 8.2.1) explicitly forbids treating bootstrap evidence as forward evidence; `tools/morning_briefing.py` shows `evidence_origin` next to every win-rate-style number it ever prints.
- **Risk**: Schema drift in `factor_signals` or `selected_tickers.json` between bootstrap window and today produces incomparable features.
  - **Mitigation**: bootstrap tool pins the schema columns it consumes; refuses to proceed if a column is missing or has changed type; refuses to proceed if `Project_optimized/selected_tickers.json` history isn't snapshotted (i.e., requires the operator to have actually saved daily snapshots, not back-compute today's screener against historical prices).
- **Risk**: Bootstrap survives past sunset, becoming a permanent crutch.
  - **Mitigation**: sunset is enforced in code: dashboard calibration badge has a runtime check `if live_count >= threshold: show_live_only()` — no operator flag can keep showing bootstrap once live qualifies.
- **Risk**: Lookahead bias creeps in via the scanner config — operator picks a scanner config in 2026 that was tuned with knowledge of 2025 outcomes, then runs it on 2025 data.
  - **Mitigation**: bootstrap config must use a scanner config hash that matches a config file committed to git on or before the bootstrap window start. The bootstrap tool refuses to proceed if the operator-provided scanner_config_hash does not appear in `git log -- configs/scanner.yaml` before the bootstrap window start date.

## Alternatives Considered

- **Option A: Pure forward-going calibration (no bootstrap).** Rejected by user as too slow to deliver visible value. Documented here so the trade-off is clear in writing.
- **Option B: Build deep-dive Pull-mode features first, calibrate later.** Rejected by user as leaving the win-rate language gap unsolved; deep-dive without win-rate numbers feels like a research report, not an advisory system.
- **Option D: Loosen Rule 8.3 / 9.4 to allow uncalibrated scores to be called win rates.** Considered and explicitly rejected — would invalidate the entire governance basis for the project.
- **Option E: LLM-generated win probability.** Explicitly forbidden by Rule 8.3 and reaffirmed by user 2026-05-25 (LLM is for narrative synthesis only, never probability output).

## Out of Scope

- Backfilling outcomes for predictions the system never actually made. The bootstrap tool generates synthetic `PredictionRecord` and real `OutcomeRecord` pairs; it does not pretend a different model would have predicted differently in the past.
- Backdating live trades. No `fills` / `orders` / paper-trade records are ever synthesized — this ADR only covers calibration evidence.
- Changing `min_samples_required` from the existing default (100). The sample threshold remains identical for bootstrap and live.
- Allowing other generation tools to emit `backdated=True` records. Only `tools/backdated_calibration_bootstrap.py` may do so.
- Re-running bootstrap on a rolling basis. Bootstrap is conceptually one-time. If a future need arises (e.g., new model_version that needs its own bootstrap), it must come back through this ADR or a successor.

## References

- `docs/02_GOVERNANCE.md` Rule 8.2.1 (this ADR's governance landing point).
- `docs/02_GOVERNANCE.md` Rule 8.3 / Rule 9.4 (preserved; not relaxed).
- `docs/adr/ADR-0003-decision-log.md` (`PredictionRecord` / `OutcomeRecord` schemas this ADR extends with origin flags).
- `src/hot_theme_rotator/calibration/reporter.py` (`build_calibration_report` — to be extended with `evidence_origin`).
- `tools/backdated_calibration_bootstrap.py` (to be created under task P10-13).
- `docs/01_TASKS.md` Milestone P10 (Personal Advisory System, this ADR's parent program).
