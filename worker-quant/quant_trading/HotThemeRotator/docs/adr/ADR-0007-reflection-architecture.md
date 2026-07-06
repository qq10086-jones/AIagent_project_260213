# ADR-0007: Reflection System Architecture (Policy Replay, Not Causal Identification)

## Status

Accepted 2026-05-25, post Codex cross-discipline review (senior quantitative trader + mathematician + statistician + automatic control theory professor).

## Context

User asked for design of P11 reflection system as automatic-control feedback loop. Initial proposal (Phase 0 v3 draft) used "Pearl do-calculus" for Layer 3 counterfactual replay and "Shapley value" for Layer 4 module attribution. User then asked Codex to review the design as four professional roles. Codex review rejected both terminology choices and identified one missing foundational component.

### Codex critique (summary)

1. **"Pearl do-calculus" mislabel** — Pearl 2009 framework requires explicit causal graph + intervention semantics + adjustment assumptions, none of which exist for an opportunity scanner with config mutations. Actual mechanism is **policy replay / off-policy evaluation / counterfactual backtest**.
2. **"Shapley value" misapplication** — sequential pipeline (data → scanner → filter → alert → notifier) does not satisfy cooperative game semantics. Coalition value `v(S)` is ill-defined for arbitrary subset like "scanner + alert without filter or data".
3. **CUSUM threshold h=4σ not defensible** — Montgomery SPC requires Average Run Length (ARL) analysis under no-change baseline, not generic σ multiples. Multi-KPI + autocorrelation + non-Gaussian returns cause false-positive explosion at naive σ thresholds.
4. **Missing PIT Observability Ledger** — Codex's #1 missing component. Without recording full PIT state at decision_cutoff (universe, watchlist, filters active, source freshness, alert budget state, silent_queue, user action state, missing-data reasons, config version, model versions), reflection becomes a "polished hindsight machine".
5. **Counterfactual validity must carry explicit class enum** — `exact_replay` / `partial_replay` / `universe_reconstructed` / `price_only_replay` / `invalid`. Output language MUST be conditional: "under reconstructed universe U and config C, this would have appeared", NEVER "system would have alerted you".

## Decision

Accept Codex review in full. P11 architecture revised as follows.

### 1. Layer 0 added — PIT Observability Ledger (P11-00)

Foundation layer. Records all decision-cutoff state. Without it, all higher layers produce invalid counterfactuals. P11-00 MUST complete before any L1-L5 work.

`PitSnapshot` fields (minimum viable):

- `decision_cutoff` (ISO ts)
- `candidate_universe` (set of tickers eligible at that time)
- `watchlist` (set of tickers user is tracking)
- `active_filters` (config hash + summary of which rules apply)
- `source_freshness` (dict per source: data_ts vs wall_ts)
- `alert_budget_state` (used / remaining for that day)
- `silent_queue_count` (suppressed alerts at that moment)
- `user_action_state` (last user action timestamp)
- `missing_data_reasons` (dict: source → reason)
- `config_version` (git hash of `configs/`)
- `model_versions` (dict: model_name → version_string)
- `shadow_panel` (random sample of K non-alerted candidates)

Stored at `reports/observability/pit/{trade_date}/{snapshot_id}.json`.

### 2. Layer 3 renamed "Policy Replay Engine"

Mechanism is off-policy evaluation, not Pearl do-calculus. Output language constrained:

- Allowed: "under reconstructed state U and config C, this would have appeared"
- Allowed: "this module blocked the candidate"
- Forbidden: "caused the miss"
- Forbidden: "would have alerted you"

Every replay output MUST carry `counterfactual_validity` enum.

### 3. Layer 4 renamed "Root Cause Analysis"

Method: **structured ablation + funnel loss decomposition + stale-data attribution + module diagnostics**. NOT Shapley.

- Sequential ablation: walk pipeline in order, measure marginal recovery under each module's intervention
- Funnel loss: count candidates lost at each stage (eligible → scored → not-filtered → alert-triggered → alert-pushed → user-acted)
- Stale-data attribution: identify staleness as direct contributor when freshness > threshold
- Output: ordered list of contributors with `marginal_recovery` metric, NOT "Shapley %"

### 4. Layer 2 thresholds derived from ARL bootstrap

Block bootstrap on historical system days targeting desired `ARL_0` (default 100 days false-alarm-free), not generic σ multiples. Multi-KPI correction via Bonferroni or Holm.

### 5. Rule 13 expanded to 17 sub-rules

Original 13.1-13.5 retained. Added per Codex review:

- 13.6: Proposal metadata (`evidence_class` / `sample_size` / `confidence_interval` / `counterfactual_validity`) required
- 13.7: Parameter proposals require pre/post backtest on holdout window or rolling-origin backtest
- 13.8: No single-event proposals (NDK panic-trigger forbidden)
- 13.9: Rejected proposals logged with rejection reason
- 13.10: Meta-reflection triggers when proposals repeatedly rejected / expire / post-acceptance failed

Added per R2 hardening review on 2026-05-27:

- 13.11: `counterfactual_validity` controls allowed action, not just wording
- 13.12: LLM cannot originate proposals; actions must come from deterministic L3/L4 layers
- 13.13: Proposal tiering by blast radius
- 13.14: Accepted parameter changes start in shadow / canary before active use
- 13.15: Same-target parameter changes have a 14-day anti-oscillation cooldown
- 13.16: Expiry/rejection reasons are machine-readable and meta-reflection distinguishes operator context
- 13.17: Reproducibility metadata (`source_trace_ids`, config hashes, outcome window, denominators) required at intake

### 6. Statistical methods reorder

- Bernoulli outcomes (hit/miss) → beta-binomial intervals or sequential likelihood ratio, NOT Gaussian
- Returns → robust statistics (median, MAD, winsorized mean, block bootstrap), NOT Gaussian residuals
- Sample size tiers: n≥30 investigate / n≥100 directional language / n≥300 parameter change proposal

## Consequences

### Positive

- Epistemic honesty — claims match what evidence supports
- Cleaner separation: detection (L2) vs analysis (L3-L4) vs narration (L5)
- PIT ledger benefits beyond reflection: ADR-0006 backdated calibration becomes more defensible; audit trails for any future broker integration; P10-13 inputs simplified
- Codex's "polished hindsight machine" failure mode is avoided

### Negative

- Lower academic prestige (no "Pearl" / "Shapley" labels)
- Phase 1 timeline extends from 6-7 weeks to 10-12 weeks (Codex realistic estimate)
- L0 (PIT Observability) is heavy infrastructure with no immediate user-visible value — risk of being deprioritized in favor of "more visible" features

## Risks + Mitigations

- **Risk**: Without Shapley %, user asks "what % does each module contribute" and we have to answer "marginal recovery under intervention" which is less catchy.
  - **Mitigation**: per-module ablation gives richer + valid answers (recovered candidates / recovered alerts / recovered P&L per intervention); document the rename rationale in user-facing reflection reports; cite ADR-0007 in any UI explanation.
- **Risk**: PIT ledger scope-creep (everyone wants to log everything).
  - **Mitigation**: P11-00 acceptance criteria pin the minimum viable schema; new fields require explicit Rule 4 change with backfill plan.
- **Risk**: CUSUM ARL bootstrap requires historical system days that don't yet exist.
  - **Mitigation**: use synthetic injection / parametric bootstrap initially; transition to empirical block bootstrap as production data accumulates over 3+ months.
- **Risk**: User accepts proposals based on small sample size despite Rule 13.3 tiers.
  - **Mitigation**: P11-06 Human Decision Gate enforces sample-size tier at intake; UI shows tier badge prominently next to proposal.
- **Risk**: LLM (L5) regex-misses a hedged probability phrase.
  - **Mitigation**: Rule 13.4 inherits Rule 8.3.1 regex enforcement; expand keyword list iteratively as new evasions discovered.
- **Risk**: Reflection starts tuning parameters against noisy windows and oscillates.
  - **Mitigation**: Rule 13.11 limits action by validity class; Rule 13.14 forces shadow/canary; Rule 13.15 enforces same-target cooldown.
- **Risk**: Expired proposals are mistaken for generator failure when the user was simply unavailable.
  - **Mitigation**: Rule 13.16 adds `expiration_reason`; meta-reflection ignores operator-context expiry.

## Alternatives Considered

- **Keep Pearl + Shapley with caveats** — rejected. Codex correctly identified these as overclaiming causality; mitigation caveats would be footnotes nobody reads.
- **Skip PIT Observability, rely on existing `decision_log/`** — rejected. Decision log captures prediction output, not the full state that produced it. Reflection without full PIT = hindsight bias amplifier.
- **Use causal forests (Athey & Wager 2019)** — rejected. Config choices were never randomized in production; causal forests would manufacture precision.
- **Use BOCPD (Adams & MacKay 2007) instead of CUSUM** — deferred. BOCPD adds modeling burden (likelihood family, hazard function, prior, posterior interpretation). Better path: CUSUM/EWMA first; BOCPD later only for monthly regime-level reflection (per Codex).
- **Use Heckman correction (1979) for selection bias** — rejected. Heckman requires valid exclusion instrument we don't have. Codex's pragmatic alternative: log denominators (full universe + filter rejections + suppressed alerts + shadow panel) instead of statistical correction.

## Out of Scope

- True causal identification (would require RCT / IV / valid causal DAG with do-calculus rigor — none available for opportunity scanner)
- Online auto-update of parameters (Rule 13.1 mandates proposals only, never auto-apply)
- LLM-generated counterfactual prices (Rule 13.2 forbids; counterfactual prices MUST be real historical OHLC via `LegacyDailyPriceFetcher`)
- True real-time tick data for P10-19 (paid only; out of "no money" constraint per user)
- "Why" / "intent" reasoning by LLM (Rule 8.3.1 + 13.4 — LLM does narrative synthesis of evidence already computed, not generative speculation)

## References

- Codex review session 2026-05-25 (verbatim transcribed in `PROJECT_STATUS.md` Change Log)
- Page, E.S. (1954) "Continuous Inspection Schemes" — CUSUM original
- Montgomery, D.C. *Introduction to Statistical Quality Control* — SPC textbook (ARL methodology)
- Adams, R.P. & MacKay, D.J.C. (2007) "Bayesian Online Changepoint Detection" — deferred for regime-level
- Pearl, J. (2009) *Causality* — do-calculus framework (rejected for our setup)
- Hernán, M.A. & Robins, J.M. (2020) *Causal Inference: What If* — causal identification assumptions
- Wager, S. & Athey, S. (2018); Athey, S. & Wager, S. (2019) — causal forests (rejected)
- Lundberg, S.M. & Lee, S-I. (2017) — SHAP (rejected for sequential pipeline)
- Heckman, J.J. (1979) — sample selection correction (rejected; logging preferred)
- `docs/02_GOVERNANCE.md` Section 13 (Rule 13.1-13.10) — discipline enforcement
- `docs/01_TASKS.md` Milestone P11 (P11-00..07) — task definitions
- `docs/adr/ADR-0003-decision-log.md` — predictions infrastructure that L1 trace logger extends
- `docs/adr/ADR-0006-backdated-calibration.md` — also depends on P11-00 PIT ledger
- `src/hot_theme_rotator/observability/` — to be created in P11-00
- `src/hot_theme_rotator/reflection/` — to be created in P11-01 through P11-07
