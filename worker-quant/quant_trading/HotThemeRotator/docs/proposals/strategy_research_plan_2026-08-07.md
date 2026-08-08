# Strategy Research Plan (P34)

Date: 2026-08-07
Status: **PROPOSAL — advice-only (Rule 3).** This document changes no config, no
mandate, no weight, no user-facing output. Every lane below is research/shadow
until it passes the locked verification protocol (Rule 8.2.3 / Rule 16.0 family).
Scope: fix the research agenda for the next two quarters — which market, which
signals, in what order, verified how.
Non-scope: capital deployment, Sleeve B enlargement (frozen per P31 rename),
screener weight changes (Rule 4 + walk-forward required), broker migration.

Provenance: three review rounds on 2026-08-07 (initial evidence survey → two
adversarial external reviews → line-by-line repo verification). Consensus is
recorded in this file; the memory index carries a pointer. Corrections adopted
along the way are folded in silently — this document supersedes all three drafts.

---

## 1. Market decision: stay in Japan; US is a read-only research lane

**Decision rule applied:** the system's own Rule 16.0 hurdle `IC > τ · c_rt / σ_r`
(`research/cost_model.py`, `τ = 0.7` per `evidence_review_63d.py:78`).

| Channel | Commission (one-way) | Round-trip c_rt | Required IC (illustrative σ₆₃ = 0.104) |
|---|---|---|---|
| JP lot stocks, SBI/Rakuten | ¥0 (ゼロ革命/ゼロコース, permanent) | ~10–25 bp (tick crossing only) | **0.0067–0.0168** |
| JP S株 (SBI) | ¥0, **market-only**, 4 submission windows → 3 same-day slots | auction slippage, **unmeasured** | unknown — gap, see §5 |
| JP かぶミニ (Rakuten) — realtime | ¥0 + 0.22% spread | 44 bp | 0.0296 |
| JP かぶミニ (Rakuten) — 寄付 | ¥0, spread not charged on the opening-auction route | ~lot-like | ~0.0067–0.0168 |
| US via SBI/Rakuten | 0.495%, max $22 | 99 bp | 0.0666 |
| US via moomoo JP | 0.132% | 26 bp | 0.0175 |

**Arithmetic, corrected 2026-08-08.** `0.7 × 0.0010 / 0.104 = 0.0067` (an earlier
draft printed 0.010 for the 10 bp row). The US-vs-JP-lot ratio is therefore
**≈4.0× against the 25 bp end and ≈9.9× against the 10 bp end**, not the "4–6.6×"
previously stated — that range had silently used a 15 bp low end while the row
said 10–25 bp.

**The $22 maximum is never reached at this account size**, so the *percentage*
rate applies in full on every order; the cap is irrelevant here rather than
binding. (Phrasing corrected — the earlier "the cap always binds" invited the
opposite reading.)

σ_r cancels in the JP-vs-US ratio, so the ratio survives even though the absolute
IC column is illustrative pending `reports/research/cost_model.json`. **The
absolute column is NOT a validated hurdle** — σ₆₃ = 0.104 is a schema example,
not a measured dispersion (§5).

**Provisional status (O-1).** Every fee above is broker- and course-specific.
Until the owner supplies the account facts in §7/O-1, the cost advantage of
staying JP is **provisional**; what is *not* provisional is the infrastructure and
research-focus decision, which the non-cost arguments below already carry.

Non-cost arguments, all pointing the same way:
- **Coverage inefficiency lives in Japan**: ~2 sell-side estimates per listed
  company (US: 7); >70% of TOPIX Small has zero coverage; worst coverage of 23
  MSCI developed markets. McLean–Pontiff (JF 2016): published anomalies decay
  post-publication, most where arbitrage is cheap — i.e. US large caps.
- **FX**: base currency is JPY. Hedging USDJPY is not *impossible* — it is
  uneconomic at ¥400k and within this system's scope (no FX instruments, no
  derivatives lane), and every available hedge introduces its own basis and
  rolling risk. So unhedged USD exposure would dilute the −75% tolerance /
  ¥100k kill-switch semantics in practice, which is the operative point.
- **Hours**: US cash session is 22:30/23:30–05:00/06:00 JST; advice-only + manual
  execution means either night sessions or MOO/MOC (giving up price control).
- **Migration cost**: JPX calendar, TDnet, EDINET, S株 overlay, tick-cost model
  are all JP-specific; the 8-26 protocol gaps would be multiplied by two markets.

**The one input that could flip this:** if the actual channel is IBKR-class
(≈5–10 bp round trip), the cost argument collapses and only the four softer
arguments remain. Naming the broker is not enough — SBI 特定口座, SBI NISA,
Rakuten, moomoo and IBKR price the same order differently. **Owner decision O-1**
therefore needs all of: `broker`, `account_type` (特定/一般/NISA),
`fee_course` (ゼロ革命 / ゼロコース / other), `tax wrapper`,
`currency_settlement_mode` (円貨 vs 外貨決済), `typical_order_notional`, and
`whole-share vs fractional requirement`. Until then the plan assumes
SBI/Rakuten and marks the cost conclusion provisional.

US lane (kept, last priority): SEC EDGAR companyfacts/full-text — free, no key,
PIT by filing date. Read-only, replication-of-the-same-signal only, no execution,
no capital, SKHY-ADR-lane pattern (read-only / fail-open / no edge language).

## 2. Research lanes, in priority order

Evidence grading used below: **[H]** = hypothesis with literature support, not a
validated tradable effect; **[V]** = validated in THIS system's forward data.
Nothing below is [V] today.

### T1 — Buyback-announcement drift [H] — first priority
- Literature: positive announcement CARs in Japan with post-announcement upward
  drift not observed in the US comparison studies; a 2025 *Pacific-Basin Finance
  Journal* **registered report** tests whether costly-arbitrage explains
  long-horizon buyback returns in Japan. A registered report pre-declares design
  and hypotheses — it supplies **design credibility, not a positive result**, and
  must never be cited as evidence the effect is validated. This stays [H].
- Catalyst tailwind: TSE cost-of-capital reform since 2023. **FY2025 buyback
  authorizations totalled ¥22.325tn, +18% YoY, a 5th consecutive record**
  (fiscal-year basis; calendar-2025 tallies read slightly differently, so the
  basis must be stated). An earlier draft cited "~¥14tn by December", which was
  an interim figure and is superseded. TSE has also said explicitly that one-off
  buybacks are not the intended end point, so "reform tailwind" is a statement
  about event *rate*, not about effect size.
- **Repo gap, corrected after measurement.** The earlier claim "zero hits for
  buyback across the repo" was scoped too narrowly and is withdrawn. Buyback
  keywords DO exist in `theme_detection/theme_detector.py` (`buyback_dividend`)
  and `free_web_opportunity_adapter.py` (`shareholder_return`) — but both
  **conflate buybacks with dividends**, which is precisely the contamination T1
  must exclude. The real defects, now measured on 2,344 stored disclosures
  (2026-06-30..2026-08-07):
  - `tdnet_parser._CATEGORY_RULES` had **no buyback rule**, and its `order` rule
    matches `株式の取得` — which 「自己株式の取得」 contains. **All 547 treasury
    disclosures were misfiled**: governance 276 / order 225 / other 30 /
    earnings 16.
  - No structured parser for resolutions, cancellations, or execution reports.
  - **`自己株式の処分` (disposal, 294 records — the largest subtype) is not a
    buyback at all**; counting it would put the sign backwards on the majority
    of the corpus.
  Fixed under P34-01a; see §6 and the smoke artifact.
- **Measured event rate: 20 resolutions, of which 15 uncontaminated → ~144/yr.**
  This is the honest sample-accrual input, and it is now the plan's frozen
  `expected_event_rate_per_year`.
- **Parser limitation, recorded at freeze:** amount/share caps and windows are
  absent from RSS titles in ~91% of treasury disclosures (confidence low 497 /
  medium 50 / high 0). Size-based strata therefore need PDF extraction and are
  **not** in the primary plan.
- Time-sensitivity: TDnet's public window is short (~1 month); Yanoshin backfill
  exists but real-time capture preserves PIT-grade timestamps. **Raw-event
  capture is P0 and irreversible; do not defer it behind anything.**
- Holding-period grid is a research grid, not a trading conclusion; it is
  registered as trials (5 horizons × 3 method strata = 15) in `P34_T1_v1`.
- Method heterogeneity is pre-declared: `auction` / `tostnet` / `method_unknown`
  are separate strata, because off-auction ToSTNeT buybacks and on-market
  purchases are not one population.

### T2 — Ownership-conditioned PEAD [H] — second priority
- Jinushi (TJAR vol.13; 2002–2020, 60,124 firm-years): aggregate Japanese PEAD
  decays over the period, but does **not** decay in low-foreign-ownership /
  high-individual-ownership firms. Mechanism (Arrowhead 2010, 3s→5ms; XBRL;
  institutional technology) is the paper's interpretation — correlational, not
  causal; the authors state the holder-structure≠trader limitation themselves.
- Our earlier disclosure-drift failure (P19-04: 277 directional events/yr, IC
  −0.17…−0.33, DSR 0.50) is explained first by "title regex ≠ true SUE", not by
  the unconditional-PEAD-is-dead story. T2 requires the full chain, not one
  field: PIT earnings-announcement timestamps; announcement-day reaction +
  ~60D returns; a defensible surprise measure (or Jinushi's unbiased
  return-regression design); annual PIT ownership snapshots with validity
  windows; size/liquidity controls and pre-declared conditioning buckets.

### T3 — E/P value lane [H, live-log running] — keep, do not touch composition
- **E/P 21D live-log, current artifact reading (2026-08-06): IC +0.04926,
  t +2.325, independent date clusters 27, maturity coverage 52%.** The earlier
  "+0.085 / t +3.1" is stale and is withdrawn. 63D is **0 matured of 2,216 rows**;
  2026-08-26 is a readiness check, NOT a verdict date.
- ⚠ **`n_obs_effective = 1` at 21D for BOTH E/P and B/P** under the artifact's
  own disjoint-blocks method. The t-statistics above are computed on overlapping
  rows, so neither should be read at face value as evidence strength. This is the
  single most important qualifier on this lane and it applies to the negative
  reading as much as the positive one.
- **B/P is prohibited from weighting**: live-log 21D IC **−0.0713, t −3.95**.
  That is a strong negative alert and sufficient grounds for a fail-closed
  freeze — but with `n_obs_effective = 1`, and with PIT, survivorship, cost and
  effective-sample protocols still unmet, it is **not a completed validation of
  negative alpha**. Correct standing: B/P does not enter weights, is not
  composited with E/P, stays in shadow, and is described as an alert rather than
  a settled result.
- Cross-sectional price momentum: frozen, not declared eternally dead. Asness
  (2011), stated in full: JP momentum standalone Sharpe ≈**0.03**, value
  ≈**0.71**, 50/50 ≈**0.65**, ex-post optimal 70/30 ≈**0.88**. So 50/50 does
  **not** beat value standalone — the correct reading is that weak momentum still
  carries diversification value, not that combining improves on value. Fama–French
  (2012): momentum present everywhere except Japan. Engineering conclusion for
  THIS system: **no standalone production weighting of the current cross-sectional
  momentum score; retain as shadow.** Not an eternal prohibition.

### T4 — TSMOM overlay for Sleeve A [H, weakest transfer] — shadow only
- Moskowitz–Ooi–Pedersen (2012) validates **diversified futures, long/short,
  volatility-scaled** TSMOM (58 instruments; positive predictability in all,
  significant at 5% in 52). Mapping that onto a single long/cash switch on a 2×
  JP equity ETF crosses three gaps — universe breadth, long/short vs long/cash,
  and vol-scaling — so **the original paper does not validate this
  implementation**. Whipsaw after sharp reversals is a mechanism risk *of our
  mapping*; the paper documents trend-reversal losses (e.g. 2009), which is
  related but not the same claim.
- Also: the paper's standard specification is 12-month lookback with 1-month
  holding. "12-1" is cross-sectional-momentum shorthand and should not be
  attributed to MOP.
- Therefore a **six-arm shadow comparison** (an earlier draft listed five and
  called it six): (1) buy & hold, (2) 12M time-series trend long/cash,
  (3) 10M SMA, (4) volatility target, (5) trend + volatility double gate,
  (6) **trend with a re-entry delay / whipsaw filter** — all reported net of
  1568.T fees, tracking difference, and gap risk.
- **No Sleeve A mandate change.** With **17/17 observed sessions below the
  authorized band** — an execution failure per the 2026-08-04 retrospective, not
  a design feature — a new timing rule must not become an ex-post
  rationalization of under-deployment.

### Opportunistic (not a core lane)
- Nikkei 225 constituent changes: direction supported (additions rise,
  deletions fall; JP price effects more persistent than S&P 500). Specific CAR
  figures previously quoted (5.70%/2.38%) are **withdrawn** pending table-level
  verification against the original papers. Review calendar is anticipatable;
  actual changes are facts only after announcement. Few events/yr, crowded —
  opportunistic shadow only.
- News heat: stays a re-ranker. Never promoted to alpha.

## 3. The opportunity gate ("score ≥ θ ⇒ opportunity") — design

**Classification:** a selective-prediction / reject-option decision rule
(Chow 1970; El-Yaniv & Wiener 2010). It is a strategy *component*; a full
strategy additionally fixes universe, PIT signal time, entry, size, holding
period, exit, costs, risk limits, benchmark, and a kill condition.

**Prior finding — P34-00 AUDIT COMPLETE, verdict `DORMANT` (2026-08-08).**
An earlier draft said `min_entry_score = 70.0` "already ships". **That was
asserted, not audited, and the audit says otherwise.**
`tools/audit_gate_reachability.py` builds the static import graph and finds:
- score lineage: `entry_score = 0.30 × market_temperature.score
  + 0.70 × leader.leader_score`, then `× risk_weight_multiplier` (capped at 1.0)
  — a **different lineage** from `opportunity_scanner`'s hand-weighted score
  (which is persisted as `buy = score/100`, a stored fraction, not a probability).
- callers: `generate_signal` is called from exactly one place,
  `reporting/daily_pipeline.py:124`.
- reachability: `run_daily_pipeline` is imported **only by tests**
  (`tests/integration/test_daily_pipeline.py`,
  `tests/integration/test_signal_backtest_report.py`,
  `tests/unit/test_signal_engine.py`). **Zero** static paths from `tools/` or
  `api/`. The production orchestrator `tools/daily_routine.py` never touches it.
- artifact/UI influence: **none observed** — no artifact, report, or API route
  consumes `TradingSignal`.
- verdict: **legacy / dormant, test-reachable only.** Not shipping.

Consequences: there is **no live undeclared threshold strategy**, so **O-2 is
NOT a blocking owner decision** — it is an optional cleanup choice (keep as a
test fixture / isolate / delete). And the 70 must **not** be retroactively
described as pre-registered under any disposition; it is `legacy`.
Audit limits are stated in the artifact: static graph only, blind to dynamic
imports; `shipping` would be an upper bound on liveness, while `dormant` means
no static path exists.

**Two orthogonal fields** (an earlier draft mixed event state with model
validation state into one three-state enum — corrected):
```
candidate_status  : INSUFFICIENT_DATA | NO_CANDIDATE | CANDIDATE
validation_status : UNVALIDATED | VALIDATED | INVALIDATED
```
`CANDIDATE` means "crossed a pre-declared line, worth studying". It carries **no
expectancy claim**, and no user-facing surface may render `CANDIDATE` as a
recommendation, probability, or win rate. Every candidate emitted today is
`UNVALIDATED` by construction, and the two axes move independently: a signal can
be `INVALIDATED` while still producing `CANDIDATE` rows, which is exactly the
state a naive single enum would hide.

**Shadow prediction record** (frozen before outcomes; outcomes are a SEPARATE
append-only event keyed by `prediction_id` — never written back onto the
prediction, so a prediction cannot be edited once its outcome is known):
```json
{"prediction_id": "...", "symbol": "...", "universe_id": "...",
 "decision_cutoff": "...", "score_definition": "...", "model_version": "...",
 "model_hash": "...", "threshold": 0.0, "threshold_provenance": "...",
 "expected_trigger_rate": 0.0, "trigger_rate_estimation_window": "...",
 "horizon_days": 0, "entry_rule": "...", "benchmark": "...",
 "family_id": "P34_GATE_v1", "family_version": 1,
 "cost_profile_id": "...", "cost_profile_version": "...",
 "candidate_status": "CANDIDATE", "validation_status": "UNVALIDATED",
 "outcome_due_at": "..."}
```

**Verification metrics — EV after cost, never bare win-rate:**
`EV = p·avg_win − (1−p)·avg_loss − cost`. Report: trigger rate × effective
independent sample count; conditional net return vs non-trigger days, same-day
market, matched names, and a naive baseline; avg win / avg loss; drawdown/tail;
regime stability; date-cluster bootstrap CI; turnover and realized shortfall.
Precision–coverage and risk–coverage curves; PR over ROC for rare triggers
(Saito & Rehmsmeier 2015).

**Statistical honesty:** a gate is a *measurement and selection* mechanism, not
an improvement mechanism — it cannot add information the score lacks. Overall
AUC ≈ 0.46 does not mathematically preclude a locally valid tail, but
tail-hunting on history is a fresh multiple-testing window. Any θ examined must
be registered in the P34-05 registry under its own family (`P34_GATE_v1`).
**It must NOT be "fed into the P31 frozen family":** P31's count is a frozen
2026-08-06 historical snapshot and is never written to. Where a program-wide
conservative denominator is needed, `program_snapshot()` emits a NEW as-of
snapshot that ADDS the cited P31 count to the registry count (currently
15 + 100 = 115) while leaving P31's artifact untouched.
Selectivity slows validation: a gate firing 2×/mo accrues effective samples an
order of magnitude slower than an unconditional signal — declare
`expected_trigger_rate` up front for exactly this reason.

## 4. What the system can honestly claim today (product boundary)

| Capability | Status |
|---|---|
| Per-stock analysis | Usable — "research summary" grade (price/trend/RS/liquidity/holdings/disclosure/ladder); fundamentals thin |
| Investment advice | Discipline-type only (sizing, brackets, risk bounds, scenarios). Return-type advice is blocked by design (Rule 9.4/11.14) — correctly so |
| Recommendations + operating hints | Candidate list + conditional plan; NOT "validated recommendations". UI fields should read `research_priority / candidate / conditional_plan / evidence_status` |
| Auto-find high-win-rate opportunities | "Auto" exists; "high-win-rate" does not (calibration OOS Brier 0.2823 ≥ 0.25 random; forward AUC ≈ 0.46). The gate above is the honest path toward it |

`price_ladder.py` is a range heuristic, not a validated entry/stop model — the
words "optimal entry" / "recommended stop" must not appear; Sleeve mandates
override the generic ladder for held names.

## 5. Cost model producer (unblocks Rule 16.0 everywhere)

Contract stays **one file**, `reports/research/cost_model.json` — that contract
was just unified (P31+P33) after two tools diverged, and splitting it would
re-fragment it. **Schema v2 (`research/execution_profiles.py`, built) fixes the
real defect: v1 held ONE scalar `round_trip_cost`,** which cannot represent
channels whose costs differ by an order of magnitude. v2 adds
`execution_profiles`, and **the consumer must name an `execution_profile_id`**.
A profile that is absent or costless resolves `available=False` and **never
borrows another profile's number** — substitution is exactly how a signal priced
at lot cost gets executed as S株.

Provenance is now structured, not a bare string: `source`, `producer`,
`version`, `asof`, `sample_size`, `method` per field. Cost and `sigma_r` are
**different DGPs and keep separate provenance blocks** — a fill-aggregating
producer has no standing to assert a signal's return dispersion — while still
travelling in the one canonical file.

Dimensions (**S株 has NO limit orders, so there is no market/limit product for
it** — an earlier draft's cartesian split was wrong):
- **Lot stocks**: market/limit × session × book state; per-fill shortfall.
- **S株**: submission window → auction slot, reference-price deviation, realized
  shortfall. **Slot table corrected 2026-08-08** (the previous 13:30/15:00 times
  were stale on both the cutoff and the close):

  | Submission window | Executes | Session |
  |---|---|---|
  | 00:00–07:00 | 09:00 | 前場寄付 |
  | 07:00–10:30 | 12:30 | 後場寄付 |
  | 10:30–**14:00** | **15:30** | 大引け |
  | **14:00**–24:00 | 09:00 next trading day | 翌営業日前場寄付 |

  The 15:30 close reflects TSE's 2024-11-05 session extension and closing
  auction; a stale 15:00 silently mis-times every S株 shortfall observation.
  Encoded and tested in `s_kabu_slot_for()`.
- Aggregation is **median (not mean) × 2**, and cells below `min_observations`
  are emitted **empty rather than estimated** — one fill is an anecdote, and a
  cost built from it would be quoted with the same authority as one built from
  fifty.
- **O-3 is NOT required to proceed.** Schema, producer, observed-cell
  aggregation, and insufficient/fallback logic are built and tested without any
  owner figure. O-3 becomes a choice between *declaring* costs and *accepting
  observed-only* costs once fills accrue. Nothing is defaulted; absent both, the
  hurdle stays uncomputable, which is the honest state.

## 6. Build order (dependencies, not dates)

**P34-05 moved forward** — it must exist before the first real outcome read, not
after it. A registry written afterwards cannot prove trials were declared in
advance, which is the only thing it is for.

```
P34-00  gate reachability audit                     ── before any new gate
   │
   ├── P34-01a T1 raw buyback event capture   ─┐  P0, PARALLEL
   ├── P34-01b cost schema v2 + producer      ─┤  (irreversible data first;
   └── P34-05  trial registry                 ─┘   O-3 not required)
   │
P34-02  FREEZE T1 plan (event def, entry, benchmark, horizons, strata)
   │    ── no outcome may be read before this
P34-03  Event-study upgrade ON the existing backtest_disclosure_drift_history
   │    skeleton: CAR/BHAR, calendar-time portfolio, matched controls,
   │    same-day clustering, overlapping holds, cluster bootstrap.
   │    Generic code may be developed against SYNTHETIC fixtures at any time;
   │    the real T1 confirmatory run waits on P34-02.
P34-04  Wire calibration/purged_walk_forward.py (EXISTS, P12-02 — this is
   │    wiring + extension, not from scratch) into event/factor labels;
   │    add PBO/CPCV for MULTI-CONFIGURATION sweeps only — a single
   │    pre-registered hypothesis goes the frozen-definition + calendar-time
   │    + cluster-bootstrap route instead
P34-06  Opportunity gate shadow (two orthogonal status axes, schema in §3)
P34-07  T2 data chain (announcement timestamps, SUE, ownership snapshots)
P34-08  TSMOM six-arm shadow report (no mandate change)
P34-09  US EDGAR replication lane (last; SKHY-lane pattern)
```
Screener weights: frozen throughout (Rule 4 + validation infra first).

**Exploratory vs confirmatory must be reported separately.** Anything computed
on data that existed before its freeze is `legacy` or `hypothesis_generating`
and may not be presented as confirmatory — enforced in code by
`preregistration.freeze_plan()`, which refuses a `prospective` claim whose rule
predates the freeze.

## 7. Owner decision points

| # | Decision | Status | Blocks |
|---|---|---|---|
| O-1 | Account facts, not just a broker name: `broker`, `account_type` (特定/一般/NISA), `fee_course`, tax wrapper, `currency_settlement_mode` (円貨/外貨), `typical_order_notional`, whole-share vs fractional | **OPEN** | Only the *finality* of §1's cost argument. Infrastructure and research focus proceed; the cost conclusion is marked provisional meanwhile |
| O-2 | Disposition of dormant `min_entry_score=70`: keep as test fixture / isolate / delete | **OPEN but NON-BLOCKING** — P34-00 returned `dormant`, so no live gate exists and nothing is at risk | nothing |
| O-3 | Either declare cost figures, **or** choose observed-only accrual | **OPEN, non-blocking for build** | Only a *computable* Rule 16.0 hurdle. Schema/producer/aggregation are built without it; the hurdle stays honestly uncomputable until costs exist |
| O-4 | Adopt T1>T2>E/P>TSMOM as the research agenda | **GRANTED 2026-08-08** (engineering/research ordering only) | sequencing only |

No decision below has been made on the owner's behalf. Where this round had to
choose (e.g. keeping one canonical cost file, median-×2 aggregation, 20D primary
horizon), the choice is recorded in the frozen plan or module docstring so it can
be overridden by inspection rather than archaeology.

## 8. Sources (key)

- Asness (2011) *Momentum in Japan* — JPM; AQR PDF.
- Fama & French (2012) *Size, Value, and Momentum in International Stock
  Returns* — JFE 105.
- Moskowitz, Ooi & Pedersen (2012) *Time Series Momentum* — JFE 104.
- McLean & Pontiff (2016) *Does Academic Research Destroy Stock Return
  Predictability?* — JF 71.
- Jinushi et al., *PEAD and Ownership Structure in the Modern Japanese Stock
  Market* — TJAR 13, DOI 10.11640/tjar.13.2023.01.
- *Do buyback anomalies explain the stock return in Japan? A **registered
  report*** — PBFJ 2025, DOI 10.1016/j.pacfin.2025.102666. Design credibility,
  **not** a positive result.
- *Share repurchases on the Tokyo Stock Exchange Trading Network* — PBFJ 2021,
  S0889158321000277 (source of the "post-announcement drift observed in Japan,
  not in the referenced US studies" claim).
- FY2025 Japanese buyback authorizations ¥22.325tn, +18% YoY, 5th consecutive
  record (Nikkei, fiscal-year basis).
- Chow (1970) IEEE-IT reject option; El-Yaniv & Wiener (2010) JMLR selective
  prediction; Saito & Rehmsmeier (2015) PR-vs-ROC, PLOS ONE.
- Bailey & López de Prado — PBO (JCF 2016), Deflated Sharpe (2014).
- Broker/exchange facts: SBI ゼロ革命 + S株 rules; Rakuten ゼロコース/かぶミニ;
  SBI/Rakuten US 0.495% cap $22; moomoo 0.132%; JPX tick tables; J-Quants free
  tier 12-week delay; SEC EDGAR APIs.
