# T2 — Ownership-Conditioned PEAD: Preregistration DRAFT v3

Status: **DRAFT — NOT FROZEN.** Freezing happens only via a freeze tool after
the open items in §10 are closed, and **no CAR/BHAR/AR may be computed before
that freeze**. v3 supersedes v2 (same day): v2 fixed the estimand, fiscal-year mapping and
power model after owner review; **v3 corrects the null VALUE (β₁ = 1, not 0)
and replaces the inference method after the Monte Carlo showed the ordinary
cluster-robust test over-rejects 2× on this sample.** Rule 3 advice-only.

**This design is a "Jinushi-inspired pooled adaptation". It does NOT replicate
the paper's year-by-year β₁ trend test** — with two complete fiscal years and
one partial, a decay-over-time trend cannot be tested here. What can be tested
is whether conditional PEAD **exists** in the current sample.

---

## 1. Hypotheses and estimand (corrected in v2)

**The estimand is the SLOPE of post-announcement abnormal return on the
announcement-window reaction — not an unconditional mean CAR.** v1 defined
H1/H2 as "mean 60-session CAR > 0 in the bucket"; that tests a different claim:
positive and negative earnings news cancel in an unconditional mean, so a real
drift can produce a mean near zero. Jinushi's main model is

    AR_i,[-1,+60] = β0 + β1 · AR_i,[-1,+1] + ε_i

where β1 > 0 is underreaction: the market keeps moving in the direction of its
initial reaction.

Our primary specification, per bucket (pooled across fiscal years):

    AR_i,[-1,+60] = β0 + β1 · AR_i,[-1,+1] + γ' X_i + δ_FY + ε_i

- `AR_i,[w]` — abnormal return of firm i over window w, sessions indexed
  relative to the **event date** of §2 (0 = first tradable session); abnormal =
  split-adjusted return minus 1306.T return over the same window.
- `X_i` — controls: log market cap (shares outstanding × close at session −2)
  and log 60-session ADV, both from data available before the event.
- `δ_FY` — fiscal-year fixed effects.
- SEs two-way clustered by **event day and firm**.

**⚠ The null value is 1, not 0 (corrected in v3).** The LHS `AR[-1,+60]`
mechanically CONTAINS the regressor `AR[-1,+1]`: writing
`AR[-1,+60] = AR[-1,+1] + AR[+2,+60]` shows that if the post-window is
unrelated to the reaction (an efficient market) the slope is **1**. Verified by
simulation: independent post-window ⇒ β̂₁ = 0.991. v2 wrote "β₁ > 0", which
**market efficiency itself satisfies** — the test would have been vacuous.

**H1 (low foreign ownership):** β₁ > 1 in the bottom within-fiscal-year foreign
ownership quintile.
**H2 (high individual ownership):** β₁ > 1 in the top within-fiscal-year
individual-ownership quintile.

(For the disjoint robustness LHS `AR[+2,+60]` the null is β₁ = 0; simulation
confirms β̂₁ = −0.009 under no drift. Each specification carries its own null
value, stated with it, so the two can never be crossed.)

H1 and H2 are **tested separately**. Full-sample β1 is reported for context.
Direction predicted positive; a null or negative β1 is a valid, reportable
outcome. **Note the paper's LHS window [-1,+60] mechanically contains the
regressor window [-1,+1]**; we keep the paper's definition as primary for
comparability and register `AR_[+2,+60]` as the overlap-free robustness LHS
(secondary, §6).

---

## 2. Event definition (unchanged from v1)

- **Event**: TDnet 決算短信 classified `annual` by `classify_tanshin`
  (p36-02-v1) — not quarterly/中間, not 訂正, not a notice about a 短信.
- **Event date**: first trading session at or after `published_ts`; at or after
  the 15:30 close ⇒ NEXT trading day (73% of annual 短信 are after-close).
- **Benchmark**: 1306.T. **Returns**: split-adjusted only (P35-01 contract);
  windows crossing an unresolved corporate action are excluded.
- **Exclusions**: quarterly/correction/notice; <30 pre or <60 post sessions;
  no prior ownership snapshot; ambiguous corporate action in window;
  `validate_bars` failure.

---

## 3. Conditioning variable (fiscal-year mapping corrected in v2)

- Source and PIT rule unchanged: 所有者別状況 fractions, matched to the latest
  snapshot **published strictly before** the event.
- **Fiscal year = April–March, labelled by ENDING year** (Jinushi's
  convention). v1's "per-year" buckets silently used calendar years — not the
  paper's design; every bucket was mis-sized.
- Sort: within each fiscal year, 20th percentile of `pct_foreign_total` (H1)
  and 80th percentile of `pct_individual_total` (H2). Fixed absolute
  thresholds are OUR configuration and register separately.

---

## 4. Assembled sample (fiscal years, measured 2026-08-10)

Ladder: 3,752 annual 短信 → 2,785 with prices → 2,397 with windows → **2,099
with a prior ownership snapshot** (1,844 symbols; 246 event days, max 178 on
one day).

| fiscal year (Apr–Mar) | events | H1 bucket | H2 bucket | cluster CV (H1/H2) |
|---|---|---|---|---|
| FY2025 | 647 | 130 | 130 | — |
| FY2026 | 1,303 | 260 | 261 | — |
| FY2027 (**partial**, truncated) | 149 | 30 | 30 | — |
| pooled | 2,099 | **420** (121 days, CV 1.58) | **421** (125 days, CV 1.64) | m_e ≈ 12.1 / 12.5 |

FY2027 is an incomplete fiscal year and is never interpreted alongside complete
ones; it enters the pooled regression through its fixed effect only.

---

## 5. Power (corrected in v2 — the honest version)

**v1's power table is superseded and was too optimistic on two counts.**

1. **Equal-cluster Kish overstated effective N by ~70%.** The buckets are
   dominated by a few huge event days (CV ≈ 1.6, one day = 178 events). Using
   the size-weighted cluster size m_e = Σm²/Σm ≈ 12.1–12.5 at ρ = 0.10:
   **effective N ≈ 196–199** per pooled bucket (not 337–340). For a mean-CAR
   style read at σ = 0.20 that implies **MDE ≈ 4.0% and ~29% power against a
   2% effect** (v1 claimed 3.0% / 47%).
2. **These numbers are for the MEAN estimand and do not transfer to the slope.**
   Power for β1 depends on the variance of the announcement-window reaction,
   the residual variance of the 60-session window, and the real (unequal)
   cluster structure — none of which can be assumed from a single σ.

### v3: the Monte Carlo is done, and it changed the inference method

`research/slope_power_mc.py` simulates the slope on the ACTUAL event-day
cluster sizes (no outcome data touched). Checking **size before power**
surfaced a defect that would have invalidated the whole test:

| cluster shape | CR1 t-test size at nominal 5% |
|---|---|
| balanced (42 × 10) | **0.054** ✅ |
| real T2 shape (one 178-event day) | **0.102** ❌ — over-rejects 2× |

With one day holding ~42% of a bucket, the ordinary cluster-robust t-test
rejects at roughly **twice** its nominal level: a "significant at 5%" result
would really be at 10%. An earlier draft of this analysis mistook that
over-rejection for *higher power* from lumpy clusters — the opposite of the
truth, and caught only by simulating the null first.

**Wild cluster bootstrap (Cameron–Gelbach–Miller, null imposed) restores size
to 0.045**, and is therefore **mandatory for this sample, not a refinement.**

**Power under WCB on the pooled H1 shape** (σ_a = 0.06, σ_post = 0.20, ICC = 0.10):

| true β₁ | rejection rate |
|---|---|
| 1.00 (null) | 0.072 — size |
| 1.10 | 0.16 |
| 1.20 | 0.35 |
| 1.30 | 0.57 |
| 1.50 | 0.90 |

**So the study can detect a large drift (β₁ ≈ 1.3–1.5) and cannot reliably
detect a modest one (β₁ ≈ 1.1).** β₁* — the minimum economically meaningful
slope — is proposed at **1.30** (≈57% power), the smallest value this sample
can see at better than a coin flip. Below that, a null is *imprecise*, not
evidence of absence.

Per-fiscal-year testing remains severely underpowered (FY2027: 30 events per
bucket) and is secondary/descriptive only.

---

## 6. Analysis plan

- **Primary family (`P36_T2_v1`): exactly 2 trials** — H1 and H2, slope β1 at
  the [-1,+60] LHS window. (v1 declared 16 primary trials while calling 60
  sessions "the primary horizon" — contradictory; resolved in favour of 2.)
- **Secondary (registered, not primary):** LHS `AR_[+2,+60]` (overlap-free);
  horizons 5/20/120; BHAR variants; per-fiscal-year estimates (reported with
  their MDE, nulls labelled underpowered); fixed-threshold and AND-combination
  buckets; full-sample β1.
- **Overlap-robust cross-check:** calendar-time portfolio (P34-03) long the
  bucket's positive-reaction events and short the negative-reaction events,
  which carries the slope's sign content in portfolio form. If regression and
  calendar-time disagree, the disagreement is the finding.
- **Inference:** **wild cluster bootstrap by event day, null imposed, ≥999
  replications** — required, because the plain CR1 test over-rejects 2× on this
  cluster shape (§5). Two-way clustered SEs (event day × firm) reported
  alongside as a diagnostic, never as the decision rule.
- All trials register in `P36_T2_v1` before any outcome read; P31 is cited
  additively, never written.

---

## 7. Controls

- **Size**: market cap from shares outstanding (coverage **2,058/2,058 =
  100%**, range 224,507..16.3bn shares) × pre-event close.
- **Liquidity**: 60-session ADV from raw close × volume (raw correct here).
- Ownership sorts are within fiscal year, absorbing year-level conditions.

---

## 8. Interpretation rules (v2 — replaces the realized-σ downgrade)

v1 allowed realized σ to retroactively downgrade the study to "exploratory";
the owner correctly rejected that — a preregistered test's status must not
depend on what the data turned out to be. Instead, **interpretation is by
pre-registered interval criteria**:

- **Supported**: β1 > 0 with the two-way-clustered 95% CI excluding 0, AND the
  calendar-time cross-check agreeing in sign.
- **Refuted / effect excluded**: the 95% CI excludes the pre-declared minimum
  economically meaningful slope β1* (to be fixed at freeze together with the
  Monte Carlo power analysis — an equivalence-style bound).
- **Inconclusive**: the CI contains both 0 and β1*. Reported as *imprecise*,
  never as "no drift".

σ enters planning only; results are judged by the intervals above.

## 9. Cost hurdle is about tradability, not truth

The Rule 16.0 hurdle decides whether a supported effect is **tradable at our
costs**. It does not modulate whether the statistical hypothesis is supported.
The two verdicts are reported separately (statistical: §8; economic:
cost-model contract, currently uncomputable pending O-3).

## 10. Open items blocking the freeze

1. ~~Monte Carlo power for the slope estimand~~ — **CLOSED (v3).** Done on the
   real cluster shape; it forced the wild cluster bootstrap into the plan and
   yields the power curve above. **β₁* = 1.30 is now a proposal awaiting owner
   sign-off**, together with the planning assumptions σ_a = 0.06,
   σ_post = 0.20, ICC = 0.10 (these are DECLARED PLANNING VALUES, not
   measurements).
2. ~~Shares coverage~~ — CLOSED (2,058/2,058).
3. Registry family `P36_T2_v1` created; 2 primary + all secondary trials
   registered before the confirmatory run.
4. Rule 16.0 cost figures (O-3) for the tradability verdict.
5. **Owner sign-off on THIS revised design** (slope estimand, fiscal-year
   mapping, pooled-primary, 2-trial primary family, interval-based
   interpretation).

## 11. Provenance

Join report `t2_join_report_2026-08-10.json` (fiscal-year buckets + cluster
stats now emitted by the tool itself); parsers p36-01-v1 / p36-02-v1; power
`research/event_power.py` (unequal-cluster + two-sided corrections 2026-08-10).
Nothing computed from an outcome.
