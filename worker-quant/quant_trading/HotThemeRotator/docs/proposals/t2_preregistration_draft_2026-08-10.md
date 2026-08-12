# T2 — Ownership-Conditioned PEAD: Preregistration DRAFT v6

Status: **DRAFT — NOT FROZEN. Trial family NOT registered.** Rule 3 advice-only.

**v4 supersedes v3 after owner review. v3's simulation used a cluster shape that
does not exist in this study**, so every number it produced is withdrawn (§5).
v4 also switches the primary specification to the disjoint window (§1), because
the additive identity that gives "H0: β₁ = 1" its meaning does not hold for the
paper's BHAR left-hand side.

Version history: v1 mean-CAR estimand · v2 slope estimand + fiscal-year fix ·
v3 null value = 1 + Monte Carlo · v4 real cluster shapes, disjoint primary ·
v5 claimed items 2-4 complete ·
**v6 withdraws that claim: the joint simulation was a no-op, the design was
rank deficient, and the "primary specification power" was not the primary
specification's (see 5c).**

---

## 1. Hypotheses and estimand (primary changed in v4)

The estimand is the relation between the **announcement reaction** and the
**subsequent** abnormal return. An unconditional mean CAR is not PEAD (signed
news cancels); the slope is.

### Primary specification (P) — disjoint windows

    AR_i[+2,+60] = β₀ + β₁ · AR_i[-1,+1] + γ'X_i + δ_FY + ε_i        H₀: β₁ = 0

**H1** — β₁ > 0 in the bottom within-fiscal-year foreign-ownership quintile.
**H2** — β₁ > 0 in the top within-fiscal-year individual-ownership quintile.

Chosen as primary because its null value follows from the design rather than
from an accounting identity, and because the regressor is not contained in the
dependent variable. Simulation check: with an independent post-window,
β̂₁ = −0.009.

### Secondary replication specification (R) — the paper's overlapping window

    CAR_i[-1,+60] = β₀ + β₁ · CAR_i[-1,+1] + …                        H₀: β₁ = 1

**⚠ Two conditions, both stated because v3 got this wrong.** (a) The null of 1
follows from `CAR[-1,+60] = CAR[-1,+1] + CAR[+2,+60]`, which holds for
**additive** CAR / log abnormal returns — **not for BHAR**, which does not
decompose that way. (b) Jinushi's LHS is a BHAR while the regressor is an
announcement CAR, so **this specification is a comparability exercise, not a
literal replication**; it is run on additive CAR and labelled as such. v3's
"β₁ > 0 on the overlapping LHS" is withdrawn outright — market efficiency alone
satisfies it (simulated β̂₁ = 0.991 with no drift).

H1 and H2 are tested **separately**. Direction predicted positive; a null or
negative result is valid and reportable.

---

## 2. Event definition — unchanged from v2

TDnet 決算短信 classified `annual` (p36-02-v1); event date = first session at
or after publication, **after-close ⇒ next trading day** (73% of annual 短信);
benchmark 1306.T; split-adjusted returns only (P35-01), windows crossing an
unresolved corporate action excluded. Exclusions: quarterly/correction/notice;
<30 pre or <60 post sessions; no prior ownership snapshot; `validate_bars`
failure.

## 3. Conditioning variable — unchanged from v2

所有者別状況 fractions (p36-01-v1), matched to the latest snapshot **published
strictly before** the event. **Fiscal year = April–March, labelled by ending
year.** Within-fiscal-year sorts: 20th percentile of `pct_foreign_total` (H1),
80th percentile of `pct_individual_total` (H2). Fixed absolute thresholds are
our own configuration and register separately.

---

## 4. Assembled sample (measured 2026-08-10)

Ladder: 3,752 annual 短信 → 2,785 with prices → 2,397 with windows → **2,099
with a prior ownership snapshot** (1,844 symbols).

| fiscal year | events | H1 | H2 |
|---|---|---|---|
| FY2025 | 647 | 130 | 130 |
| FY2026 | 1,303 | 260 | 261 |
| FY2027 (**partial**) | 149 | 30 | 30 |
| pooled | 2,099 | **420** | **421** |

**Real bucket cluster structure** — emitted by the join tool
(`bucket_cluster_sizes`), never hand-entered:

| bucket | events | event days | largest day |
|---|---|---|---|
| H1 low-foreign | 420 | 121 | **36** |
| H2 high-individual | 421 | 125 | **38** |

The full sample's largest day is 178, but **no bucket contains such a day**.

---

## 5. Power — v3's numbers WITHDRAWN, recomputed on the real shape

**What went wrong.** v3 simulated 42 clusters with a 178-event day, taking the
full-sample maximum as if it were a bucket's. The real buckets have 121–125
days with a maximum of 36–38. Everything v3 derived from that shape is void:
CR1 size 0.102, WCB size 0.045, the β₁ = 1.10–1.50 power curve, and the
conclusion "CR1 over-rejects 2×". The join tool now emits the real arrays and
the tests read them, so a shape can no longer be invented.

**Size on the real H1 shape (420 / 121 / max 36), α = 0.05 one-sided:**

| method | size | n_sims |
|---|---|---|
| CR1 t-test | **0.0503** ✅ essentially exact | 3,000 |
| Wild cluster bootstrap | 0.033 (conservative) | 600 |

**CR1 is not broken on this sample.** The wild cluster bootstrap is retained as
the **primary inference on robustness grounds** — the standard choice under
unbalanced clusters and moderate cluster counts (Cameron–Gelbach–Miller 2008;
MacKinnon–Nielsen–Webb 2023) — *not* because CR1 was shown to fail. CR1 is
reported alongside.

**Power on the real H1 shape** (central scenario σ_a = 0.06, σ_post = 0.20,
ICC = 0.10; drift shown per 1 s.d. announcement reaction):

| β₁ | implied drift | power |
|---|---|---|
| 1.00 | 0 | 0.050 (size) |
| 1.05 | 0.30% | 0.09 |
| 1.10 | 0.60% | 0.15 |
| 1.15 | 0.90% | 0.24 |
| 1.20 | 1.20% | 0.34 |
| 1.30 | 1.80% | 0.55 |

*(Tabulated for specification R's parameterisation; P's β₁ equals R's β₁ − 1
under the additive identity, so the drift column is the quantity that carries
across.)*

### β₁* is NOT proposed in v4

v3 proposed β₁* = 1.30 "because it is the smallest effect we can see at better
than a coin flip". **That reasoning is rejected: detectability cannot define
economic importance.** β₁* must be argued from economics — the drift per 1 s.d.
reaction (1.8% at β₁ = 1.30) set against round-trip cost and the literature's
effect sizes — and the resulting power then *reported*, not used to pick it.
Left open (§10).

### Sensitivity grid (pre-declared, replacing a single scenario)

σ_a ∈ {0.04, 0.06, 0.08} × σ_post ∈ {0.15, 0.20, 0.30} ×
ICC_announce ∈ {0.05, 0.10, 0.20} × ICC_post ∈ {0.05, 0.10, 0.20} ×
corr(day shocks) ∈ {0, 0.3} — run on **each bucket's own** cluster array.
The central scenario is reported with the grid, never alone.

Per-fiscal-year testing stays secondary and underpowered (FY2027: 30 events).

---

## 5b. NEW (v5): correlated day shocks BIAS the slope — event-day fixed effects

Running the sensitivity grid surfaced a threat that is **not** about standard
errors. If the announcement-day market shock is correlated with the shock over
the following sessions, the regressor is correlated with the error term: the
slope is **biased**, and no amount of clustering repairs it — clustering fixes
standard errors, not endogeneity.

Measured size under H₀ on the real H1 shape, full model:

| corr(announcement-day, post-day shock) | size at nominal 5% |
|---|---|
| 0.0 | 0.050 |
| 0.1 | 0.075 |
| 0.2 | 0.105 |
| 0.3 | **0.147** |
| 0.5 | **0.259** |

Abnormal returns are already net of 1306.T, so the residual day-level
correlation should be small — but "should be" is not a design.

**Remedy, now part of the plan: event-day fixed effects in the primary
specification.**

WARNING - claim scope, tightened in v6. Day fixed effects are shown to absorb
the common, additive, identical-across-firms event-day shock IN THIS DGP. They
are NOT shown to remove heterogeneous exposure to a common shock, firm-level
persistent shocks, or any omitted variable correlated with the announcement
reaction. This is not "an unbiased estimate bought" - it is one named threat
closed.

Cost, reported for BOTH buckets: the slope is identified from within-day
variation only, so H1 loses 52 singleton days (367 events keep within-day
identification) and H2 loses 60 (359 keep it).

**Verified both ways at ρ = 0.3**: size 0.152 without day fixed effects,
**0.033 with them**.

### WITHDRAWN: the v5 "primary specification power" table

It was computed with CR1, without bootstrap, Holm, or the real overlap, so it
was a single-bucket marginal figure and NOT the power of the rule this document
commits to. The replacement is being simulated under day FE + wild cluster
bootstrap + Holm on the real shared-event mapping and will be stored as a
reproducible artifact before any freeze. **Until then this document states no
power number for the primary rule.**

What survives: day fixed effects cost power (the singleton-day loss plus the
degrees of freedom the dummies consume), so the eventual figure will sit
materially below any no-day-FE comparison.

The specification without day fixed effects is retained as a **registered
secondary**, and a disagreement between the two is itself reportable.

## 5c. v5 defects found in review and corrected (2026-08-10)

Three, each verified by reproduction before being accepted.

1. The "joint" Holm simulation was two independent experiments. It took an
   overlap RATIO and added `overlap x sigma x shared` to both outcomes - a
   constant, which an intercept absorbs entirely. Reproduced: overlap 0.0,
   0.382 and 1.0 gave identical power to three decimals. FIXED by emitting the
   real `bucket_events` mapping (419 events per bucket, 159 shared) and
   simulating ONE draw per unique event (679 unique), so shared events carry
   identical realised values into both regressions. Verified to matter: real
   overlap `both = 0.175` vs artificially disjoint `both = 0.025`.
2. Fiscal-year and event-day fixed effects are exactly collinear, because an
   event day lies in exactly one fiscal year. The v5 simulation assigned FY
   PER EVENT, which is impossible in data and hid the collinearity. On the real
   structure the design is 420x126, rank 124, condition number 4e16 - and
   `np.linalg.inv` does NOT raise: it returned slope 0.6354 where the truth was
   0.30 (0.2575 with the redundant block dropped). FIXED: FY assigned per day,
   FY dummies dropped when day FE is used, and `ols_cluster_robust` now checks
   rank before inverting and refuses.
3. The headline power figure was not the decision rule's. Withdrawn above.

## 5d. The authoritative power run (2026-08-12) — the replacement §5c promised

Artifact: `reports/research/t2_power/2026-08-12.json`, produced by
`tools/t2_power_artifact.py`. Simulated data only; no price, announcement
return, CAR, BHAR or test statistic on real outcomes is computed, so producing
it is not an outcome access.

Specification simulated **and** inferred: `AR[+2,+60] = β₀ + β₁·AR[-1,+1] +
controls + event-day FE`, H₀: β₁ = 0, wild cluster bootstrap clustered on event
day, Holm across H1 and H2, α = 0.05 one-sided, on the **real** mapping
(419 / 419 events, **159 shared**, sha256 `ec9066df…f64f` over the mapping
actually used). Seed 20260812, numpy 2.3.5, 199 bootstrap replications.

### Size first — and this time it was verified before power existed

| | value |
|---|---|
| family-wise rejection under the complete null (1 000 draws) | **0.0500** |
| Clopper–Pearson one-sided 95% lower bound | 0.0392 |
| verdict | **at nominal level** |
| H1 Holm / H2 Holm marginals under the null | 0.0320 / 0.0260 |

The runner refuses to print a power table at all if that lower bound exceeds α.
This is the first T2 power figure that rests on a level verified for the rule
that produced it, which is precisely what the three withdrawn versions lacked.

### Power, central scenario (σ_a 0.06, σ_post 0.20, ICC 0.10/0.10, ρ 0), 500 draws/cell

| β₁ (spec P) | β₁ (spec R) | drift per 1 s.d. | H1 Holm | H2 Holm | **any** | **both** |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 0.00% | 0.028 | 0.022 | 0.044 | 0.006 |
| 0.05 | 1.05 | 0.30% | 0.052 | 0.028 | 0.078 | 0.002 |
| 0.10 | 1.10 | 0.60% | 0.058 | 0.076 | 0.112 | 0.022 |
| 0.15 | 1.15 | 0.90% | 0.124 | 0.100 | 0.202 | 0.022 |
| 0.20 | 1.20 | 1.20% | 0.158 | 0.184 | 0.268 | 0.074 |
| 0.25 | 1.25 | 1.50% | 0.262 | 0.262 | 0.388 | 0.136 |
| 0.30 | 1.30 | 1.80% | 0.330 | 0.354 | **0.468** | **0.216** |

**§5c's prediction is confirmed and it is not comfortable.** v3 reported 0.55 at
drift 1.8% under a specification with no day fixed effects, no bootstrap and no
Holm. Under the rule this document actually registers the same effect gives
**0.468** to reject *either* hypothesis and **0.216** to reject *both*. The
study is underpowered across the whole plausible range: nowhere on this grid
does "any" reach 0.5, and "both" never reaches 0.25.

**The gap between the `any` and `both` columns is a decision-rule question, not
a statistical one, and it is unresolved (§10).** If confirming T2 requires both
H1 and H2, the study's power is the `both` column and the design as it stands
cannot support that reading. If either suffices, it is the `any` column. This
must be settled BEFORE freezing, because choosing afterwards on the basis of
which column looks better is exactly the degree of freedom preregistration
exists to remove.

### Pre-declared sensitivity (one axis at a time, β₁ = 0.20, 500 draws/cell)

| axis | value | any | both |
|---|---|---|---|
| central | — | 0.268 | 0.074 |
| σ_a | 0.04 / 0.08 | 0.182 / 0.378 | 0.034 / 0.152 |
| σ_post | 0.15 / 0.30 | 0.378 / 0.182 | 0.152 / 0.034 |
| ICC_announce | 0.05 / 0.20 | 0.272 / 0.256 | 0.072 / 0.066 |
| ICC_post | 0.05 / 0.20 | 0.260 / 0.278 | 0.068 / 0.080 |
| ρ(day shocks) | 0.3 | 0.264 | 0.072 |

Power is driven almost entirely by the signal-to-noise ratio σ_a/σ_post and is
close to flat in both intraclass correlations. **ρ = 0.3 moves power by 0.004**
(0.264 vs 0.268) — the day fixed effects that §5b introduced to remove the
correlated-day-shock BIAS are also, in this DGP, nearly free in power terms
once their singleton-day cost is already paid. §5b's claim-scope warning still
stands: one named threat is closed, not "an unbiased estimate bought".

### β₁* is still not proposed

Deliberately. The artifact records `beta1_star: null`. Deriving it from this
table would be the fourth repetition of the rejected reasoning. See §10 and the
blockers listed there.

## 6. Analysis plan

- **Primary family `P36_T2_v1`: 2 trials** — H1 and H2 under specification P.
  **Family-wise error controlled at 5% by Holm**; power must be recomputed
  against the actual Holm decision rule, not a marginal 5% test.
- **Secondary (registered, not primary):** specification R; horizons 5/20/120;
  BHAR variants; per-fiscal-year estimates (reported with their power); fixed
  threshold and AND-combination buckets; full-sample β₁.
- **Inference:** wild cluster bootstrap by event day, **null imposed**, ≥999
  replications, **run on the SAME model as the point estimate** — intercept,
  slope, size, ADV, fiscal-year fixed effects **and event-day fixed effects**
  (§5b). Implemented and simulated on that specification
  (`full_model_power.wild_cluster_bootstrap_p_general`); v3's two-parameter
  simulation is superseded.
- **Family-wise control by Holm**, with power measured against the Holm rule
  itself. H1 and H2 share **38.2%** of their events (416 each, 159 in common),
  so their statistics are correlated; `simulate_holm_power` simulates the two
  jointly on their real cluster arrays rather than multiplying independent
  figures.
- **Confidence intervals by bootstrap test inversion**, so the interval and the
  decision come from one procedure. CR1 intervals are diagnostics only.
- **Cross-check:** calendar-time portfolio long positive-reaction and short
  negative-reaction events within the bucket. Disagreement with the regression
  is itself the finding.

## 7. Controls

Size = shares outstanding (**2,058/2,058 = 100%** coverage, 224,507..16.3bn) ×
pre-event close; liquidity = 60-session ADV on raw prices; ownership sorts
within fiscal year.

## 8. Interpretation (rewritten in v4 for the new nulls)

Stated per specification, against its own null:

- **Supported** — specification P: bootstrap p ≤ Holm-adjusted threshold **and**
  β̂₁ > 0 **and** the calendar-time cross-check agrees in sign.
- **Effect excluded** — the bootstrap-inverted CI lies entirely below β₁*
  (once β₁* is set on economic grounds).
- **Inconclusive** — the CI contains both 0 and β₁*. Reported as *imprecise*,
  never as "no drift".

Specification R uses the same wording with its null of 1 and β₁* − 1.
Nothing about this classification depends on realized σ.

## 9. Cost hurdle is about tradability, not truth

Rule 16.0 decides whether a supported effect is tradable at our costs. It never
modulates statistical support. Two verdicts, reported separately.

## 10. Open items blocking the freeze

1. **β₁\* from economics, not detectability** — drift per 1 s.d. reaction versus
   round-trip cost and literature effect sizes. Then report the implied power.
   **STILL OPEN, and three of its inputs are not mine to supply:**
   - *round-trip cost* — the declared JP-lot figure is 10–25 bp, but P34-01b is
     PARTIAL: no fill has ever accrued, so there is a declared cost and no
     observed one. O-3 (declare vs observed-only) is unanswered.
   - *σ_a* — the 0.06 used throughout §5d is a SIMULATION ASSUMPTION, not a
     measurement. Reading it off the real `AR[-1,+1]` would touch the regressor,
     not the dependent variable, so it is probably not an outcome access — but
     "probably" is not a protocol, and this document is the protocol. Owner
     decision required before anyone measures it.
   - *the traded fraction of the cross-section* and whether the position is
     long-only or long-short, which set how much of the fitted drift is
     actually capturable.
   Producing a β₁\* number without these would be the fourth repetition of the
   reasoning §5 rejected.
2. ~~**Sensitivity grid executed**~~ — DONE 2026-08-12, §5d, on the real mapping.
3. ~~**WCB extended to the full model**~~ — DONE; simulation and inference now
   run the same specification (`full_model_power.py`).
4. ~~**Holm-adjusted power** for the 2-trial family~~ — DONE 2026-08-12, §5d,
   with size verified at nominal (0.0500) before any power number was printed.
5. Family registration — **deliberately not done yet**, since the specification
   is still moving.
6. O-3 cost figures, for the separate tradability verdict.
7. Owner sign-off on: disjoint-window primary, the sensitivity grid, and β₁*.
8. **NEW, raised by §5d — is the confirmation rule `any` or `both`?** Power is
   0.468 vs 0.216 at the top of the grid, so this single choice roughly halves
   or doubles what the study can conclude. It must be fixed BEFORE the freeze;
   choosing afterwards, on the basis of which column reads better, is exactly
   the degree of freedom preregistration exists to remove.
9. **NEW — the study is underpowered across the whole plausible range.** Nowhere
   on the pre-declared grid does `any` reach 0.5, and `both` never reaches 0.25.
   That is a finding about the design, not a reason to widen the grid until a
   comfortable number appears. The honest options are: accept a low-powered
   study and say so in the registration, wait for more events, or abandon T2.
   Owner decision.

## 11. Provenance

Join report `t2_join_report_2026-08-10.json` (now emitting
`bucket_cluster_sizes`); parsers p36-01-v1 / p36-02-v1;
`research/slope_power_mc.py` (real-shape simulation, WCB) and
`research/event_power.py`. **No AR, CAR or BHAR has been computed on real
data.** All simulation figures come from `numpy` draws under declared
assumptions.
