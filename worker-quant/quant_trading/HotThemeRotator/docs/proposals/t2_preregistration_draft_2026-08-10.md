# T2 — Ownership-Conditioned PEAD: Preregistration DRAFT

Status: **DRAFT — NOT FROZEN.** This document is for review. Freezing happens
only via `tools/freeze_t2_preregistration.py` (to be written) after the open
items in §10 are closed, and **no CAR/BHAR may be computed before that freeze**.
Date: 2026-08-10 · Task: P36-06 · Rule 3 advice-only; no capital, config,
weight, mandate or UI change follows from this document.

---

## 1. Hypothesis

Jinushi et al. (*TJAR* 13, 2002–2020, 60,124 firm-years) report that Japanese
post-earnings-announcement drift decayed over their sample **but did not decay
in firms with low foreign ownership or high individual ownership**. The proposed
mechanism is that sophisticated, low-latency investors arbitrage the drift away
where they are present, and that firms they neglect retain it.

**H1 (low foreign ownership).** Among firms in the bottom foreign-ownership
quintile, uncontaminated annual earnings announcements are followed by positive
cumulative abnormal return over the post-announcement window.

**H2 (high individual ownership).** The same holds among firms in the top
individual-ownership quintile.

**H1 and H2 are tested SEPARATELY**, as in the paper. They are correlated but
distinct claims, and an AND-combination is a *different* hypothesis — it is
available as a registered secondary configuration (§6), never as "the paper's
design".

Direction is predicted positive. **A null or negative result is a valid,
reportable outcome** and does not license re-specification.

---

## 2. Event definition (frozen wording)

- **Event**: a TDnet 決算短信 classified `annual` by
  `earnings_events.classify_tanshin` (parser version p36-02-v1) — i.e. not
  quarterly (四半期/中間), not a correction (訂正), not a notice *about* a 短信.
- **Event time**: the disclosure's `published_ts`.
- **Event date (tradable)**: the first trading session at or after publication;
  a disclosure at or after the **15:30 TSE close is dated to the NEXT trading
  day**. In the assembled sample **73% of annual announcements are after-close**,
  so omitting this shift would credit essentially the whole study with a day of
  return nobody could have captured.
- **Entry**: the open of the event date as defined above.
- **Benchmark**: 1306.T (TOPIX ETF), the benchmark already used by the existing
  event-study skeleton.
- **Returns**: split-adjusted only, via the P35-01 `adjusted_prices` contract;
  any window containing an unresolved corporate action is **excluded**, never
  computed through.

### Exclusions (frozen)
1. Quarterly 短信, corrections, and notices about 短信.
2. Events without 30 pre-event and 60 post-event trading sessions.
3. Events with no ownership snapshot **published before** the event.
4. Windows intersecting an ambiguous corporate action.
5. Symbols whose price series fails `validate_bars`.

---

## 3. Conditioning variable

- **Source**: 所有者別状況 from EDINET 有価証券報告書, parser `p36-01-v1`.
  Values are **fractions** (68.83% → 0.6883) and are stored only when the
  categories partition to 1.0 ± 0.02.
- **PIT rule**: ownership is an *instant* at fiscal year end, public at
  `submitDateTime`. Each event is matched to the **latest ownership snapshot
  published strictly before the event's `published_ts`** — never on `as_of`.
- **Sort**: Jinushi's design — **within each fiscal year**, the 20th percentile
  of `pct_foreign_total` (H1) and the 80th percentile of `pct_individual_total`
  (H2). Fixed absolute thresholds are NOT the paper's design and are registered
  separately if used at all.

---

## 4. Assembled sample (measured 2026-08-10, not projected)

| stage | events |
|---|---|
| primary annual 決算短信 | 3,752 |
| …with price history | 2,785 |
| …with 30 pre + 60 post bars | 2,397 |
| **…with a prior ownership snapshot** | **2,099** (1,844 symbols) |

**Clustering (the number that governs inference):** 2,099 events fall on only
**246 distinct event days, with up to 178 on a single day**; 2025 alone holds
1,503 (71.6%).

Per-fiscal-year buckets — what a within-year test actually has:

| year | events | low-foreign (H1) | high-individual (H2) |
|---|---|---|---|
| 2024 | 391 | 78 | 79 |
| 2025 | 1,503 | 300 | 301 |
| 2026 | 205 | 41 | 41 |
| pooled | 2,099 | 419 | 421 |

---

## 5. Power — declared BEFORE any outcome is read

σ is the cross-sectional standard deviation of the 60-session abnormal return.
**It is an ASSUMPTION here, not a measurement**, because measuring it requires
computing abnormal returns. Three plausible values are carried; the realized σ
will be reported alongside results, and if it lands materially above the assumed
range the power statement — not the hypothesis — is what gets revised.

Effective N applies the Kish design effect `1 + (m−1)ρ` at ρ = 0.10.

**Minimum detectable effect (α = 0.05 two-sided, power = 0.80):**

| bucket | n | σ=0.15 | σ=0.20 | σ=0.25 |
|---|---|---|---|---|
| 2024 | 78–79 | 4.9% | 6.6% | 8.2% |
| 2025 | 300–301 | 2.5% | 3.3% | 4.2% |
| 2026 | 41 | 6.8% | 9.1% | 11.3% |
| **pooled** | **419–421** | **2.2%** | **3.0%** | **3.7%** |

**Achieved power against a 2% drift:**

| bucket | σ=0.15 | σ=0.20 | σ=0.25 |
|---|---|---|---|
| 2024 | 21% | 13% | 10% |
| 2025 | 61% | 39% | 27% |
| 2026 | 13% | 9% | 7% |
| **pooled** | **71%** | **47%** | **33%** |

### What this forces us to declare
1. **Per-year testing is severely underpowered.** At σ = 0.20 the 2024 and 2026
   buckets have 9–13% power. **A null result in those years carries essentially
   no evidence about the hypothesis** and must never be reported as "no drift".
2. **Even pooled, power is adequate only at the optimistic σ.** 71% at σ=0.15,
   47% at σ=0.20, 33% at σ=0.25 — against the conventional 80% bar, the study is
   **underpowered for a 2% effect at central assumptions**.
3. Events needed per bucket for 80% power at a 2% effect: **521 (σ=0.15),
   927 (σ=0.20), 1,448 (σ=0.25)**. We have 419–421.

**Consequence for the design (§6): the pooled specification is PRIMARY**, and
per-year is secondary/descriptive, precisely because the paper's within-year
design has no power on our sample. This is a deliberate, pre-declared departure
with its reason stated — not a post-hoc rescue.

---

## 6. Analysis plan

**Primary (P1).** Pooled across fiscal years with year fixed effects; buckets
formed by the within-year percentile sort of §3. Statistic: mean CAR at the
primary horizon, H1 and H2 separately.

**Overlap-robust cross-check (P2).** Calendar-time portfolio (P34-03), which
absorbs the 178-events-on-one-day problem structurally rather than adjusting for
it.

**Secondary, explicitly underpowered (S1).** Per-fiscal-year tests. Reported
with their MDE alongside every estimate; a null is reported as *underpowered*,
not as evidence of absence.

**Registered configurations (S2), never "the paper's design".** Fixed-threshold
buckets (e.g. foreign < 5%, individual > 50%) and the AND-combination
(1,132 deduped symbols). Each is its own trial in the registry.

### Horizons
Primary **60 sessions** (the paper's BHAR window). Secondary 5 / 20 / 120.
Both CAR and BHAR reported; they answer different questions and diverge with
horizon.

### Inference (frozen)
- Standard errors **clustered by event day AND by firm** (two-way). The nominal
  2,099 overstates information; the clustering block in the join report is the
  evidence for why.
- Date-cluster bootstrap CI as the primary interval.
- Calendar-time portfolio as the independent cross-check. **If P1 and P2
  disagree, the disagreement is the finding** and no single number is promoted.

### Multiple testing
2 hypotheses × 4 horizons × {CAR, BHAR} = 16 primary trials, plus per-year
(S1) and alternative configurations (S2). **All register in family `P36_T2_v1`
BEFORE any outcome read**; the deflation denominator is the registry count.
P31's frozen family is cited additively, never written.

---

## 7. Controls

- **Size**: market cap = `shares_outstanding` × price at event date, from the
  same EDINET 経営指標等 block. ✅ **RESOLVED 2026-08-10: 2,058 / 2,058 =
  100% of join-paired documents carry shares outstanding, 0 failures.**
  Observed range 224,507 .. 16,314,987,460 shares — five orders of magnitude, so
  size is a live control rather than a formality. Size control is therefore
  IN the plan and does not need to be dropped.
- **Liquidity**: ADV from raw close × volume (raw is correct here — turnover
  asks what traded).
- Buckets are formed within year, which absorbs year-level market conditions.

---

## 8. Stopping rule

- No interim peeking. No CAR/BHAR is computed before the freeze.
- The first confirmatory read happens **once** on the frozen specification.
- **A confirmatory claim additionally requires the achieved-power condition**:
  realized σ must imply ≥ 60% power against the pre-declared 2% effect on the
  pooled specification. Below that, results are reported as **exploratory and
  underpowered** regardless of their p-values.
- Accrual continues; a re-read requires a new plan version, and both versions
  stay on disk.

---

## 9. What would falsify / what would not

- **Supports H1/H2**: positive mean CAR in the low-foreign / high-individual
  buckets, surviving two-way clustering, agreeing between P1 and P2, and
  exceeding the Rule 16.0 cost hurdle for the execution profile that would trade
  it.
- **Refutes**: a precisely-estimated null (CI excluding the pre-declared 2%)
  under adequate power.
- **Neither**: an imprecise null under the power shown in §5 — which, on current
  numbers, is the **most likely outcome** and is being stated now rather than
  discovered later.

---

## 10. Open items blocking the freeze

1. ~~Shares-outstanding coverage~~ — **CLOSED 2026-08-10, 2,058/2,058 (100%),
   0 failures.** Size control stays in the plan.
2. **σ assumption** confirmed as a declared range (owner or engineering call);
   the MDE table is only as good as it.
3. **Registry**: create `P36_T2_v1` and register all 16 primary trials plus the
   secondary configurations, before the confirmatory run.
4. **Cost model**: the Rule 16.0 hurdle still has no declared cost figures
   (O-3), so "exceeds the cost hurdle" is currently uncomputable.
5. Owner sign-off on making the **pooled** specification primary in place of the
   paper's within-year design, given §5.

---

## 11. Provenance

Corpus 90,307 disclosures / 842 trading days (TDnet main ∪ probe); ownership
panel 29,294 rows spanning 2019-05-31..2026-07-02; join report
`reports/research/t2_join_report_2026-08-10.json`; parsers `p36-01-v1`
(ownership) and `p36-02-v1` (events); power via
`research/event_power.py`. Nothing in this document was computed from an
outcome.
