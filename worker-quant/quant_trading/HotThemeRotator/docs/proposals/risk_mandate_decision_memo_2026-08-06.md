# Risk-Mandate Decision Memo (P32)

Date: 2026-08-06
Status: **PROPOSAL — advice-only (Rule 3).** This memo changes nothing. `configs/risk_mandate.json` is owner-declared; any change goes through Rule 4 (field, old value, new value, reason, expected impact, verification).
Scope: reconcile ADR-0012's stated derivation with the parameters actually recorded, and put ONE bounded choice in front of the owner.
Non-scope: capital deployment, band occupancy, signal promotion, broker anything.

All figures below are **model outputs under ADR-0012's own assumptions**, never forecasts and never probabilities of an outcome (Rule 8.3). They are reproducible from `mu=0.055`, `sigma=0.18` as recorded in `configs/risk_mandate.json`.

---

## 1. The inconsistency

`configs/risk_mandate.json` records four parameters that cannot all be outputs of one derivation:

```
equity_premium_mu     0.055
market_sigma          0.18
kelly_fraction_lambda 0.75
target_exposure_ratio 1.4      band [1.2, 1.6]
p_floor_hit_target    0.10
```

From those inputs:

| Quantity | Value |
|---|---|
| Full Kelly `f* = mu/sigma^2` | **1.6975x** |
| `lambda = 0.75` implies exposure `0.75 x 1.6975` | **1.2731x** |
| `target = 1.4x` implies `lambda` | **0.8247** |

**ADR-0012 line 34 states `lambda·f* ≈ 1.4x`.** Even using the ADR's own rounded `f* ≈ 1.7`, `0.75 × 1.7 = 1.275`. The stated target does not follow from the stated fractional-Kelly bound by either route. This is an arithmetic error in the derivation text, not a licence to change an owner-declared parameter.

Consequence for the floor claim, using the same continuous-rebalancing GBM approximation the ADR uses, `P(ever reach 0.25 x NAV) = 0.25^(2/lambda - 1)`:

| | `lambda` | P(floor) |
|---|---:|---:|
| As declared (0.75) | 0.75 | **9.92%** |
| As targeted (1.4x) | 0.8247 | **13.87%** |

So `target=1.4x` and `P(floor) <= 10%` cannot both be presented as outputs of the stated approximation. One of them has to give.

## 2. What the choice is actually worth

Expected log growth under the same model is `g(lambda) = lambda(2 - lambda) · g_max`, with `g_max = mu^2/(2 sigma^2) = 4.668%/yr`:

| Exposure | `lambda` | `g` | P(floor) |
|---|---:|---:|---:|
| 1.2731x | 0.7500 | 4.3764%/yr | 9.92% |
| 1.4000x | 0.8247 | 4.5248%/yr | 13.87% |
| **Difference** | | **+0.148pp/yr** | **+3.95pp** |

On the provisional post-8035 NAV of ¥384,321 that is **≈¥570/year** of modelled expected growth, bought with ≈4 percentage points of modelled tail. That is the entire decision. It does not need a simulation programme; it needs a signature.

## 3. Two omitted terms that are 10x larger

The derivation's precision is not the binding constraint. Two terms it omits dominate it.

**(a) Leveraged-ETF drag — a real, recurring, computable cost.** ADR-0012 allocates Sleeve A as roughly ¥42k unlevered (1306.T) + ¥175k in 1568.T (TOPIX 2x, daily reset). Daily-reset leverage earns less than `L ×` the index's compounded growth by `sigma^2 · L(L-1)/2`:

```
0.18^2 x 2 x 1 / 2 = 3.24%/yr   — on the LEVERAGED LEG ONLY (~¥175k), not all ¥217k
  variance drag   3.24% x ¥175,000 = ¥5,670/yr
  fee (ASSUMED)   0.75% x ¥175,000 = ¥1,312/yr
  total                             ≈ ¥6,982/yr
```

Sleeve A's β-adjusted exposure at target is `¥42k×1 + ¥175k×2 = ¥392k`, whose gross equity premium at `mu=0.055` is ¥21,560/yr. **The drag is ≈32% of the sleeve's entire gross expected premium** — and it appears nowhere in ADR-0012's `g_max` story.

> ⚠ The **0.75% fee is an assumption carried in from review, not a verified figure.** Before this memo informs any Rule 4 change, the official 1568.T total expense ratio must be checked as of that date and the number restated. The 3.24% variance term is analytic; realised tracking difference is a separate, observable quantity and must not be conflated with it.

**(b) Estimation error in `mu` — which swallows the whole §2 debate.** `SE(mu_hat) = sigma/sqrt(T)`:

| Sample | `SE(mu_hat)` | illustrative 95% CI for `mu` | implied `f* = mu/sigma^2` |
|---|---:|---|---|
| 10 years | 5.69pp | [−5.66%, +16.66%] | [−1.75, 5.14] |
| 30 years | 3.29pp | [−0.94%, +11.94%] | [−0.29, 3.69] |

Even at thirty independent annual observations, full Kelly is not distinguishable from zero at one end or from 3.7x at the other. Every candidate exposure in [1.0, 1.5] sits inside that interval. Serial dependence and regime change would widen it further; the normal interval is illustrative, not a calibrated statement.

Note the structural point: **the floor formula `0.25^(2/lambda - 1)` contains neither `mu` nor `sigma`.** Conditional on `lambda`, the "≤10%" claim is parameter-free. Uncertainty re-enters only because a *fixed 1.4x target* must be divided by an *estimated* `f*` to get `lambda`. The floor claim is therefore not a statement about tail risk — it is a bet on knowing `mu/sigma^2`.

## 4. The alternatives

Exactly one must be chosen. None may be activated by the system (Rule 3); each requires the owner and, where it touches config, a Rule 4 record.

**A. Keep 1.4x, withdraw the ≤10% claim.** Amend `p_floor_hit_target` to the value the target actually implies (≈13.9%), or delete the field. Honest, changes no capital, and admits the floor number was derived backwards.

**B. Align the target to the declared bound.** Move `target_exposure_ratio` to ≈1.27x with a band around it, keeping `lambda=0.75` and the ≤10% claim internally consistent. Costs ≈¥570/yr of modelled growth.

**C. Abandon the Kelly provenance.** Keep 1.4x but re-justify the band as a declared owner risk preference with no derivation attached, and delete `derivation_assumptions`. Given §3(b), this is the most defensible option intellectually: the parameters were never really identified by the data, and dressing a preference as a derivation is the thing that produced this contradiction.

**D. Defer** — only via the P28 time-bounded exception path, with a hard expiry date. Indefinite silence is not an option; the band is currently breached (≈0.415x vs a 1.2x floor) and that unresolved state is itself the larger cost (retrospective §4.4).

*Recommendation, stated as such and not as authority:* **C, or A if the derivation language is worth keeping.** B buys internal consistency by paying ¥570/yr for a number that §3(b) shows is not identified anyway. The genuinely load-bearing decision is not 1.27 vs 1.4 — it is whether Sleeve A gets occupied at all, and at what LETF cost.

## 5. Deferred by design

Block bootstrap, jump/regime Monte Carlo, and a full daily-reset path-dependence model are **not** proposed here. They become a separate task only if the owner picks a quantitatively risk-calibrated band, intends to occupy it, and first states what decision the simulation could change. Absent that, a more precise estimate of a parameter whose 95% interval spans the entire feasible range is precision theatre.

## 6. Non-claims

This memo does not claim the Kelly approximation models the owner's account; that 1.2731x is correct and 1.4x is wrong; that the LETF drag estimate equals realised tracking difference; that `mu = 5.5%` is the true premium; or that any exposure level will produce any particular outcome.

**Evidence:** `configs/risk_mandate.json`; `docs/adr/ADR-0012-owner-risk-mandate-sleeves.md` lines 26–52; Busseti, Ryu & Boyd, *Risk-Constrained Kelly Gambling*, DOI `10.3905/joi.2016.25.3.118`; Avellaneda & Zhang, *Path-Dependence of Leveraged ETF Returns*, DOI `10.1137/090760805`. Arithmetic reproducible from §1's four inputs.
