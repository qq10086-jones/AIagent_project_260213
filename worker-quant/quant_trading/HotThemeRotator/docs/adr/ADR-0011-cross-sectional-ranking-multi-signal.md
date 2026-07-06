# ADR-0011: Cross-Sectional Ranking & Multi-Signal Composite (methodology layer on ADR-0010)

- Status: Proposed (2026-06-23)
- Supersedes: nothing. **Extends** ADR-0010 — it does NOT replace ADR-0010's
  edge direction, execution gate, or anti-overfit gate. It adds the *how*
  (evaluation + combination methodology) on top of ADR-0010's *what*.
- Related: ADR-0010 (disclosure-drift direction; execution gate Rule 5.1;
  anti-overfit promotion gate), ADR-0009 (news/theme engine), ADR-0006
  (backdated calibration / bootstrap), ADR-0003 (decision log / forward
  sampling), Rule 3 (advice-only), Rule 5 / 5.1 (costs / execution gate),
  Rule 8.2.x (validation integrity), Rule 9.4 (uncalibrated honesty),
  Rule 13.x (proposal / shadow promotion lifecycle), §12 (anti-FOMO),
  §15 (Local Beta). Introduces **Section 16** (the enforceable rules).

## Context

ADR-0010 concluded *top-down* (literature + Codex) that short-horizon
(1–5d) absolute-direction prediction has no tradable edge and that the
cost-bearing structure (turnover × spread) is the binding constraint. A
*bottom-up* data derivation (Phase −1, 2026-06-23) independently reached the
same place on live-only data (2026-05-27 → 06-23, 19 trading days, ~48
names/day):

- The existing screener `buy` score has **negative** cross-sectional Rank-IC
  on live data at every horizon — 1D −0.032, 3D −0.041, 5D −0.085 (t=−2.0):
  higher score → *lower* forward relative return. The cross-sectional reframe
  did NOT rescue it; it confirmed the live AUC≈0.46 finding from a second angle.
- The transaction-cost break-even, derived as `IC > τ·c_rt/σ_r` (below),
  evaluates to a hurdle of **0.09–0.17 at 1–3D** — multiples of any realistic
  equity Rank-IC (~0.02–0.05). At ≥5D the hurdle falls to **0.025–0.07**,
  because cross-sectional return dispersion σ_r grows ~√horizon while
  per-rebalance cost is fixed. This *is* ADR-0010's "1–5d is the worst horizon"
  and Rule 5.1's net-of-cost gate, expressed as one a-priori inequality.

What ADR-0010 does NOT yet specify, and this ADR adds (methodology, not a new
direction):

1. **Cross-sectional Rank-IC** as the primary, cheap, a-priori signal metric
   (vs the heavier strategy-level Deflated Sharpe / PBO gate, which remains the
   *promotion* gate per ADR-0010).
2. **Multi-signal composition** — how to combine several weak, decorrelated
   signals into one ranking, with quantitative bounds on when stacking helps.
3. **The IC→cost-hurdle inequality** as a build/no-build screen applied *before*
   any code is written.

## Decision

Adopt **cross-sectional ranking + decorrelated multi-signal stacking** as the
methodology for the edge-seeking effort defined in ADR-0010, while **reusing
ADR-0010's execution gate (Rule 5.1), anti-overfit promotion gate, and chosen
direction unchanged**. The derivations that bind the design (enforced by
Section 16):

1. **Cross-sectional removes the market factor.** With
   r_i = α_i + β_i·F + ε_i, cross-sectional demeaning cancels the common F term
   when betas are homogeneous; residual signal-to-noise becomes
   (α-dispersion / ε-dispersion) instead of being swamped by β²·Var(F).
   ⇒ predict and evaluate *relative rank*, not absolute direction;
   beta-residualize when candidate betas are heterogeneous.

2. **IC→net-return cost hurdle.** For a rank-weighted book,
   E[gross per period] ≈ IC·σ_r; net ≈ IC·σ_r − τ·c_rt; hence
   **build only if `IC > τ·c_rt/σ_r`**. Since σ_r grows ~√horizon while c_rt per
   rebalance is fixed, the hurdle falls with horizon ⇒ **operate at ≥5D (target
   weekly–monthly), low turnover**; short horizons are structurally uneconomic
   regardless of signal quality.

3. **Stacking gain and its ceiling.** Equal-weight composite of K signals
   (equal IC, pairwise correlation ρ):
   `IC_comb = IC·√K / √(1+(K−1)ρ)`. ρ=0 → IC·√K; ρ>0 → ceiling IC/√ρ.
   ⇒ a few *decorrelated* signals capture most of the gain; correlated signals
   add ~nothing past the ceiling. Require |ρ|<0.5 before a signal joins.

4. **Estimation-noise bounds complexity.** Weights estimated from N samples are
   noise-dominated when K/N is not small (random-matrix noise eigenvalues up to
   (1+√(K/N))²); James-Stein shows shrinkage dominates the naive estimate for
   ≥3 means. ⇒ **equal-weight (maximal shrinkage) by default**; fitted weights
   require the Rule 13.3 sample tier + purged-CV; keep K single-digit at N≈600.

The first two seed signals for the composite are ADR-0010's own:
**disclosure-drift / PEAD** (small, low-coverage names) and **stale-news
reversal** (Tetlock) — the latter is the literature form of the negative-IC
mean-reversion lead Phase −1 surfaced (treated as a hypothesis to verify
forward, never sign-flipped into a live trade on one month of data).

## Consequences

- **No new direction, no parallel track.** This is the *how* layered on
  ADR-0010's *what*. Execution realism (Rule 5.1) and the anti-overfit
  promotion gate are unchanged and remain the final word; Rank-IC vs hurdle is
  only the cheap pre-screen.
- **A cheaper go/no-go**: a signal can be killed by the IC-cost-hurdle
  (Rule 16.0) before incurring the cost of the heavy promotion gate.
- **Honest framing preserved** (Rule 3 / 9.4 / §12): nothing here promises edge.
  Phase −1 already shows the existing signal is perverse and short horizons are
  uneconomic; passive core remains the null hypothesis.
- **Codex re-review** at the forward-test-harness and first-composite
  milestones, consistent with ADR-0010.
