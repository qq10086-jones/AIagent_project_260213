# ADR-0012: Owner Risk Mandate & Three-Sleeve Architecture

- Status: Accepted (owner-declared 2026-07-13)
- Deciders: owner (mandate + parameters), Claude (derivation + encoding)
- Governance: Section 17 (Rules 17.0–17.6); relaxes NOTHING in Rules 3/5/8/12/14/16

## Context

On 2026-07-13 the owner declared: the ¥400k account is **experimental capital**; a −75% drawdown
(to a NAV floor of ¥100k) is acceptable; the owner explicitly wants a **risk-accepting** system
rather than a purely conservative one, and chose to run the leveraged-beta path and the live
value-experiment path **simultaneously**, with 8035.T re-underwritten into a conviction sleeve.

The system's measured state at declaration (all live-only, Rule 16.2):

- Screener buy-score forward 5D Rank-IC **−0.083 (t −2.54)** — no demonstrated edge; mildly anti-predictive.
- price_reversal family DSR 0.725 < 0.95, n_obs_eff 5/60 — research-only, gated.
- value/E-P: the ONLY family to pass the historical anti-overfit gate (63D DSR 0.992, 2026-07-03);
  live 21D E/P IC ≈ +0.09 (t ≈ 5.5–6.5) but survivorship-bounded; 63D live verdict matures ~**2026-08-26**.
- Holdings: 1306.T ×100 @403 (+3.7%), 8035.T ×1 @77,600 (−8.1%, past its old −4% stop reference), cash ¥287k.

## Decision

### Derivation (recorded so the numbers are auditable, not vibes)

Objective: maximize long-run growth subject to **P(NAV ever hits ¥100k floor) ≤ 10%**.

1. Research assumptions (uncertain, labelled): Japan equity premium μ ≈ 5.5%/yr (DMS long-run;
   SE ±2.5% — the dominant uncertainty), σ ≈ 18%/yr, TEL β ≈ 1.5, 2x-ETF variance drag
   ≈ (L²−L)/2·σ² ≈ 3.2%/yr, S株 round-trip cost 5 bps (`S_KABU_COST`).
2. Growth-optimal (full-Kelly) leverage f\* = μ/σ² ≈ **1.7×**.
3. Drawdown constraint: under constant-fraction rebalancing, P(wealth ever hits fraction x) = x^(2/λ−1).
   x = 0.25, P ≤ 10% ⇒ **λ ≤ 0.75** (three-quarter Kelly).
4. Target β-adjusted exposure = λ·f\* ≈ **1.4× NAV** (band [1.2, 1.6]); expected log-growth
   g = fμ − f²σ²/2 ≈ **4–4.5%/yr**, NAV vol ≈ 25%/yr.
5. Ceiling theorem: max over all leverage of expected growth is g\* = μ²/2σ² ≈ **4.7%/yr**.
   No leverage level "doubles fast" without edge; only μ (edge) raises the ceiling. Under these
   assumptions P(double within 3y) ≈ P(ever hit the floor) ≈ 10% — stated as derivation
   provenance, not prediction (Rule 17.6).

### Allocation (initial; capital figures owner-adjustable via Rule 4)

| Sleeve | Contents | Capital | β-adj exposure | Expectation label |
|---|---|---|---|---|
| A | 1306.T ¥42k + ~¥175k 2x broad ETF (owner buys; live-price-verified at order time) | ¥217k | ~¥392k | compensated beta |
| B | value/E-P top-quantile 5–8 names equal-weight (S株) | ¥60k | ¥60k | ≈0 — evidence purchase |
| C | 8035.T ×1, re-underwritten @¥71,300 2026-07-13; thesis REQUIRED (pending → `thesis_missing`) | ¥71k | ~¥107k (β1.5) | zero demonstrated edge, pure variance |
| Cash | buffer | ~¥52k | 0 | — |
| Total | | ¥400k | ~¥559k ≈ **1.40×** | |

Stress check (2020-scale, index −30%): ≈ −¥168k → NAV ¥232k; with C zeroed ≈ −¥207k → NAV ¥193k.
Both far above the ¥100k floor; the floor needs ~2008-scale plus abandoned rebalancing.

### Mechanics

- `configs/risk_mandate.json` — declarative mandate (parameters, sleeve_map, betas, theses).
- `risk/sleeve_engine.py` — read-only assembler: sleeve rows, exposure ratio vs band,
  kill-switch buffer, discipline flags; unmapped symbols fail-closed to UNASSIGNED.
- `/api/dashboard` `riskMandate` key (fail-open → null) + shared `RiskMandateCard` on V1–V4.
- `daily_routine` afterclose appends `reports/observability/risk_mandate_trace.jsonl` (non-fatal).
- Journal (Section 14) untouched; execution stays manual/external (Rule 3).

## Alternatives considered

1. **Pure leveraged beta** — simplest, but abandons the one live edge candidate (value/E-P) and
   gives C no disciplined container (8035.T would stay a rule-less orphan).
2. **Pure live edge experiment** — cleanest science, but parks ~70% of NAV at zero expected
   return, contradicting the owner's explicit risk appetite.
3. **Concentrated bets / options as the core** — can double fast; expectation zero, variance
   maximal; rejected as a *core* and contained in capped Sleeve C instead.
4. **Reject the mandate (stay conservative)** — overruled by the owner's explicit, informed
   declaration; the honest response is containment-by-design, not refusal.

## Consequences

- The account's expected NAV vol roughly triples (≈8%→≈25%); expected growth rises to ≈4–4.5%/yr.
  A ~10% lifetime chance of hitting the kill-switch is accepted and monitored daily.
- The rebalance discipline (Rule 17.2) becomes load-bearing: the floor-probability math is void
  without it. The surface must nag when the band is breached.
- 8035.T's old −4% exit-board stop is superseded by C discipline (re-underwrite reference
  ¥71,300, review at −20%, thesis mandatory) — an explicit owner decision, recorded here.
- B's size is frozen until the 2026-08-26 verdict (Rule 17.5) — pre-commitment against FOMO.
- No auto-execution anywhere; if the owner stops rebalancing or ignores flags, the system can
  only surface that fact honestly (which is its job).
