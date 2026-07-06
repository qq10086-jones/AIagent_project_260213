# ADR-0010: Disclosure-Drift Direction, Execution-Gated and Overfit-Guarded

- Status: Proposed (2026-06-17)
- Supersedes: nothing (augments ADR-0009; redirects the *edge-seeking* effort)
- Related: ADR-0009 (news/theme engine), ADR-0003 (decision log / forward sampling),
  Rule 3 (advice-only), Rule 5 (no backtest without costs), Rule 8.2.x (validation
  gate integrity / locked criteria), Rule 9.4 (uncalibrated honesty), Rule 11.13
  (Event Desk), §12 (anti-FOMO), §15 (Local Beta).

## Context

A literature pass (3 research agents) + an adversarial Codex review settled the
honest picture for this small (~¥400k) retail JP-equity, advice-only system:

- The current model (price-momentum + liquidity screener + news-theme catalyst
  overlay) has **no demonstrated out-of-sample edge** vs passive TOPIX — the expected
  outcome, not a bug. Short-horizon equity predictability has a brutally low R²
  ceiling (Welch-Goyal; Martin-Nagel), **momentum is famously weak in Japan**
  (Asness; Chui-Titman-Wei), and 1-5d is the worst horizon (weakest signal × highest
  turnover/cost).
- The **one** information edge with direct Japan OOS evidence that has NOT been
  arbitraged: **multi-day post-disclosure / earnings drift (PEAD) in small,
  illiquid, low-foreign-ownership, low-coverage names** (Jinushi 2023, TJAR: JP PEAD
  survives precisely where individual investors dominate). Reinforced by stale-news
  reversal (Tetlock 2011) and limited-attention (DellaVigna-Pollet 2009).
- **Codex verdict: PURSUE-WITH-CHANGES.** The central killer is EXECUTION: the same
  illiquidity that preserves the edge also consumes it for a retail trader. Prove
  clean event data + executable fills FIRST; the edge is real only inside a narrow
  tradable window.
- **Phase 0 go/no-go (2026-06-17):** the tradable window for ¥400k is narrow
  (~¥600-1,300 names: cheap names' tick-cost kills it, expensive names can't be
  diversified at 100-share lots). The TDnet collection pipeline EXISTS
  (`poll_tdnet_rss` + parser + storage) but a production disclosure corpus is not yet
  accrued. Verdict: **CONDITIONAL — build the gates + data accrual; let the forward
  log decide; be ready to accept "no edge → passive core".**

## Decision

Redirect the **edge-seeking** effort (keeping the whole existing framework —
advice-only screener + news engine + Event Desk + forward log + governance — intact)
toward **execution-gated, overfit-guarded disclosure-drift**, built in this order
(Codex's priority):

1. **Execution / tradability gate (FIRST).** A deterministic gate: JPX tick ladder →
   estimated round-trip cost, 100-share lot affordability for the account, ADV floor,
   and net-expected-after-cost (+ 2× cost stress). A signal that does not clear
   net-of-cost in the tradable window is **not actionable** (fail-closed).
2. **Disclosure-surprise data layer.** Accrue the TDnet corpus (existing poller) and
   compute a surprise/novelty signal (genuinely novel disclosure vs stale-news
   reprint), event-time-stamped from the **Japanese release time + first-seen crawl
   time** (never the later English timestamp).
3. **Small-cap / low-coverage universe tilt** (where the JP drift survives).
4. **Anti-overfit promotion gate.** Trial-counter N + Deflated Sharpe + PBO/CPCV +
   embargo ≥ label horizon (≥5d) + a written numeric prior before testing + Harvey
   t≥3 hurdle. Nothing is called "edge" until it clears this AND the append-only
   forward log (the only un-overfittable proof).
5. **(Deferred) vol-regime risk gate** for sizing — kept simple, not a stock picker.

The existing news-theme catalyst rerank (ADR-0009 / P15) is **kept as honest ordering
metadata** (Rule 11.12 — it never claimed edge), NOT promoted to validated alpha.

## Consequences

- **Honest framing preserved:** this does not promise profit. It builds the
  capability + gates to test ONE evidence-backed direction with discipline, and to
  accept passive if the forward log says no edge. Rule 3 / 9.4 / §12 unchanged.
- **Execution realism becomes first-class** (new Rule 5.1) — the project previously
  modeled no transaction cost in candidate selection.
- **Codex stays in the loop**: re-review at the data-layer and validation-gate
  milestones.
- Capacity/sizing honesty: at ¥400k the tradable universe is a handful of
  ~¥600-1,300 names; this is a small satellite, not a money pump.
