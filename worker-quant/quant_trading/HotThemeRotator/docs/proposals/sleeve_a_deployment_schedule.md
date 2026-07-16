# Proposal — Sleeve A Deployment Calendar (P26)

Status: **PROPOSAL — awaiting owner confirmation.** Not active until the owner adopts a
cadence via a Rule 4 change to `configs/risk_mandate.json`. Advice-only (Rule 3): the
system never places orders; this document only replaces "deploy at owner's pace" (which
in practice degrades into day-by-day emotional market-timing) with a pre-committed
mechanical schedule the owner executes at the broker.

## Why

Sleeve A is the only positive-expectation engine (compensated β). As of 2026-07-16 it is
deployed to ~0.68× NAV vs the mandate band [1.2, 1.6] — under half the authorized,
*paid* risk. The gap has stayed open because "owner's pace" has no written trigger, so
each tranche becomes a fresh discretionary timing call (and the discretion skews toward
inaction on red days — exactly backwards for a scheduled accumulator). A calendar removes
the daily decision: deploying becomes a mechanical act, not a market call.

Target: reach the ¥217,000 Sleeve A capital target (≈180 units of 1568.T at ~¥977 basis).
Deployed to date: 60 units (2026-07-14). Remaining: ~120 units.

## Options (owner picks one; all mechanical, all owner-executed)

**Option 1 — Fixed weekly tranche (recommended default).**
Deploy 60 units of 1568.T every Wednesday close-decision → next-day fill, until the ¥217k
target is reached, regardless of price. ~2 more tranches. Simplest; removes timing
entirely; classic dollar-cost-averaging discipline.

**Option 2 — Value-averaging to the band.**
Each week, deploy however many units bring β-adjusted exposure to a rising weekly target
(e.g. +0.15× NAV/week until within band). Buys more when NAV/price has fallen, less when
risen — mechanically counter-cyclical, but larger tranches on down weeks (owner must be
comfortable with that).

**Option 3 — Price-grid (accumulate on weakness).**
Pre-set 1568.T buy rungs (e.g. every −3% from ¥980: 950 / 920 / 890 …), a fixed unit count
per rung. Only fills on dips. Risk: in a straight-up tape it never completes and the band
gap stays open — must pair with a time backstop (e.g. "any rung unfilled after 4 weeks →
deploy at market").

## Guardrails (apply to whichever option)

- Every tranche uses **live-verified price at order time**, never stale EOD
  (the 1346.T/1568.T price-verification discipline).
- 1568.T is a **daily-rebalanced 2× ETF**: in a prolonged choppy correction it bleeds to
  volatility decay. The schedule is for *accumulating the band*, not for backing up the
  truck; it stops at the ¥217k target and does not chase above the band.
- Exposure is re-checked each afterclose against the band (Rule 17.2). If a tranche would
  push exposure above 1.6×, it is deferred.
- Sleeve B (¥60k basket) and the 8035.T wind-down proceeds re-enter this same schedule as
  available cash; they do not create a parallel timing decision.

## To activate

Owner confirms an option → add `sleeve_a_schedule` block to `configs/risk_mandate.json`
(cadence, unit count, target) via a Rule 4 Change Log entry. The daily afterclose can then
surface "next scheduled tranche: N units on <date>" as an advice line. No auto-execution.
