# OpenClaw Nexus — Roadmap On Hold

**Frozen**: 2026-04-17
**See also**: [NEXUS_PAUSE_STATE.md](./NEXUS_PAUSE_STATE.md)

This is the plan that existed when development paused. It is **not a commitment**. When/if work resumes, the author should re-evaluate whether these items are still the right bets given what was learned during the Hermes exploration period.

---

## Tier 1 — Unblock v3.7.3 (~1 day of work when resuming)

These three items are all that's needed to **close the v3.7.3 loop** and get a conclusive E2E verdict:

1. **Restart orchestrator container and re-run E2E**
   - `docker compose restart orchestrator` then trigger a webapp_crm run
   - Expected: `impl_be` prompt contains the literal string `WEBAPP_CRM MANDATORY`; BE artifacts include `activity_log` table + `/api/activity` endpoint + DELETE with 404
   - If B3 verifies, v3.7.3 score likely lands at 8/10 algorithmic

2. **Reproduce or dismiss the `impl_fe_skeleton` timeout**
   - Seen once in `f7f03662`, not yet repeated
   - If repeats: instrument OpenCode adapter round-trip timing; check skeleton prompt length
   - If doesn't repeat after 3 runs: treat as transient, mark resolved

3. **Fix PM step to call a real LLM** (root-cause of 3-score gap since v3.6)
   - Current: returns scaffold template, goal truncated mid-word at ~45 lines
   - Without this, C path (spec.md injection) is structurally unable to help, because the spec itself has no keywords for v3.7.2 features
   - Location: worker-coder PM task handler (exact file: search for `pm_spec` scaffold emitter in `worker-coder/`)
   - This is the single highest-ROI fix remaining

## Tier 2 — Lift score from 7 → 8 (~1 week)

4. **static_audit: dry_run → blocking + retry feedback loop**
   - Flip `static_audit_mode: blocking` in config
   - On scanner finding, feed finding back to the producing step's LLM as a retry hint (impl_be with DELETE 404 finding, POST validation finding, etc.)
   - Requires: refinement re-entry path to accept scanner findings as a refinement trigger (currently only QA failures trigger refinement)
   - Covers the known "prompt layer blind spot": BE contract issues prompt rules can't prevent

5. **Triage 7 pre-existing test failures**
   - registry validator / workflow integration tests ×3 / arch interface contract regex / context packet / smoke_test run_acceptance_test.mjs regex
   - None block pipeline; they block a green CI signal

## Tier 3 — Address the 3/10 user-terminal score (~2 weeks)

The user terminal-acceptance score has been stuck at 3/10 since v3.4. Known gaps from user review:

6. **Delete affordances** — CRM has delete endpoints but the FE doesn't always expose them ergonomically
7. **Feedback on actions** — success/error toasts, disabled states, loading indicators
8. **Multi-module FE** — FE tends to collapse to a single module; navigation between views is weak

These are spec-layer problems, not model-layer. Likely fix: domain acceptance pack expansion for webapp_crm to enumerate UX requirements, not just API contract.

## Tier 4 — Architectural bets (re-evaluate after Hermes exploration)

These ideas came up during the Hermes comparison on 2026-04-17. They are **speculative** and should be re-examined with fresh eyes after the Hermes learning period:

9. **Procedural memory / skill library**
   - Inspired by Hermes Agent's `/skills/` directory
   - Every successful webapp_crm run emits a reusable "pattern" (activity_log schema, delete semantics, quick-actions layout) indexed by task signature
   - Next run retrieves matching skills, injects into prompt instead of the hardcoded B-path
   - Would replace the current manual-prompt-engineering cycle (v3.7.2, v3.7.3 B paths) with self-accumulating library

10. **Cross-run FTS5 memory over artifacts**
    - Index historical run artifacts and failure modes
    - Warm-start new runs by retrieving similar past tasks

11. **Interactive checkpoint mode**
    - Allow pause-review-redirect between pipeline stages
    - Trade some determinism for mid-flight steerability

12. **User model** (Honcho-style dialectic)
    - Persist user-specific preferences (tech stack, code style, acceptance bar)
    - Feed as system prompt prefix across runs

## Explicitly NOT doing

- **MiniMax / stable_cloud_lane**: decided 2026-04-11 that gemma4:26b meets quality bar; cloud lane stays available but policy is local-first. Subscription not renewed.
- **gemma4:31b**: removed from escalation chain 2026-04-12 — CPU offload on 24GB GPU made it unusably slow.
- **project_planner**: `project_planner_enabled: false` stays off unless Tier 3 work shows it's needed.

## Re-start criteria

Come back to Nexus when **at least one** of:

- Hermes exploration reveals a specific pattern that's clearly missing from Nexus and worth building (write a one-page comparison before resuming)
- A real project needs Nexus's batch-factory model and the 7/10 quality is insufficient
- An external user expresses interest in the system (currently: solo project)

Do **not** resume because of nostalgia or sunk-cost. The stack runs; it's usable now.
