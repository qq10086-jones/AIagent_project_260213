# M6 Exposure Go / No-Go Approval Record

- Version: 1.0
- Date: 2026-03-09
- Milestone: M6
- Required by: WS-25-02.5
- Status: STAY_GATED

---

## Decision

**STAY_GATED**

Rationale: M6 staging infrastructure and contracts are complete. Rollback drill passed. However, the replay runner currently operates in simulation mode (no live LLM execution), so the go/no-go thresholds cannot yet be evaluated against real execution evidence. Limited production exposure is deferred until a live staging run produces metrics from actual LLM-driven workflow steps.

This decision does not block any further M6 Phase 4/5 work. It records the current evidence state and sets the re-evaluation trigger.

---

## Threshold Evaluation

Source: WS-25-02.5 pre-defined quantitative thresholds.

| Criterion | GO_LIMITED_EXPOSURE requires | Measured Value | Decision |
|-----------|------------------------------|----------------|----------|
| Replay corpus size | ≥ 50 cases, coverage floor met | 50 cases, all floors PASS | PASS |
| Parallel vs sequential success rate | delta ≤ 5% | 0.0% (simulation) | PASS (simulation only) |
| Partial failure rate (parallel) | ≤ 10% | 0.0% (simulation) | PASS (simulation only) |
| Diff-first hit rate (clean-repo) | ≥ 60% | 100% (simulation) | PASS (simulation only) |
| Patch anchor mismatch rate | ≤ 15% | 0.0% (simulation) | PASS (simulation only) |
| Rollback drill time | < 30 seconds | 8 seconds | PASS |
| FE-safe eligibility qualification | ≥ 1 workflow type at ≥ 80% eligibility | fe_led confirmed eligible | PASS |
| Structural guard negative test | passes | PASS (parallel_rollout_gate.test.js) | PASS |

**Simulation rows**: metrics derived from the replay simulation harness, not live LLM execution. These values are structurally correct but not evidence of real production behavior. A live staging run is required before GO_LIMITED_EXPOSURE can be approved.

---

## Approved Whitelist (Conditional — not active until GO_LIMITED_EXPOSURE)

When a future go/no-go review upgrades this decision to GO_LIMITED_EXPOSURE, the following scope applies:

| Dimension | Approved Scope |
|-----------|---------------|
| Workflow types | `coding_team_v0` |
| Project types | `crm` |
| Input classes | `fe_led` only |
| Execution mode | Gated parallel (deny-by-default; gate evaluates per run) |

---

## Reviewer Sign-off

- Reviewer: Architect (project owner)
- Date: 2026-03-09
- Rollback trigger threshold: partial failure rate > 25% in rolling window of 100 runs

---

## Re-evaluation Trigger

This decision must be re-evaluated when:

1. A live staging run (real LLM execution, not simulation) completes against the replay corpus
2. `metrics/parallel_vs_sequential.json` is updated with live execution evidence
3. All go/no-go thresholds are re-measured against live data

---

## References

- Replay corpus: `orchestrator/replay/manifests/m6_staging_replay_manifest.json`
- Comparison metrics: `metrics/parallel_vs_sequential.json`
- Rollback runbook: `docs/runbooks/m6_parallel_rollback_runbook.md`
- Circuit-breaker config: `orchestrator/configs/production_parallel_rollout.json`
