# OpenClaw Nexus Engineering Task List - M10 (v2 - QA/Arch Reviewed)

**Date:** 2026-03-12
**Theme:** Execution Promotion, Routing Enforcement & System Resilience (M10)

## Overview: M10
M10 is about safely transitioning the system from highly-guarded observation (M9) to autonomous, high-concurrency production execution. Based on deep QA and Architectural reviews, M10 is explicitly structured to avoid assuming "advisory" success translates seamlessly to "enforced" safety. It introduces strict atomic promotion semantics, canary rollouts for parallel routing, and rigorous quantitative load testing.

---

## Phase 0: M9 Governance Closeout (Hard Gate)
*M9 cannot be closed on interrupted or dirty artifacts. A clean run is mandatory.*

- [x] **T-00 (Hard Gate):** Execute a complete, uninterrupted, and authoritative `full-slice revalidation` of the 4-case shadow cohort. Obtain a clean `4/4` artifact before proceeding.
- [x] **T-01:** Update `PROGRESS_LATEST.md` and M9 documentation using the T-00 artifact as the authoritative operational reference for resolving `C-BUG-01`.
- [x] **T-02 (Config Cleanup):** Consolidate runtime authority sources. Eliminate compatibility copies of rollout/cohort configs between `root/configs` and `orchestrator/configs` to prevent drift.
- [x] **T-03:** Merge `worker-coder` stability fixes and test patches into the main trunk.
- [x] **T-04:** Officially declare M9 **CLOSED**.

---

## Phase 1: Execution Promotion Engine (Atomic & Safe)
*Transitioning from `shadow` mode to `promote` mode requires strict consistency guarantees, not just file copying.*

- [ ] **T-11 (Review & Extend):** Review the existing M9 `promotion_workspace.js` scaffold (preflight, shadow mode, out-of-scope blocking). Define the delta for M10: strict patch generation, diff semantics, and rollback atomicity.
- [x] **T-12 (ADR: Promotion Consistency):** Draft an Architecture Decision Record (ADR) detailing:
  - Baseline Authority: (Snapshot manifest vs. Git tree hash).
  - Conflict Unit: (File-level vs. Hunk-level).
  - Promotion Atomicity: (Temp apply + rename vs. Journaled copy).
  - Failure Semantics: (`PROMOTION_CONFLICT` vs. partial aborts).
- [ ] **T-13 (Detector Impl):** Implement the `Conflict Detector` based on the T-12 ADR to catch workspace drift during step execution.
- [ ] **T-14 (Executor Impl):** Implement the `Atomic Promote Executor`. Ensure partial applies are impossible (all-or-nothing). Include rigorous rollback-proof tests.
- [ ] **T-15 (Validation):** Run the standard `4-case cohort` using `promote` mode. Verify zero data loss and exact target path adherence.

---

## Phase 2: Dynamic Routing Enforcement (Graduated Canary)
*Advisory accuracy does not guarantee enforced safety. We must rollout routing in graduated stages.*

- [ ] **T-21 (Observation Review):** Review M8/M9 advisory logs. Acknowledge the statistical limitations of the 89-sample set and prepare for enforced behavioral shifts.
- [ ] **T-22 (Configuration):** Update `configs/production_parallel_rollout.json` to switch `router_mode` to `dynamic_routing_enforced`.
- [ ] **T-23a (Enforced Canary A):** Execute low-risk, pure UI tasks with no cross-module dependencies in enforced parallel mode.
- [ ] **T-23b (Enforced Canary B):** Execute `FE + BE` parallel tasks where `target_paths` are strictly isolated and disjoint.
- [ ] **T-23c (Enforced Canary C):** Execute full `fe_safe` DAGs requiring complex merge, verification, and release-pack tracing.
- [ ] **T-24 (Observability):** Verify `waterfall_trace_service` accurately correlates parallel task execution timelines to the single parent `run_id`.

---

## Phase 3: Chaos Engineering & Quantified Load Testing
*Moving beyond vague "10+ tasks" to strict, metric-driven resilience verification.*

- [ ] **T-31 (Load Test Spec):** Draft a rigorous Load Test Specification defining:
  - Task mix (Short/Medium/Long duration).
  - P50/P95 LLM latency assumptions.
  - Redis consumer group count, `XAUTOCLAIM` intervals, and `maxlen`.
  - Pass/Fail metrics (e.g., zero lost terminal results, zero duplicate finalizations).
- [ ] **T-32 (Load Test Execution):** Execute the load test against the T-31 spec and publish the performance baseline artifact.
- [ ] **T-33 (Failure Injection):** Perform Chaos Testing:
  - Inject main-workspace drift just before a `promote` triggers to verify `PROMOTION_CONFLICT` fires.
  - Force a worker container crash/timeout during an enforced parallel execution to verify safe DAG branch collapse.
  - Simulate DB connection drops during `Result-Consumer` XAUTOCLAIM recovery.

---
## Exit Criteria for M10 (Quantitative & Auditable)

1. **Promotion Safety:** 
   - `PROMOTION_CONFLICT` catch rate matches injected drift rate exactly.
   - Partial apply rate = `0`. Data loss = `0`.
2. **Enforced Parallel Execution:** 
   - 100% success rate on Canary A, B, and C DAG templates in enforced mode.
   - Zero rollback-triggering misroutes caused by the AI classifier during parallel execution.
3. **Performance:** 
   - On standard complex benchmark DAGs, `P50` end-to-end latency improved by ≥ `X%` and `P95` improved by ≥ `Y%` compared to forced-sequential.
   - Stale recovery median time < `Z` seconds under maximum load.
4. **Observability:** 
   - Waterfall traces, routing audit logs, and promotion artifacts are perfectly correlated by `run_id`. Any promotion failure has a deterministic, queryable rejection reason.