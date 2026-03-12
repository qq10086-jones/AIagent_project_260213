# M10 Load Test Specification

- Date: 2026-03-12
- Scope: `T-31` quantified load and recovery validation spec for M10
- Authority: `coding_team_v0 / webapp_crm / fe_led` limited enforced cohort only
- Status: APPROVED FOR EXECUTION

---

## 1. Goal

This spec defines the first quantitative load-validation pass after M10 Phase 2 canaries A/B/C and observability correlation completed.

The purpose is not broad rollout. The purpose is to measure whether the current limited enforced posture can sustain controlled concurrent workflow traffic without:

- losing terminal results
- duplicating workflow finalization
- collapsing `fe_safe` parallel DAGs into unexplained partial failures
- regressing result-consumer stale-message recovery

---

## 2. Fixed Boundary

This spec is valid only under the following runtime boundary:

- `workflow_type = coding_team_v0`
- `project_type = webapp_crm`
- `input_class = fe_led`
- `master_enabled = true`
- `dynamic_routing_enabled = true`
- `router_mode = dynamic_routing_enforced`
- no cohort widening
- no classifier fallback disabling
- no production-risking config mutations outside the parameters listed in this spec

---

## 3. Test Mix

The workload must use only synthetic or controlled CRM tasks that map to the approved cohort and produce auditable artifacts.

Required task mix for one execution batch:

| Class | Share | Shape | Target |
|------|------:|------|------|
| Short | 40% | pure / near-pure FE-safe DAG | low-contention branch validation |
| Medium | 40% | standard `impl_be + impl_fe + qa + release` | default enforced path |
| Long | 20% | artifact-heavy `fe_safe` DAG with release-pack evidence | tail-latency and finalization stress |

Per-class requirements:

- Short:
  - 6-step DAG still required
  - lightweight PM/arch payloads
  - minimal artifact bodies
- Medium:
  - same structure as `T-23c`
  - disjoint `target_paths`
  - full `go_no_go_result.json`
- Long:
  - same DAG as Medium
  - larger markdown/json artifacts
  - full context-budget reports on every step

Recommended first execution scale:

- total workflow runs: `30`
- expected branch tasks:
  - sequential stages: `30 * 4 = 120` (`pm_spec`, `arch_design`, `qa_verify`, `release_pack`)
  - parallel impl stages: `30 * 2 = 60`
  - total workflow steps: `180`

Stretch batch after first pass is green:

- total workflow runs: `60`
- total workflow steps: `360`

---

## 4. Latency Assumptions

This spec uses bounded assumptions so pass/fail is interpretable.

LLM latency assumption bands:

| Role band | P50 assumption | P95 assumption |
|------|------:|------:|
| PM / QA / Release | `2s` | `8s` |
| Architect | `4s` | `12s` |
| Impl FE / Impl BE | `10s` | `30s` |

System-level baseline reference already observed:

- `execution_dispatch` P50 baseline: `6548 ms`
- `execution_dispatch` P95 baseline: `10834 ms`

Directional expectations for this load pass:

- `policy_evaluation` P95 should remain `< 200 ms`
- routing overhead percentage should remain `< 5%` of end-to-end workflow latency
- result-consumer stale recovery median should remain `< 30 s`

---

## 5. Redis / Consumer Parameters

Execution must use explicit stream and recovery parameters derived from current code defaults unless an execution artifact records an override.

Streams / groups:

- task stream: `stream:task`
- coding task stream: `stream:task:coding`
- result stream: `stream:result`
- DLQ stream: `stream:task:dlq`
- worker consumer group: `cg:workers`
- result consumer group: `cg:orchestrator`

Consumer and recovery settings:

- result consumer count: `1`
  - current implementation hard-codes consumer id `orchestrator-1`
- worker consumer group count: `1` logical group, multiple workers allowed within group
- result-consumer `XREADGROUP COUNT`: `20`
- result-consumer `XREADGROUP BLOCK`: `5000 ms`
- result-consumer `XAUTOCLAIM min idle`: `30000 ms`
- result-consumer `XAUTOCLAIM batch size`: `20`
- watchdog interval: `30 s`
- running timeout: `900 s`
- queued timeout: `21600 s`
- watchdog auto-DLQ: enabled

`maxlen` policy for `T-32`:

- do not enable approximate trimming during the first baseline run
- if stream growth becomes a practical issue during execution, record the chosen `MAXLEN ~` value explicitly in the execution artifact before using it

---

## 6. Required Observability

Each load run must preserve and later summarize:

- `routing_decision_log`
- `waterfall_stage_log`
- workflow run and step status rows
- task terminal results
- watchdog timeout / stale events
- `go_no_go_result.json`
- `strict_canary_report.json`
- release-pack `run_manifest.json` and `run_summary.md`

For the sampled runs in the final report, the following stages must be queryable by the same parent `run_id`:

- `policy_evaluation`
- `branch_completion_be`
- `branch_completion_fe`
- `qa_admission`
- `release_pack_readiness`

---

## 7. Pass / Fail Criteria

`T-32` passes only if all of the following are true:

1. No lost terminal results.
   - every dispatched task reaches exactly one durable terminal state
2. No duplicate workflow finalization.
   - every workflow run ends in exactly one terminal workflow state transition
3. No unexplained partial failures.
   - any `partial_failure` must map to an injected or recorded fault condition
4. No rollback-triggering misroutes.
   - no approved `fe_safe` DAG enters conflict or validation failure due to routing alone
5. Release-pack closure remains intact.
   - `go_no_go_result.json` exists for every succeeded workflow
   - `GO` rate for the approved synthetic workload is `100%`
6. Latency targets hold.
   - `policy_evaluation` P95 `< 200 ms`
   - routing overhead percentage `< 5%`
   - stale recovery median `< 30 s`
7. Queue safety holds.
   - no task remains indefinitely in Redis PEL
   - no DLQ growth without corresponding watchdog / failure evidence

`T-32` fails immediately if any of the following occur:

- duplicate finalization detected
- missing `go_no_go_result.json` on a succeeded workflow
- stale result message remains unrecovered beyond `60 s`
- workflow success recorded without complete release-pack evidence

---

## 8. Execution Outline For T-32

The first execution pass should run in three stages:

1. Warm-up batch
   - `6` workflows
   - verify stream health and artifact completeness
2. Baseline batch
   - `30` workflows using the required task mix
   - record full latency and recovery statistics
3. Stretch batch
   - `60` workflows only if baseline batch passes with no P0/P1 failures

---

## 9. Required Output Artifact

`T-32` must publish a single execution artifact bundle containing:

- run metadata and exact config snapshot
- task mix summary
- workflow counts and terminal-state distribution
- duplicate / loss checks
- `policy_evaluation`, `execution_dispatch`, `qa_admission`, `release_pack_readiness` P50/P95
- routing-overhead percentage
- stale-result recovery statistics
- DLQ counts
- sampled correlated `run_id` traces
- final verdict: `PASS` or `FAIL`

Recommended path:

- `orchestrator/artifacts/validation/m10_load_test/<timestamp>/m10_load_test_report.json`
- `orchestrator/artifacts/validation/m10_load_test/<timestamp>/m10_load_test_report.md`

---

## 10. Next Step

With this spec recorded, the next authorized step is:

- `T-32`: execute the M10 load test and publish the baseline artifact
