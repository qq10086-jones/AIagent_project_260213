# Worker-Coding Residual Issues

- Date: 2026-03-11
- Scope: post-cohort follow-up after live verification enforcement
- Status: reopened on 2026-03-12 for live result-consumer recovery follow-up

## Latest authoritative cohort artifact

- failing signal reference:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-57-01-157Z/worker_coding_cohort_result.json`
- current passing reference:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T12-56-17-290Z/worker_coding_cohort_result.json`
- latest debug post-restart reference:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T23-20-45-381Z/worker_coding_cohort_result.json`

## Historical baseline

- total: `4`
- pass: `4`
- fail: `0`
- partial: `0`

## Latest operational status

- post-restart debug shadow cohort:
  - total: `2`
  - pass: `2`
  - fail: `0`
  - partial: `0`
- post-restart full four-case rerun:
  - started but interrupted before completion
  - no new authoritative full-slice result should be claimed yet

## Resolved issues

1. stale `worker-coder` runtime source under container `/app` was corrected so live validation now runs current workspace code.
2. orchestrator request building now preserves worker-coding contract fields including `verification_plan`.
3. single-file implementation targets no longer trigger out-of-scope fallback stub writes under `worker-coder`.
4. cohort result aggregation now treats verification supersets as satisfying declared verification targets.
5. earlier `4 partial` and `0/4 fail` artifacts remain useful as historical evidence, but neither should be treated as the current readiness state.
6. orchestrator workflow finalization now fails closed instead of leaving runs stuck in `running`.
7. orchestrator result-consumer now reclaims stale pending result-stream messages on restart/recovery.

## Important environment note

- task-class authority is now config-backed:
  - `configs/registry/worker_coding_task_classes.json`
- orchestrator and worker container path issues around `shared/worker_coding_contract.mjs` were removed by moving to config-visible authority loading.

## Remaining issue

1. a live full four-case shadow rerun after the result-consumer recovery fix was started but interrupted before completion, so today the full-slice post-fix evidence is still incomplete.
2. one pre-fix live run showed `C-BUG-01` failing not because worker execution failed, but because the task result was consumed too late and the workflow had already been failed by `TASK_QUEUED_STALE`.
3. current operational follow-up is therefore not coding-logic debugging; it is end-to-end revalidation of the full cohort after the consumer recovery fix.

## Remaining caution

1. current evidence is still deterministic-provider-backed worker-coding validation, not broader uncontrolled business-task diversity.
2. single-file scope fallback should remain a permanent regression class with targeted tests.
3. future cohort expansion should proceed by adding harder scenarios, not by re-debugging the already-closed root causes from the earlier four-case slice.

## Suggested next order

1. rerun the full four-case shadow cohort now that result-consumer pending recovery is landed.
2. if the rerun returns `4/4 pass`, update governance closeout notes to use the post-restart artifact as the latest live operational reference.
3. keep regression coverage for single-file target fallback, cohort verification superset handling, workflow finalization fail-close, and result-consumer pending recovery.
