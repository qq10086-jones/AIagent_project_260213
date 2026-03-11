# Worker-Coding Residual Issues

- Date: 2026-03-11
- Scope: post-cohort follow-up after live verification enforcement
- Status: open

## Latest authoritative cohort artifact

- `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T08-57-01-157Z/worker_coding_cohort_result.json`

## Current truthful result

- total: `4`
- pass: `0`
- fail: `4`
- partial: `0`

## Residual issues

1. `fe_create`, `fe_modify`, `bug_fix` currently fail as `verification_failure`.
2. `be_create` currently fails as `coding_logic_failure`.
3. Earlier `4 partial` cohort outputs should not be used for readiness claims; those runs happened before live verification enforcement was fully truthful.

## Important environment note

- task-class authority is now config-backed:
  - `configs/registry/worker_coding_task_classes.json`
- orchestrator and worker container path issues around `shared/worker_coding_contract.mjs` were removed by moving to config-visible authority loading.

## Suggested next debug order

1. Reproduce one FE cohort case and inspect the first failed implementation-step verification artifact.
2. Reproduce the `be_create` case and inspect why the generated code still fails after retry.
3. Only after one FE and one BE case are understood, rerun the full four-case cohort.
