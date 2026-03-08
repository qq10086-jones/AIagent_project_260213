# M5 Progress Update - DAG Scheduling, Parallelization Gate, and DAG Canary

## Date
2026-03-08

## Scope
This update records completion of the remaining M5 workflow-engine work under the approved scope:

- `WS-21-02 Add DAG Scheduling Primitive to Workflow Engine`
- `WS-21-03 BE / FE Parallelization Feasibility Gate`
- `WS-21-04 DAG / Parallel Execution Canary`

This work was implemented under the authoritative constraint set:

- `docs/01_design/system/260308/260308_2053/OpenClaw_Nexus_Design_Document_v3.1.md`
- `docs/01_design/system/260308/260308_2053/OpenClaw_Nexus_Engineering_Task_List_M5_v2.md`
- `docs/01_design/system/260308/260308_2053/OpenClaw_Execution_Governance_Scope_Control_v3.md`

No out-of-scope expansion was introduced.

## Completed In This Update

### 1. WS-21-02 DAG scheduling primitive

Implemented dependency-aware workflow scheduling with support for:

- sequential default behavior when `depends_on` is absent
- dispatch of multiple ready steps in parallel
- blocking of downstream dispatch until all dependencies succeed
- `partial_failure` workflow state for mixed parallel outcomes
- retry eligibility preservation by allowing resume on failed step only

Key implementation files:

- `orchestrator/src/domain/dag_scheduler.js`
- `orchestrator/src/workflow_engine.js`
- `orchestrator/src/data/workflow_repository.js`
- `orchestrator/src/domain/workflow_resume.js`
- `orchestrator/src/domain/workflow_state.js`

### 2. WS-21-03 BE / FE parallelization feasibility gate

Implemented an explicit policy and runtime layer controlling when `coding_team_v0` may dispatch `impl_be` and `impl_fe` in parallel.

Gate behavior:

- remains sequential if FE still requires BE handoff
- remains sequential unless the project type explicitly enables FE-safe mode
- allows BE/FE parallelization only when:
  - project type marks FE-safe parallel mode as enabled
  - architect handoff explicitly declares:
    - `parallelization.fe_safe_parallel = true`
    - `parallelization.requires_be_handoff = false`

Key implementation files:

- `orchestrator/src/domain/workflow_parallelization_policy.js`
- `orchestrator/src/domain/workflow_parallelization_runtime.js`
- `orchestrator/contracts/coding_team_arch_handoff.schema.json`
- `configs/registry/capability_registry.json`

Logging behavior:

- gating decision is emitted as `workflow.parallelization.gate_decided`
- both sequential-denied and parallel-approved cases are test-covered

### 3. WS-21-04 DAG / parallel canary

Added:

- `orchestrator/scripts/canary_workflow_dag.js`
- npm script entry: `canary:workflow_dag`

Canary coverage:

- sequential workflow remains unchanged when FE-safe gate is not enabled
- coding workflow dispatches BE/FE in parallel when gate is approved
- mixed success/failure produces observable `partial_failure`
- dual failure produces `failed`
- downstream join step remains blocked after upstream failure
- artifact isolation is preserved (`impl/be_changes/` vs `impl/fe_changes/`)

Canary artifact:

- `orchestrator/artifacts/canary/workflow_dag/workflow_dag_canary.json`

## Verification Performed

### DAG / Gate Tests

- `node orchestrator/test/workflow_dag.test.js` - pass
  - dependency readiness covered
  - all four error-matrix cells covered
  - sequential gate case covered
  - parallel gate case covered

### Canaries

- `node scripts/canary_workflow_dag.js` - pass
- `node scripts/canary_m4_compat.js` - pass
- `node scripts/canary_patch_bundle.js` - pass

### Config / Schema

- `node scripts/validate_registry.js` - pass

## Complexity Budget Outcome

The M5 task list required extraction of DAG logic if `workflow_engine.js` exceeded 560 lines.

Result:

- DAG logic extracted to `orchestrator/src/domain/dag_scheduler.js`
- parallelization runtime extracted to `orchestrator/src/domain/workflow_parallelization_runtime.js`
- final `workflow_engine.js` line count: `572`

This keeps the file within the approved M5 close budget ceiling of `600`.

## Current Milestone 5 Status

### WS-19 Structured Diff / Patch Execution
- WS-19-01: DONE
- WS-19-02: DONE
- WS-19-02.5: DONE
- WS-19-03: DONE
- WS-19-04: DONE

### WS-20 Context Budget Tracking
- WS-20-01: DONE
- WS-20-01.5: DONE
- WS-20-02: DONE
- WS-20-03: DONE
- WS-20-04: DONE

### WS-21 Workflow DAG / Parallel Execution Readiness
- WS-21-01: DONE
- WS-21-02: DONE
- WS-21-03: DONE
- WS-21-04: DONE

### WS-22 Governance & Regression
- WS-22-01: DONE
- WS-22-02: DONE

## M5 Overall Status

Milestone 5 implementation scope is now effectively complete at the approved workstream level.

Remaining work is documentation / closure oriented:

- refresh latest progress snapshot
- optionally add final M5 closure note
- optionally update design-document complexity table snapshot if a separate closure artifact is required

## Notes

- In the current Codex sandbox, `node --test` may still fail with `spawn EPERM`
- For this session, targeted direct execution of the relevant test file and canary scripts was used instead
