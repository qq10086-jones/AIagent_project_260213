# OpenClaw Nexus vNext
## Engineering Task List - Milestone 5
## Date: 2026-03-08
## Focus: Structured Patch Execution + Context Budget Control + Workflow Parallelization Readiness

---

## 0. Current System State Summary

Milestone 4 is closed and provides the baseline for M5.

### Implemented in M4

1. `LLM Routing Layer` is complete
- `llm_providers.json` and `llm_role_policy.json` exist and are wired
- `src/vnext/llm_dispatcher.js` handles role-based model selection, retry, fallback, and provider validation
- PM / Architect / Backend / Frontend / QA / Release execution paths route through the dispatcher

2. `Coding Team Workflow` is complete
- PM -> Architect -> Backend -> Frontend -> QA -> Release chain is runnable
- typed handoff schemas exist across the execution chain
- QA and Release artifact contracts are validated
- end-to-end canary passes

3. `Memory Layer` is complete at minimal governed scope
- Architect prompt injects read-only project memory context
- successful workflows write task history and ADR artifacts back to memory
- memory layer canary passes

4. `Verification baseline` is complete
- M4 canaries and targeted integration tests pass
- full orchestrator test suite passes: `cmd /c npm --prefix orchestrator test`

### Why M5 exists

M4 intentionally kept execution simple and stable:
- implementation steps output full files instead of structured diffs
- workflow execution remains strictly sequential
- context growth is only partially mitigated by fallback logic, not explicitly measured

M5 addresses those deferred limitations without changing the North Star pipeline.

---

## 1. Objective

Milestone 5 improves execution efficiency and control while preserving the M4 pipeline:

1. Replace full-file implementation output with structured diff / AST-aware patch execution where practical
2. Add explicit context budget tracking across workflow steps
3. Prepare the workflow engine for safe DAG-based parallel execution, beginning with BE / FE independence analysis and gated execution

This milestone is about execution quality and scalability, not new user-facing domains.

---

## 2. Embedded Design Decisions

### D1 - North Star pipeline remains unchanged

M5 does not alter the primary execution path:

`Human -> Discord -> Brain Router -> Task Envelope -> LLM Dispatcher -> Workflow Engine -> Coding Team -> Artifacts`

M5 only improves how implementation steps produce and transport artifacts.

### D2 - Diff-first execution is the new default for implementation steps

Backend and Frontend steps should prefer structured patch outputs over full-file replacement when the workspace already contains target files.

### D3 - Full-file fallback remains available

If AST/diff application fails or the target file does not exist, the system may fall back to full-file write. Diff-first is preferred, not absolute.

### D4 - Context budget is a governed artifact

Each workflow step must emit measurable context size metadata so overflow risk is observable, not inferred.

### D5 - Parallel execution is gated, not assumed

BE and FE parallelization may only begin after explicit dependency analysis confirms there is no blocking upstream handoff requirement for the specific run.

### D6 - No expansion beyond current system domain

M5 does not introduce:
- new agent teams
- vector memory
- Brain Router LLM classification
- dashboard/UI work
- adaptive model routing

---

## 3. Workstream Overview

| ID | Name | Type | Blocks |
|----|------|------|--------|
| WS-19 | Structured Diff / Patch Execution | A | WS-20, WS-21 |
| WS-20 | Context Budget Tracking | A | WS-21 |
| WS-21 | Workflow DAG / Parallel Execution Readiness | A | E2E closure |

All three workstreams are Type A because they directly improve the Coding Team execution pipeline.

---

## 4. Detailed Task List

---

## WS-19 Structured Diff / Patch Execution

**Type:** Type A / Critical Path  
**Pipeline node:** Coding Team implementation execution path

---

### WS-19-01 Define Structured Patch Contract

**Deliverables:**
- `orchestrator/contracts/coding_team_patch_bundle.schema.json`
- valid and invalid fixtures under `orchestrator/contracts/fixtures/`

**Required fields:**
- `bundle_id`
- `step_id`
- `mode: "structured_patch" | "full_file_fallback"`
- `operations`: array of patch operations
- `target_files`: array of repo-relative target files
- `summary`

**Acceptance criteria:**
- schema validates valid fixture
- schema rejects malformed operation lists
- schema is added to registry validation coverage

---

### WS-19-02 Implement Patch Bundle Application Service

**Deliverable:** `orchestrator/src/domain/patch_bundle_service.js`

**Requirements:**
- apply structured patch operations deterministically
- support at minimum:
  - `replace_range`
  - `insert_after_anchor`
  - `delete_range`
  - `create_file`
- reject path traversal and out-of-root writes
- return typed error on failed application

**Acceptance criteria:**
- unit tests cover success and failure paths
- file system writes remain scoped to workspace root

---

### WS-19-03 Wire Backend / Frontend Steps to Diff-First Execution

**Deliverables:**
- updated execution path for `impl_be`
- updated execution path for `impl_fe`

**Requirements:**
- prefer structured patch bundle when target file already exists
- fall back to full-file output when patch mode is unavailable or invalid
- record which mode was used in step result

**Acceptance criteria:**
- BE and FE step results expose `execution_mode_used`
- validators accept both patch-applied and full-file fallback outcomes

---

### WS-19-04 Structured Patch Canary

**Deliverable:** `orchestrator/scripts/canary_patch_bundle.js`

**Coverage:**
- patch applies successfully to an existing file
- create_file works
- malformed patch fails with typed error
- path traversal attempt is rejected
- fallback mode is observable

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/patch_bundle/`

---

## WS-20 Context Budget Tracking

**Type:** Type A / Critical Path  
**Pipeline node:** LLM execution and workflow artifact transport

---

### WS-20-01 Define Context Budget Metrics Contract

**Deliverables:**
- `orchestrator/contracts/context_budget_report.schema.json`
- fixture files

**Required fields:**
- `step_id`
- `artifact_count`
- `bytes_total`
- `largest_artifact_bytes`
- `prompt_chars`
- `status: "ok" | "warning" | "overflow_risk"`

**Acceptance criteria:**
- schema validation passes
- contract is consumable by workflow summary tooling

---

### WS-20-02 Emit Per-Step Context Budget Reports

**Deliverables:**
- context budget generation inside workflow execution path
- one report per step under release artifact root, for example `metrics/context_budget_{step}.json`

**Requirements:**
- measure prompt size and artifact payload size before dispatch
- classify warning and overflow risk using explicit thresholds
- attach report path into step result JSON

**Acceptance criteria:**
- each Coding Team step emits a budget report
- workflow summary can reference all reports

---

### WS-20-03 Add Context Budget Aggregation to Release Pack

**Deliverables:**
- release pack summary includes context budget overview
- artifact manifest includes context budget reports

**Acceptance criteria:**
- release pack exposes step-level budget status
- missing reports fail validation

---

### WS-20-04 Context Budget Canary

**Deliverable:** `orchestrator/scripts/canary_context_budget.js`

**Coverage:**
- normal-size run reports `ok`
- oversized synthetic artifact reports `warning` or `overflow_risk`
- release pack includes aggregated budget metadata

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/context_budget/`

---

## WS-21 Workflow DAG / Parallel Execution Readiness

**Type:** Type A / Critical Path  
**Pipeline node:** Workflow engine

---

### WS-21-01 Define Workflow DAG Metadata Contract

**Deliverables:**
- workflow definition extension for `depends_on`
- schema update for workflow registry if needed

**Requirements:**
- steps may declare zero or more upstream dependencies
- sequential behavior remains default when `depends_on` is absent

**Acceptance criteria:**
- current workflows still load without changes
- DAG metadata is explicit and schema-valid

---

### WS-21-02 Add DAG Scheduling Primitive to Workflow Engine

**Deliverables:**
- workflow engine support for dependency-based readiness

**Requirements:**
- engine may dispatch multiple ready steps
- no step may dispatch before all dependencies succeeded
- failed upstream step blocks all dependent downstream steps

**Acceptance criteria:**
- integration test covers dependency readiness
- sequential workflows still behave identically

---

### WS-21-03 BE / FE Parallelization Feasibility Gate

**Deliverables:**
- explicit policy deciding when FE may run in parallel with BE

**Requirements:**
- if FE requires BE handoff, remain sequential
- if architect handoff plus project type marks FE-safe parallel mode, allow parallel dispatch
- gating decision must be logged and testable

**Acceptance criteria:**
- both sequential and parallel cases covered by integration tests
- no silent policy fallback

---

### WS-21-04 DAG / Parallel Execution Canary

**Deliverable:** `orchestrator/scripts/canary_workflow_dag.js`

**Coverage:**
- sequential workflow remains unchanged
- dependency-aware workflow dispatches independent steps in parallel
- upstream failure blocks dependent step dispatch

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/workflow_dag/`

---

## 5. Suggested Execution Order

```
WS-19-01  Patch bundle schema
WS-20-01  Context budget schema
      -> WS-19-02  Patch application service
      -> WS-20-02  Per-step budget reports
WS-19-03  Diff-first BE/FE execution
WS-19-04  Patch bundle canary
WS-20-03  Release pack budget aggregation
WS-20-04  Context budget canary
WS-21-01  DAG metadata contract
WS-21-02  DAG scheduling primitive
WS-21-03  BE/FE parallelization gate
WS-21-04  DAG canary
```

---

## 6. Definition of Done for Milestone 5

Milestone 5 is complete when:

- structured patch bundle schema exists and is validated
- Backend and Frontend execution support diff-first mode with full-file fallback
- patch application service has unit coverage and canary coverage
- each Coding Team step emits context budget metadata
- release pack aggregates context budget reports
- workflow engine supports dependency-based dispatch without breaking sequential workflows
- BE / FE parallel readiness is explicit, gated, and tested
- all new canaries pass
- full orchestrator test suite still passes

---

## 7. Non-Scope for Milestone 5

- no new user-facing product domain
- no Brain Router LLM classification
- no vector or semantic retrieval memory
- no adaptive multi-model routing
- no distributed multi-node orchestrator
- no UI/dashboard work
- no cross-project shared memory

---

## 8. Risks

| Risk | Severity | M5 Disposition |
|------|----------|----------------|
| Patch application corrupts files | High | mitigate with typed patch schema + deterministic application tests |
| Diff-first output becomes brittle on anchor mismatch | Medium | keep full-file fallback path |
| Context budget metrics become advisory only | Medium | require report presence in release validation |
| Parallel dispatch causes hidden ordering regressions | High | dependency gating + sequential fallback + canary coverage |

---

## 9. Next Step

This task list should be reviewed and approved before any M5 implementation begins.
