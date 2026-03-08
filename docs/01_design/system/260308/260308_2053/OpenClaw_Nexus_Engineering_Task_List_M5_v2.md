# OpenClaw Nexus vNext
## Engineering Task List - Milestone 5 (v2)
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

### 1.1 Success Metrics (NEW)

M5 success is measured against these quantitative and qualitative baselines:

| Metric | M4 Baseline | M5 Target | Measurement Method |
|--------|-------------|-----------|-------------------|
| BE/FE step average output tokens | full-file (100%) | ≥30% reduction when patching existing files | compare artifact byte size in release pack |
| Context overflow fallback events | untracked | 100% observable in release pack context budget report | `context_budget_{step}.json` presence and `status` field |
| Workflow engine DAG support | sequential only | DAG scheduling primitive passes integration tests; at least 1 synthetic parallel workflow canary passes | canary exit code |
| M4 regression | N/A | zero regression — M4 full-file sequential canary passes unchanged on M5 codebase | compatibility canary exit code |

### 1.2 User Value Summary (NEW)

Although M5 is infrastructure-focused, its outcomes directly improve end-user experience:

- **Diff-first execution** reduces token waste on large existing files, lowering truncation risk and improving implementation accuracy for real-world projects.
- **Context budget tracking** makes workflow failure causes visible before they happen, rather than surfacing as mysterious LLM quality degradation.
- **Parallel readiness** is the foundation for cutting end-to-end workflow time when BE and FE are independent — a direct latency improvement users will feel in future milestones.

---

## 2. Embedded Design Decisions

### D1 - North Star pipeline remains unchanged

M5 does not alter the primary execution path:

`Human -> Discord -> Brain Router -> Task Envelope -> LLM Dispatcher -> Workflow Engine -> Coding Team -> Artifacts`

M5 only improves how implementation steps produce and transport artifacts.

### D2 - Diff-first execution is the new default for implementation steps

Backend and Frontend steps should prefer structured patch outputs over full-file replacement when the workspace already contains target files.

### D3 - Full-file fallback remains available and feature-gated (UPDATED)

If AST/diff application fails or the target file does not exist, the system may fall back to full-file write. Diff-first is preferred, not absolute.

**Feature gate requirement:** A runtime configuration flag (`execution.diff_first_enabled`, default `true`) must exist so that diff-first mode can be disabled globally without rolling back M5 code. This supports safe production rollback if diff-first proves unstable in practice.

### D4 - Context budget is a governed artifact with configurable thresholds (UPDATED)

Each workflow step must emit measurable context size metadata so overflow risk is observable, not inferred.

Budget thresholds must be externalized in a policy file (`orchestrator/configs/context_budget_policy.json`), consistent with the established pattern of `llm_role_policy.json` as single source of truth. Thresholds must not be hardcoded in service logic.

### D5 - Parallel execution is gated, not assumed

BE and FE parallelization may only begin after explicit dependency analysis confirms there is no blocking upstream handoff requirement for the specific run.

**Scope clarification:** M5 delivers the DAG scheduling primitive and the feasibility gate. At M5 close, at least one synthetic parallel workflow canary must pass. However, the default Coding Team production workflow remains sequential unless the feasibility gate explicitly approves parallel dispatch for a given run. M5 does not promise that real user workflows will run in parallel by default.

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
| WS-22 | M5 Governance & Regression (NEW) | A | E2E closure |

All workstreams are Type A because they directly improve or protect the Coding Team execution pipeline.

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
- `operations`: ordered array of patch operations
- `target_files`: array of repo-relative target files
- `summary`

**Patch operation addressing model (NEW):**

All range-based operations (`replace_range`, `delete_range`) and anchor-based operations (`insert_after_anchor`) use **content-anchor addressing**, not line numbers. Rationale: LLM-generated line numbers are unreliable; content-based anchors are self-validating.

Each operation specifies:
- `anchor`: unique string that identifies the target location in the file
- `anchor_context_before` (optional): preceding content for disambiguation when anchor is non-unique

**Operation ordering semantics (NEW):**

The `operations` array is **ordered** and **applied sequentially**. Each operation is applied against the result of the previous operation, not against the original file snapshot. Rationale: sequential application is simpler to reason about and matches how human developers think about edits. The `patch_bundle_service.js` must validate that each operation's anchor is still resolvable after prior operations have been applied.

**Supported operations:**
- `replace_range` — replace content between `anchor_start` and `anchor_end` (content-based)
- `insert_after_anchor` — insert content after anchor match
- `delete_range` — delete content between `anchor_start` and `anchor_end` (content-based)
- `create_file` — create new file (anchor fields not required)

**Acceptance criteria:**
- schema validates valid fixture
- schema rejects malformed operation lists
- schema rejects operations with missing anchor fields (except `create_file`)
- schema is added to registry validation coverage

---

### WS-19-02 Implement Patch Bundle Application Service

**Deliverable:** `orchestrator/src/domain/patch_bundle_service.js`

**Requirements:**
- apply structured patch operations deterministically in declared order
- support at minimum:
  - `replace_range`
  - `insert_after_anchor`
  - `delete_range`
  - `create_file`
- validate anchor resolvability before each operation; emit typed error with operation index on failure
- reject path traversal and out-of-root writes
- return typed error on failed application
- when multiple operations target the same file, apply them sequentially and verify each anchor after prior operations

**Acceptance criteria:**
- unit tests cover success and failure paths
- unit tests cover multi-operation same-file scenarios with anchor shift
- file system writes remain scoped to workspace root
- estimated complexity: ≤400 lines (Layer 3 budget: 500)

---

### WS-19-02.5 Adapt Prompt Scripts for Diff-First Output (NEW)

**Deliverables:**
- `backend.impl.v2` prompt script (diff-first variant)
- `frontend.impl.v2` prompt script (diff-first variant)
- updated `prompt_script_registry.json` entries

**Requirements:**
- when target files exist, prompt must instruct LLM to output structured patch bundle format instead of full files
- prompt must include target file current content as context (or relevant excerpts for large files)
- prompt must specify the content-anchor addressing model and operation format defined in WS-19-01
- prompt must include explicit fallback instruction: if LLM cannot produce valid patch format, output full file with `mode: "full_file_fallback"`

**Cross-reference with WS-20:**
- injecting target file content into prompt increases prompt size — this must be tracked by context budget (WS-20-02)
- if injecting target file would push prompt into `overflow_risk` status, step should pre-emptively use full-file mode

**Acceptance criteria:**
- v2 prompt scripts produce valid patch bundle output on test fixture
- v1 prompt scripts remain available for fallback
- prompt size delta between v1 and v2 is measured and documented

---

### WS-19-03 Wire Backend / Frontend Steps to Diff-First Execution

**Deliverables:**
- updated execution path for `impl_be`
- updated execution path for `impl_fe`

**Requirements:**
- check `execution.diff_first_enabled` feature gate before attempting diff-first mode
- prefer structured patch bundle when target file already exists and feature gate is enabled
- use v2 prompt script when in diff-first mode, v1 when in full-file mode
- fall back to full-file output when patch mode is unavailable, invalid, or feature gate is disabled
- record which mode was used in step result

**Acceptance criteria:**
- BE and FE step results expose `execution_mode_used`
- validators accept both patch-applied and full-file fallback outcomes
- disabling feature gate forces full-file mode without code changes

---

### WS-19-04 Structured Patch Canary

**Deliverable:** `orchestrator/scripts/canary_patch_bundle.js`

**Coverage:**
- patch applies successfully to an existing file
- multi-operation patch on same file with anchor shift works correctly
- create_file works
- malformed patch fails with typed error including operation index
- anchor not found after prior operation fails with typed error
- path traversal attempt is rejected
- fallback mode is observable
- feature gate disabled → full-file mode used

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
- `injected_context_bytes` (NEW — tracks target file content injected for diff-first mode)
- `status: "ok" | "warning" | "overflow_risk"`
- `threshold_source` (NEW — reference to policy file entry used for classification)

**Acceptance criteria:**
- schema validation passes
- contract is consumable by workflow summary tooling

---

### WS-20-01.5 Define Context Budget Policy Configuration (NEW)

**Deliverable:** `orchestrator/configs/context_budget_policy.json`

**Requirements:**
- define per-role or per-step warning and overflow thresholds
- thresholds expressed in bytes and/or estimated tokens
- policy file follows the same governance pattern as `llm_role_policy.json`

**Example structure:**
```json
{
  "version": "1.0.0",
  "default_thresholds": {
    "warning_prompt_chars": 80000,
    "overflow_risk_prompt_chars": 120000,
    "warning_artifact_bytes": 500000,
    "overflow_risk_artifact_bytes": 1000000
  },
  "role_overrides": {
    "architect": {
      "warning_prompt_chars": 100000,
      "overflow_risk_prompt_chars": 150000
    }
  }
}
```

**Acceptance criteria:**
- policy file is schema-valid
- `patch_bundle_service` and context budget report generation both read thresholds from this file
- changing thresholds requires editing only this file, not service code

---

### WS-20-02 Emit Per-Step Context Budget Reports

**Deliverables:**
- context budget generation inside workflow execution path
- one report per step under release artifact root, for example `metrics/context_budget_{step}.json`

**Requirements:**
- measure prompt size and artifact payload size before dispatch
- measure injected context size (target file content for diff-first mode) separately
- classify warning and overflow risk using thresholds from `context_budget_policy.json`
- attach report path into step result JSON

**Acceptance criteria:**
- each Coding Team step emits a budget report
- workflow summary can reference all reports
- threshold values in report match policy file

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
- threshold override in policy file changes classification result

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
- engine may dispatch multiple ready steps concurrently (via `Promise.all` or equivalent)
- no step may dispatch before all dependencies succeeded
- failed upstream step blocks all dependent downstream steps

**Concurrent execution error state matrix (NEW):**

| BE Result | FE Result | Workflow State | Action |
|-----------|-----------|----------------|--------|
| success | success | continue to QA | normal path |
| success | failure | `partial_failure` | report FE failure; do not proceed to QA |
| failure | success | `partial_failure` | report BE failure; do not proceed to QA |
| failure | failure | `failed` | report both failures |

- `partial_failure` is a new workflow state that must be logged and queryable
- on `partial_failure`, workflow may retry the failed step only (not the succeeded step)

**Artifact isolation (NEW):**
- parallel steps must write to isolated artifact subdirectories (`impl/be_changes/`, `impl/fe_changes/`)
- artifact merge into release pack happens only after all parallel steps succeed
- no parallel step may write to another step's artifact directory

**Complexity budget impact (NEW):**
- estimated addition to `workflow_engine.js`: +80–120 lines for DAG scheduling + concurrent error handling
- current: 431 lines; projected: 511–551 lines (within 600 line budget)
- if implementation exceeds 560 lines, DAG scheduling must be extracted to `src/domain/dag_scheduler.js` (new file, budget: 300 lines)

**Acceptance criteria:**
- integration test covers dependency readiness
- integration test covers all four cells of the error state matrix
- sequential workflows still behave identically
- artifact isolation verified in parallel execution test

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
- partial_failure state is observable when one parallel step fails
- artifact isolation is maintained during parallel execution

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/workflow_dag/`

---

## WS-22 M5 Governance & Regression (NEW)

**Type:** Type A / Protective  
**Pipeline node:** Cross-cutting

---

### WS-22-01 M4 Compatibility Canary

**Deliverable:** `orchestrator/scripts/canary_m4_compat.js`

**Purpose:** Verify that M4 standard Coding Team workflow (sequential, full-file output) produces identical behavior on M5 codebase.

**Coverage:**
- full sequential workflow: PM → Architect → BE → FE → QA → Release
- all steps use full-file output mode (diff-first feature gate disabled)
- all M4 handoff schemas still validate
- release pack structure matches M4 expectations

**Acceptance criteria:**
- canary exits 0
- artifact written to `orchestrator/artifacts/canary/m4_compat/`

---

### WS-22-02 Update Complexity Budget Table

**Deliverables:**
- updated Section 19 of Design Document v3 to include M5 new files
- budget entries for:
  - `src/domain/patch_bundle_service.js` — budget: 400 lines
  - `src/domain/dag_scheduler.js` (conditional) — budget: 300 lines
  - `configs/context_budget_policy.json` — no line budget (config file)
- reassess `workflow_engine.js` projected line count post-M5

**Acceptance criteria:**
- all new M5 files have declared budgets before implementation begins
- no file exceeds its budget at M5 close

---

## 5. Suggested Execution Order

```
Phase 1 — Schemas (parallelizable)
  WS-19-01  Patch bundle schema + addressing model
  WS-20-01  Context budget schema
  WS-20-01.5 Context budget policy config
  WS-21-01  DAG metadata contract
  WS-22-02  Complexity budget update

Phase 2 — Core services
  WS-19-02  Patch application service
  WS-19-02.5 Prompt script diff-first adaptation
  WS-20-02  Per-step budget reports

Phase 3 — Integration
  WS-19-03  Diff-first BE/FE execution wiring
  WS-20-03  Release pack budget aggregation
  WS-21-02  DAG scheduling primitive
  WS-21-03  BE/FE parallelization gate

Phase 4 — Verification
  WS-19-04  Patch bundle canary
  WS-20-04  Context budget canary
  WS-21-04  DAG canary
  WS-22-01  M4 compatibility canary
```

Phase 1 tasks have no internal dependencies and should be started in parallel to reduce critical path length.

---

## 6. Definition of Done for Milestone 5

Milestone 5 is complete when:

- structured patch bundle schema exists with content-anchor addressing model and is validated
- Backend and Frontend execution support diff-first mode with full-file fallback, controlled by feature gate
- patch application service has unit coverage (including multi-op anchor shift) and canary coverage
- prompt scripts v2 for diff-first output exist and are registered
- context budget thresholds are externalized in `context_budget_policy.json`
- each Coding Team step emits context budget metadata
- release pack aggregates context budget reports
- workflow engine supports dependency-based dispatch without breaking sequential workflows
- concurrent execution error state matrix is implemented and tested
- BE / FE parallel readiness is explicit, gated, and tested
- all new canaries pass (patch, budget, DAG, M4 compatibility)
- full orchestrator test suite still passes
- complexity budget table is updated for all new M5 files
- success metrics from Section 1.1 are measurable at M5 close

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
| Patch application corrupts files | High | mitigate with content-anchor addressing + typed patch schema + deterministic application tests |
| Diff-first output becomes brittle on anchor mismatch | Medium | keep full-file fallback path + feature gate for global disable |
| LLM-generated anchors are unreliable | Medium | anchor validation before each operation; typed error with operation index on failure |
| Prompt size increase from injecting target files | Medium | cross-reference with context budget; pre-emptive full-file fallback on overflow_risk |
| Context budget metrics become advisory only | Medium | require report presence in release validation; thresholds in policy file |
| Parallel dispatch causes hidden ordering regressions | High | dependency gating + sequential fallback + error state matrix + canary coverage |
| M5 workstreams block each other, extending delivery | Medium | Phase 1 schema tasks are parallelizable; critical path explicitly documented (NEW) |
| M5 changes regress M4 behavior | Medium | M4 compatibility canary as explicit verification gate (NEW) |

---

## 9. Rollback Strategy (NEW)

| Scenario | Rollback Action | Impact |
|----------|----------------|--------|
| Diff-first execution unstable in production | Disable `execution.diff_first_enabled` feature gate | All steps revert to full-file mode; no code rollback needed |
| Context budget thresholds too aggressive | Edit `context_budget_policy.json` thresholds | Immediate effect; no code change |
| DAG parallel dispatch causes regressions | Remove `depends_on` from production workflow definitions | Workflow engine falls back to sequential; no code rollback needed |
| Entire M5 must be reverted | Revert to M4 tagged codebase | M4 compatibility canary ensures this is safe |

---

## 10. Next Step

This task list should be reviewed and approved before any M5 implementation begins.
