# Worker-Coding Execution Isolation and Rollback Design

- Date: 2026-03-11
- Scope: `WC-NEXT-02`
- Status: in progress (`phase 1-3 landed; phase 4 pending; orchestrator finalization fail-close landed`)
- Owners: Architecture / QA / Engineering

---

## 1. Purpose

This note defines how worker-coding should execute without polluting the main workspace when a task fails, times out, or produces out-of-scope writes.

The goal is not to redesign the coding workflow. The goal is to add a safer execution envelope around the existing `worker-coder` path.

---

## 2. Current State

Current behavior is:

- `worker-coder` executes directly against the mounted repo workspace
- write scope is limited by `target_paths` and scope-guard checks
- verification and static checks run after code is written
- failure evidence is persisted, but file pollution can still happen before failure is known

This means current guardrails are useful but incomplete:

- they constrain *where* writes may happen
- they do not guarantee failure leaves the main workspace untouched

Observed risk classes:

1. failed run leaves partial edits inside allowed paths
2. timed-out run leaves an uncertain workspace state
3. retry loop compounds local modifications before terminal failure
4. historical fallback behavior proved that even structurally bounded systems can still emit misleading file changes

---

## 3. Design Requirement

Execution isolation for worker-coding must satisfy all of the following:

1. failure or timeout must not mutate the main workspace
2. successful code must still be reviewable as a scoped patch bundle
3. user dirty worktree state must not be overwritten
4. rollback must be operator-simple
5. the design must work without requiring a clean Git branch or auto-commit-based promotion

---

## 4. Recommended Architecture

Recommended architecture:

`isolated workspace execution + validated patch promotion`

Flow:

1. capture a scoped baseline from the main workspace
2. materialize an isolated task workspace outside the main repo write path
3. run delegate, static checks, and verification inside the isolated workspace
4. if the run fails, discard the isolated workspace and keep only artifacts
5. if the run succeeds, generate a scoped patch bundle from isolated workspace vs baseline
6. validate the patch bundle against `target_paths`
7. apply the patch bundle to the main workspace only after preflight succeeds

This creates a clean separation:

- isolated workspace = execution surface
- main workspace = promotion target

---

## 5. Why This Architecture

### Chosen

`isolated workspace + patch promotion`

Reasons:

- compatible with the current dirty-worktree reality
- does not require branch switching or Git worktree hygiene from the user
- allows failure cleanup by deleting one isolated directory
- keeps patch review and artifact evidence as first-class outputs

### Rejected as primary path

`git worktree + auto-commit per task`

Reasons:

- current project explicitly tolerates unrelated local changes
- auto-commit promotion is too coupled to Git cleanliness and branch semantics
- conflict handling becomes harder to explain to operators
- rollback is less obvious when multiple task commits exist beside user edits

### Rejected as primary path

`direct main-workspace edit + best-effort rollback`

Reasons:

- rollback after partial writes is weaker than no-write-before-success
- failure evidence becomes mixed with branch state
- timeout handling remains ambiguous

---

## 6. Execution Model

### 6.1 Isolation Root

Per task, create:

`artifacts/runs/<run_id>/task_<task_id>/isolated_workspace/`

or an equivalent temp directory referenced from task artifacts.

The isolated workspace should contain only what the task needs:

- files under allowed `target_paths`
- minimal repo verification context needed for `lint`, `type_check`, `unit_test`, `build`
- artifact output root for generated reports

### 6.2 Main Rule

No implementation file under the main workspace may be modified before:

1. delegate succeeds
2. static checks succeed
3. verification succeeds
4. scoped patch promotion preflight succeeds

This is the partial-write prohibition rule.

### 6.3 Promotion Rule

Promotion to the main workspace must be:

- scope-validated
- preflight-checked before apply
- recorded as a dedicated promotion step in artifacts

If promotion preflight fails, do not touch the main workspace.

---

## 7. Rollback Semantics

### Failure before promotion

Rollback action:

- delete or mark disposable the isolated workspace
- preserve logs, diff summary, verification records, and failure summary

Effect on main workspace:

- none

### Failure during promotion preflight

Rollback action:

- no patch apply
- preserve promotion failure artifact

Effect on main workspace:

- none

### Failure after promotion begins

This state should be designed to be rare.

Mitigation:

- use patch preflight before apply
- prefer one-shot scoped patch apply over incremental file copying
- if any apply path is non-atomic, require a reversible backup manifest for touched files

Target posture:

- promotion should behave as all-or-nothing from the operator point of view

---

## 8. Required Artifacts

Each isolated execution should emit:

- `isolation/execution_mode.json`
- `isolation/baseline_manifest.json`
- `isolation/isolated_workspace_manifest.json`
- `isolation/promotion_preflight.json`
- `isolation/promotion_result.json`

Recommended minimum fields:

- execution mode
- isolation root path
- target paths
- baseline source
- verification status before promotion
- promotion attempted / applied / blocked
- rollback required / not required

---

## 9. Implementation Phases

### Phase 1: Isolation Scaffold

- add isolated workspace creation utilities
- copy or materialize scoped input files
- persist isolation manifests
- keep current main-workspace execution path behind a feature flag fallback

### Phase 2: Run Delegate Inside Isolation

- execute adapter, static checks, and verification against isolated workspace
- ensure artifact outputs still land in normal run/task artifact roots

### Phase 3: Promotion Gate

- generate scoped patch bundle from isolated workspace
- validate patch bundle against `target_paths`
- add promotion preflight and apply step

### Phase 4: Rollback and Operator Evidence

- write promotion and rollback artifacts
- surface whether a failed task touched only isolation or reached promotion

Current implementation state:

- phase 1 landed:
  - isolated workspace manifests and scoped shadow copies
- phase 2 landed:
  - delegate, static checks, and verification can run from isolated workspace in `shadow` mode
- phase 3 landed:
  - promotion preflight and explicit `promote` mode exist
- phase 4 remains open:
  - richer rollback evidence
  - live evidence cleanup to remove dependence on runner-side terminal-state inference

---

## 10. Acceptance Criteria

`WC-NEXT-02` should be considered implemented when:

1. a failed isolated coding task leaves the main workspace unchanged
2. a timed-out isolated coding task leaves the main workspace unchanged
3. a successful task promotes only files within allowed `target_paths`
4. promotion failure does not partially mutate the main workspace
5. artifacts clearly show isolation mode, promotion outcome, and rollback posture
6. a live cohort can be run under the approved isolation mode without regressing the recovered passing baseline

---

## 11. Immediate Next Action

Recommended next engineering slice:

1. enrich promotion result artifacts with clearer rollback / restore evidence
2. rerun a controlled live cohort in `shadow` mode and verify runner-side terminal-state inference is no longer needed
3. remove or downgrade the temporary runner inference path once clean live evidence is captured
4. only then run the harder v2 cohort under the approved isolation mode

Latest validation note:

- controlled live `shadow`-mode cohort evidence now exists at workflow-step level:
  - step diagnostics show `isolation_mode=shadow`
  - `promotion.applied=false`
  - artifact output still lands under normal release artifact roots
- orchestrator workflow finalization now fails closed if artifact-pack generation throws:
  - `workflow_run` is marked `failed`
  - parent `run` is marked `failed`
  - `workflow.finalization.failed` is emitted
- targeted regression coverage now exists:
  - `orchestrator/test/workflow_finalization.test.js`
- debug FE+BE shadow cohort artifact:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T14-02-24-449Z/worker_coding_cohort_result.json`
  - result: `2 pass / 0 fail / 0 partial`
- full four-case shadow cohort artifact:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T14-14-49-306Z/worker_coding_cohort_result.json`
  - result: `4 pass / 0 fail / 0 partial`
- current remaining issue is no longer silent orchestrator finalization drift; it is evidence cleanup and rollback artifact completeness before broader `promote` rollout
