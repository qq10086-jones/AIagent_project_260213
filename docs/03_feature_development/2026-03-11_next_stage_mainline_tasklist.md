# Next-Stage Mainline Task List

- Date: 2026-03-11
- Status: DRAFT FOR EXECUTION
- Scope: post-M9 closeout mainline tasks aligned to Design v4.1 and Governance v3

---

## 1. Decision

Current next-stage work is classified as:

`MAINLINE WITH STRUCTURAL FOLLOW-UP`

Interpretation:

- the project should stay on the North Star pipeline
- no new feature branch or new subsystem expansion is authorized
- the next task group must focus on validation productization, runtime consistency, contract hardening, and structural debt reduction that directly protects the current pipeline

---

## 2. Task List

### P0

#### WS-NEXT-01: Validation Gate Productization

**Status**

`DONE` on 2026-03-11

**Current Progress**

- unified release-gate entrypoint added under `orchestrator/scripts/validate_next_stage_release_gate.js`
- config-only mode validated successfully via `validate:next_stage_release_gate -- --skip-live`
- full live release gate validated successfully via `validate:next_stage_release_gate`
- operator runbook added under `docs/03_feature_development/2026-03-11_validation_gate_runbook.md`
- consolidated summary artifact path established under `orchestrator/artifacts/validation/next_stage_release_gate/`

**Task Name**

`WS-NEXT-01 Validation Gate Productization`

**Pipeline Node**

`OpenClaw Orchestration -> Coding Team Workflow -> Artifacts`

**Task Type**

`Type A`

**Upstream Dependency**

- M9 live workflow validation already passes
- config preflight already exists and passes
- release-pack evidence contract already landed

**Goal**

Turn current one-off validation evidence into a repeatable release gate.

**Deliverables**

- one standard validation entrypoint covering:
  - config preflight
  - live runtime validation
  - live M9 workflow validation
- one consolidated machine-readable validation summary artifact
- one short operator runbook describing when to run each validation gate

**Non-Scope Declaration**

- no new workflow types
- no new provider expansion
- no routing policy expansion beyond current approved cohort and mode

**Acceptance Criteria**

- a single documented validation flow can be run before release review
- validation outputs are stored under stable artifact paths
- failures identify which gate failed without manual log archaeology
- PM and Architect can use the artifact set directly in go/no-go review

**LLM Role**

`none`

---

#### WS-NEXT-02: Runtime Consistency and Startup Path Hardening

**Status**

`DONE` on 2026-03-11

**Current Progress**

- local manual startup now resolves `WORKSPACE_ROOT` to repo root when unset
- `infra/docker-compose.yml` now mounts governance config from root `configs/`
- startup-path note added under `docs/03_feature_development/2026-03-11_runtime_startup_path_note.md`
- validation command `validate:runtime_boot_sources` added and passing

**Task Name**

`WS-NEXT-02 Runtime Consistency and Startup Path Hardening`

**Pipeline Node**

`Brain Router + Policy Layer -> OpenClaw Orchestration`

**Task Type**

`Type A`

**Upstream Dependency**

- M7 controlled enablement package exists
- startup config preflight exists

**Goal**

Eliminate drift between local/manual/container startup paths so runtime controls remain trustworthy.

**Deliverables**

- documented authoritative startup path for local and container runtime
- explicit config root and mount requirements list
- one validation check proving runtime mode, cohort config, and required mounts are loaded from intended locations

**Non-Scope Declaration**

- no new deployment platform
- no infra redesign
- no change to approved M7/M9 governance state without review

**Acceptance Criteria**

- startup path ambiguity is removed
- runtime control files used at boot can be verified deterministically
- container and local startup do not silently diverge on critical config
- rollback controls remain operator-simple

**LLM Role**

`none`

---

### P1

#### WS-NEXT-03: Brain Gateway Typed Contract Hardening

**Status**

`DONE` on 2026-03-11

**Current Progress**

- `brain` gateway handlers extracted into `orchestrator/src/vnext/brain_gateway.js`
- typed schema set added for latest-fact lookup and routing-decision ingest
- contract note added under `docs/03_feature_development/2026-03-11_brain_gateway_contract_note.md`
- integration test `orchestrator/test/brain_gateway.integration.test.js` added and passing

**Task Name**

`WS-NEXT-03 Brain Gateway Typed Contract Hardening`

**Pipeline Node**

`Brain Router + Policy Layer -> TaskEnvelope Normalization -> OpenClaw Orchestration`

**Task Type**

`Type A`

**Upstream Dependency**

- current HTTP fact lookup path is live
- M9 brain DB decoupling baseline is landed

**Goal**

Promote the current partial `brain -> orchestrator` HTTP boundary into a stable, typed, documented contract.

**Deliverables**

- endpoint inventory for current supported brain gateway surface
- request/response schema for each supported endpoint
- typed contract for:
  - latest fact lookup
  - routing decision ingest
  - future-approved callback/event path placeholders if required
- integration tests covering happy path and contract failure path
- short design note clarifying which side owns validation and persistence

**Non-Scope Declaration**

- no broad API expansion beyond current North Star need
- no direct DB access reintroduction in `brain`
- no gRPC migration unless HTTP contract proves insufficient

**Acceptance Criteria**

- `brain` no longer relies on undocumented orchestrator API behavior
- supported gateway surface is explicit and test-covered
- contract failures degrade clearly and do not create silent data loss
- the boundary is ready for future M7/M10 extension without schema guessing

**LLM Role**

`none`

---

### P2

#### WS-NEXT-04: Worker-Coder Structural Decomposition

**Status**

`DONE` on 2026-03-11

**Current Progress**

- extracted `prompt_contract.js`
- extracted `verification_runner.js`
- extracted `retry_policy.js`
- extracted `failure_memory.js`
- extracted `scope_guard.js`
- extracted `artifact_scaffold.js`
- extracted `scoped_delta.js`
- extracted `static_checks.js`
- added `task_lifecycle.js` to enforce single-finalization semantics for worker timeout/success/failure paths
- added `git_side_effects.js` to replace shell-based auto-commit with structured git execution and artifacted outcomes
- added targeted tests for each extracted slice
- added `startup_smoke.test.js` to catch worker entrypoint/import regressions before container startup
- added `artifact_scaffold.test.js` to cover scaffold creation and repair behavior
- added `scoped_delta.test.js` to cover scoped snapshot, diff summary, and deterministic fallback delta recovery
- added `static_checks.test.js` to cover fast static-check orchestration, severity shaping, and timeout clamping
- added `task_lifecycle.test.js` to cover timeout single-finalization and prevent duplicate result/fact/ack writes
- added `git_side_effects.test.js` to cover structured git argument handling and auto-commit artifact output
- fixed `worker-coder` startup import after decomposition and aligned live container execution with current source image
- hardened mocked `opencode` live-validation artifacts to satisfy current typed handoff / schema governance
- `worker-coder/coding_service.js` reduced materially to ~705 lines while `test:adapter` remains green
- `applyPatch` auto-commit side effects now emit structured diagnostics instead of relying on shell concatenation and console-only warnings

**Task Name**

`WS-NEXT-04 Worker-Coder Structural Decomposition`

**Pipeline Node**

`OpenClaw Orchestration -> Coding Team Workflow`

**Task Type**

`Type A (Structural Governance)`

**Upstream Dependency**

- current M9 behavior is functionally stable
- core retry / verification / evidence flow already passes targeted validation

**Goal**

Reduce regression risk in `worker-coder/coding_service.js` by separating responsibilities without changing behavior.

**Deliverables**

- decomposition plan for `coding_service.js`
- extracted modules for at least:
  - scope guard
  - verification runner
  - retry policy
  - failure memory
  - prompt contract
- updated targeted tests and canaries proving behavior parity

**Non-Scope Declaration**

- no semantic redesign of M9 behavior
- no new execution provider abstraction work
- no reasoning-layer expansion inside `worker-coder`

**Acceptance Criteria**

- `coding_service.js` is materially smaller and easier to audit
- extracted modules align with existing architectural responsibilities
- M9 canaries and targeted tests remain green
- decomposition does not change release-pack evidence semantics

**LLM Role**

`none`

---

## 3. Recommended Order

1. `WS-NEXT-01` Validation Gate Productization
2. `WS-NEXT-02` Runtime Consistency and Startup Path Hardening
3. `WS-NEXT-03` Brain Gateway Typed Contract Hardening
4. `WS-NEXT-04` Worker-Coder Structural Decomposition

Reasoning:

- the first priority is making current success repeatable and reviewable
- the second priority is protecting runtime governance from startup drift
- the third priority is hardening the `brain -> orchestrator` boundary that is already on the main path
- the fourth priority is structural debt reduction after runtime confidence is stabilized

---

## 4. Exit Criteria

This task group is complete when all of the following are true:

- release review can rely on a repeatable validation artifact set
- runtime startup path and config source are deterministic
- `brain -> orchestrator` boundary is typed, documented, and integration-tested
- `worker-coder` no longer concentrates critical M9 behavior in one oversized service file
- no task in this group expands scope beyond the current North Star pipeline

---

## 5. Summary

Current next-step posture:

- project priority: mainline stabilization, not feature expansion
- architecture priority: contract hardening and structural debt reduction
- PM priority: make validation and go/no-go review repeatable
- governance priority: keep work on Type A pipeline-supporting tasks only

Final recommendation:

`EXECUTE AS MAINLINE TASK GROUP`

---

## 6. Post-Closeout Execution Checklist

Status:

`UPDATED` on 2026-03-11 after full cohort recovery

Interpretation:

- `WS-NEXT-01` through `WS-NEXT-04` are complete for this round
- the temporary debug slice against truthful cohort failures is complete
- the next execution slice is evidence closeout plus next-cohort design, not repeated FE/BE failure archaeology

Scope discipline:

- no new feature expansion
- no new subsystem program
- no provider expansion
- no governance-state promotion without explicit artifact-backed review

### P0

#### GOV-01: Close Out Mainline Evidence State

**Status**

`READY`

**Goal**

Align progress notes, governance notes, and tasklists with the recovered authoritative result:

- full four-case worker-coding cohort = `4 pass / 0 fail / 0 partial`

**Expected Output**

- authoritative current-state references point to the latest passing cohort artifact
- stale debug guidance is removed from active tasklists
- next-step decisions are reviewable without replaying old incident context

#### REG-01: Lock Regression Guards For Recovered Failure Modes

**Status**

`READY`

**Goal**

Prevent the recovered structural failures from reappearing silently.

**Checks**

- single-file target paths must not trigger out-of-scope fallback stub writes
- worker-coding cohort result logic must treat verification supersets as pass
- coding-executor request building must continue to preserve `verification_plan` and task-contract metadata

#### ARCH-ISO-01: Land Execution Isolation Design

**Status**

`DONE`

**Goal**

Define the approved architecture for worker-coding failure-safe execution and promotion.

**Current artifact**

- design note: `docs/03_feature_development/2026-03-11_worker_coding_execution_isolation_design.md`

**Decision**

- isolation model = isolated workspace execution plus validated patch promotion
- default path is not `git worktree + auto-commit`
- no implementation writes reach main workspace before verification and promotion preflight succeed

### P1

#### ARCH-ISO-02: Implement Isolation Scaffold Behind Flag

**Status**

`DONE (phase 1-2 landed; scaffold + shadow execution)`

**Goal**

Create isolated task workspace materialization and artifact manifests without yet changing the public workflow contract.

**Current implementation**

- `worker-coder/isolation_workspace.js`
- `worker-coder/coding_service.js`
- `worker-coder/tests/isolation_workspace.test.js`
- feature flag: `CODER_ISOLATION_MODE=scaffold|shadow`

#### ARCH-ISO-03: Add Promotion Preflight And Rollback Evidence

**Status**

`DONE (preflight + explicit promote mode landed)`

**Goal**

Promote only verified scoped patches into main workspace and make rollback posture explicit in artifacts.

**Current state**

- shadow-mode isolated execution is landed for delegate/static-check/verification
- main workspace remains untouched before promotion
- promotion preflight and explicit `promote` mode are landed
- remaining gap is richer rollback evidence and clean live evidence without runner-side terminal-state inference

#### ARCH-ISO-04: Validate Live Cohort Under Shadow Mode

**Status**

`DONE (full shadow cohort passed)`

**Goal**

Confirm that worker-coding isolation preserves the recovered cohort baseline before any broader `promote` activation.

**Acceptance target**

- live cohort runs with `CODER_ISOLATION_MODE=shadow`
- recovered passing baseline does not regress unexpectedly
- artifacts clearly show isolation mode and zero main-workspace promotion

**Current evidence**

- live debug cohort workflows produced step-level diagnostics with:
  - `isolation_mode=shadow`
  - `promotion.applied=false`
  - normal release artifact roots preserved
- debug FE+BE cohort artifact:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T14-02-24-449Z/worker_coding_cohort_result.json`
  - result: `2 pass / 0 fail / 0 partial`
- full four-case shadow cohort artifact:
  - `orchestrator/artifacts/validation/worker_coding_cohort/worker_coding_cohort_2026-03-11T14-14-49-306Z/worker_coding_cohort_result.json`
  - result: `4 pass / 0 fail / 0 partial`
- workflow-engine closeout hardening landed in `orchestrator/src/workflow_engine.js`
- targeted regression coverage landed in `orchestrator/test/workflow_finalization.test.js`

**Remaining gap**

- rerun live full four-case shadow cohort and confirm the new result-consumer pending-recovery path prevents late-consumption `TASK_QUEUED_STALE` failures
- confirm runner-side terminal-state inference is no longer exercised in the fresh live rerun
- enrich rollback evidence so promotion failure and restore posture are operator-visible
- only evaluate broader `promote` rollout after those evidence items are closed

#### WC-NEXT-A: Define Worker-Coding v2 Cohort

**Status**

`DRAFTED`

**Goal**

Move beyond the recovered baseline four-case cohort into a harder but still governed validation set.

**Required shape**

- at least one multi-file FE modify case
- at least one BE case with tighter regression surface
- at least one FE/BE contract-linked case
- at least one higher-friction bug-fix case

**Acceptance target**

- difficulty increases without abandoning deterministic reviewability
- task-class balance remains governed
- no scenario widens into unconstrained autonomy

**Current artifacts**

- design note: `docs/03_feature_development/2026-03-11_worker_coding_cohort_task_matrix_v2.md`
- machine-readable plan: `configs/registry/worker_coding_cohort_plan_v2.json`
- remaining action is review plus execution readiness, not blank-page design

#### WC-NEXT-B: Decide M9 Governance Exit Recommendation

**Status**

`READY`

**Goal**

Use the recovered evidence state to recommend one of:

- keep `GO WITH CONDITIONS`
- promote to closeout-ready with explicit beta limitations
- defer promotion pending v2 cohort evidence

**Inputs**

- release gate artifacts
- runtime boot-source validation
- full four-case passing cohort
- residual caution around deterministic-provider-backed evidence scope

### Exit Rule For This Slice

Do not widen scope beyond the current North Star coding pipeline until:

- recovered failure modes are regression-protected
- v2 cohort definition is reviewed
- M9 governance-state recommendation is written against current evidence
