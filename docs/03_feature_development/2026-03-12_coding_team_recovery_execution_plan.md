# Coding Team Recovery Execution Plan

- Date: 2026-03-12
- Status: READY FOR EXECUTION
- Classification: MAINLINE TASK
- Subtype: Mainline stabilization with runtime recovery follow-up
- Scope: recover a stable coding execution lane while isolating `Qwen on opencode` as an independent provider triage track

---

## 1. Task Classification

This work is classified as:

`MAINLINE TASK`

It is **not** a patch-only task, and it is **not** a side-branch exploration task.

Reason:

- it directly protects the current North Star coding pipeline
- it does not introduce a new product surface or new subsystem program
- it is required to restore authoritative validation and release-gate confidence
- it separates runtime/provider instability from workflow-quality validation without widening governance scope

Interpretation:

- mainline objective: restore a trustworthy execution lane for worker-coding validation
- bounded follow-up: isolate `Qwen on opencode` as a provider integration issue, not as the blocker for the entire coding pipeline
- non-scope: no cohort widening, no autonomy expansion, no new provider platform rollout

---

## 2. Decision

The recovery strategy is to split the current problem into two independent lanes:

1. `Stable Execution Lane`
   - restore an authoritative, repeatable coding-validation path using a known-good stable model/runtime lane
2. `Provider Triage Lane`
   - investigate `Qwen on opencode` as a separate runtime/provider compatibility problem

This separation is required because current evidence shows:

- direct DashScope compatible-mode calls succeed
- `opencode` built-in Alibaba Coding Plan provider authentication still fails
- M10 load validation is blocked by workflow closure/evidence completeness, not by lack of dispatch acceptance

---

## 3. Goals

### P0 Goal

Restore one stable, explicit, auditable coding execution lane that can be used for:

- minimal real-verification cohort execution
- authoritative full-slice revalidation
- release-gate confidence rebuilding

### P1 Goal

Produce an unambiguous triage conclusion for `Qwen on opencode`:

- `FIXED`
- `PROVIDER_BLOCKED`
- `CREDENTIAL_SCOPE_MISMATCH`
- `CONFIG_RESOLUTION_ERROR`

### P2 Goal

Improve observability so provider/runtime failures are no longer misread as coding-quality or verification-quality failures.

---

## 4. Execution Rules

The following rules are mandatory for this task:

- no silent fallback
- every run must record its actual `execution_lane`
- fallback is allowed only if explicitly configured and explicitly artifacted
- provider failure must be typed
- stable-lane validation and `Qwen on opencode` triage must not share the same evidence pool
- no cohort widening beyond current governed scope during this recovery slice

---

## 5. Workstreams

## WS-REC-01 Stable Execution Lane

**Priority**

`P0`

**Status**

`READY`

**Goal**

Introduce one explicit stable coding lane, initially:

- `stable_local_lane = opencode + local ollama/glm-4.7-flash:latest`

This lane exists to recover trustworthy execution and validation continuity while the provider issue is investigated separately.

Runtime note:

- this lane should be wired through `opencode` against a local Ollama model first, so provider-chain triage can compare `opencode + local ollama` versus `opencode + Qwen` versus direct compatible-mode calls.

**Required Deliverables**

- config entry for `stable_local_lane`
- worker-coder default lane selection support
- artifact fields for:
  - `execution_lane`
  - `model_provider`
  - `model_name`
  - `fallback_taken`
  - `fallback_target`
- explicit operator-visible reason when fallback occurs

**Acceptance Criteria**

- worker-coder can be configured to run on the stable lane without modifying task payload semantics
- release-pack and cohort artifacts show the actual lane used
- no run is reported as if it used `opencode/Qwen` when it actually used the stable lane

**Non-Scope**

- no auto-fallback hidden behind provider exceptions
- no broad model policy redesign

---

## WS-REC-02 Typed Provider Adapter

**Priority**

`P0`

**Status**

`READY`

**Goal**

Decouple invocation paths just enough to make failures typed, comparable, and auditable.

Initial provider classes:

- `local-compatible`
- `dashscope-compatible`
- `opencode-provider`

**Required Deliverables**

- minimal invocation adapter contract
- typed provider result object
- typed provider error classes

Suggested minimum typed failure set:

- `AUTH_FAILURE`
- `PROVIDER_CONFIG_ERROR`
- `PROVIDER_UNAVAILABLE`
- `MODEL_NOT_FOUND`
- `REQUEST_SHAPE_ERROR`
- `EXECUTION_TIMEOUT`

**Acceptance Criteria**

- the same failure is not reported differently across providers without evidence
- provider failures are distinguishable from:
  - `coding_failure`
  - `verification_failure`
  - `workflow_finalization_failure`

**Architecture Guidance**

- keep the adapter narrow
- do not attempt a large generic provider framework in this slice
- optimize for diagnostic clarity, not abstraction completeness

---

## WS-REC-03 Minimal Coding MVP

**Priority**

`P1`

**Status**

`READY`

**Goal**

Restore a minimal real-verification coding cohort on the stable lane before re-running larger authoritative slices.

**Approved Initial Cohort**

- `fe_modify`
- `bug_fix`

**Required Deliverables**

- 2-case cohort run on stable lane
- real verification enabled
- machine-readable cohort artifact
- one short summary note

**Acceptance Criteria**

- both cases complete with truthful verification results
- artifacts clearly show stable-lane execution
- no provider ambiguity remains in the artifact trail

---

## WS-REC-04 Capability Boundary For Stable Lane

**Priority**

`P1`

**Status**

`READY`

**Goal**

Define what `glm4.7-flash` is allowed to handle so stable-lane success is not polluted by out-of-bound task shapes.

**Required Boundary v1**

Allowed:

- single-file or low-coupling FE modify
- scoped bug fix with explicit regression boundary
- existing component/page adjustment

Rejected or downgraded:

- FE/BE contract-linked change
- cross-module or multi-directory restructuring
- large scaffold generation
- high-verification-load tasks

**Acceptance Criteria**

- over-complex tasks are rejected or downgraded before execution
- stable-lane evidence reflects intended capability, not accidental overreach

---

## WS-REC-05 OpenCode Qwen Triage

**Priority**

`P1`

**Status**

`READY`

**Goal**

Produce an authoritative triage conclusion for `Qwen on opencode` without blocking stable-lane recovery.

**Required Deliverables**

- minimal auth/config matrix
- verified env var inventory
- request-shape comparison:
  - direct compatible-mode call
  - `opencode provider` call
- final triage conclusion

**Expected Output States**

- `FIXED`
- `PROVIDER_BLOCKED`
- `CREDENTIAL_SCOPE_MISMATCH`
- `CONFIG_RESOLUTION_ERROR`

**Acceptance Criteria**

- the failure location is narrowed to one concrete layer
- conclusion is evidence-backed and machine-reviewable

---

## WS-REC-06 Authoritative Revalidation

**Priority**

`P1`

**Status**

`READY AFTER WS-REC-01/03/04`

**Goal**

Rebuild one authoritative worker-coding evidence slice on the stable lane.

**Mandatory Snapshot Inputs**

- commit SHA
- config snapshot
- cohort plan version
- execution lane
- provider/model
- verification contract version

**Required Deliverables**

- uninterrupted full-slice cohort rerun
- machine-readable result artifact
- one-page authoritative summary

**Acceptance Criteria**

- run is reproducible from the recorded snapshot
- latest status docs can point to one authoritative artifact without ambiguity

---

## WS-REC-07 Observability Upgrade

**Priority**

`P2`

**Status**

`READY`

**Goal**

Make runtime/provider path decisions and failures queryable without log archaeology.

**Required Fields**

- `provider_error_class`
- `fallback_taken`
- `fallback_target`
- `execution_lane`
- `model_provider`
- `model_name`

**Acceptance Criteria**

- provider failure is distinguishable from coding failure and verification failure in artifacts alone

---

## 6. Recommended Order

Execute in this order:

1. `WS-REC-01` Stable Execution Lane
2. `WS-REC-02` Typed Provider Adapter
3. `WS-REC-04` Capability Boundary For Stable Lane
4. `WS-REC-03` Minimal Coding MVP
5. `WS-REC-06` Authoritative Revalidation
6. `WS-REC-05` OpenCode Qwen Triage
7. `WS-REC-07` Observability Upgrade

Reasoning:

- first restore one explicit, trustworthy execution path
- then make provider/runtime failures diagnosable
- then constrain the stable lane to safe task shapes
- only after that should authoritative revalidation be treated as current truth
- `Qwen on opencode` remains important, but should no longer block the whole coding pipeline

---

## 7. Exit Criteria

This recovery slice is complete only when all of the following are true:

1. one stable coding lane runs with explicit artifacted identity
2. no fallback is silent or ambiguous
3. minimal stable-lane cohort passes with real verification
4. one authoritative full-slice cohort artifact is published against a frozen snapshot
5. `Qwen on opencode` receives one explicit triage conclusion
6. provider failure, coding failure, and verification failure are distinguishable in machine-readable artifacts

---

## 8. Final Recommendation

Recommended task label for governance and execution boards:

`MAINLINE TASK - CODING RUNTIME RECOVERY`

Short explanation:

- this is mainline because it restores confidence in the current production-bound coding pipeline
- it is not a patch-only task because it includes validation authority, runtime lane control, and observability hardening
- it is not a side-branch task because the outputs directly govern current release-gate and worker-coding evidence

