# FE-safe Completion Contract

- Version: 1.0
- Date: 2026-03-09
- Milestone: M6
- Status: APPROVED
- Supersedes: none (first issue)

---

## 1. Purpose

This contract defines the precise conditions under which a workflow may run BE and FE steps in parallel (the "FE-safe" execution path), and what constitutes valid completion for each branch, the combined workflow, and the QA admission gate.

This contract is a prerequisite for WS-24.5 runtime wiring. No production parallel dispatch may be opened without an approved version of this document.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **FE-safe workflow** | A workflow class where FE implementation does not require any artifact, schema, or API contract produced by the BE step of the same workflow run |
| **BE branch** | The backend implementation step and its output artifacts |
| **FE branch** | The frontend implementation step and its output artifacts |
| **Branch completion** | The state where a branch has produced all required contract-valid artifacts and no step within it is in a failed, timeout, or ambiguous state |
| **Merge-ready state** | Both branches have reached branch completion and their artifacts are structurally compatible for release pack assembly |
| **Partial-output state** | One branch has reached branch completion; the other has not |
| **be_to_fe handoff** | The artifact produced by the BE step that carries shared types, API contracts, or data schemas required for FE implementation |

---

## 3. FE-safe Eligibility Criteria

A workflow run may enter the FE-safe parallel execution path only when ALL of the following are true:

1. **No be_to_fe handoff dependency**: The FE implementation for this workflow class is explicitly documented as not requiring any content from the `be_to_fe` handoff artifact of the same run. This must be declared in the exposure eligibility policy (`parallel_exposure_policy.json`), not inferred at runtime.

2. **Stable upstream API contract**: All API endpoints, data schemas, and shared types that FE will consume must already exist and be stable prior to this workflow run. The run must not introduce new BE contracts that FE needs.

3. **Independent artifact namespace**: BE and FE output artifacts write to non-overlapping file paths. There must be no file-level conflict possible between the two branches.

4. **No cross-branch runtime dependency**: Neither branch may block on a signal, lock, or output from the other branch during execution.

5. **Policy approval**: The workflow type and project type combination must appear on the approved whitelist in `parallel_exposure_policy.json`.

If any criterion is not met, the workflow must remain on the sequential path. This determination must be made at dispatch time by the runtime structural guard (see Section 7).

---

## 4. Branch Completion Conditions

### 4.1 BE Branch Completion

The BE branch is considered complete when:

- `impl/be_changes/` directory contains at least one file
- `coding_team_impl_to_qa_handoff` artifact is present and schema-valid
- No step within the BE branch is in state: `failed`, `timeout`, `error`, `quarantined`
- Context budget report for the BE step is present (overflow is permitted but must be flagged)

### 4.2 FE Branch Completion

The FE branch is considered complete when:

- `impl/fe_changes/` directory contains at least one file
- FE handoff artifact is present and schema-valid
- No step within the FE branch is in state: `failed`, `timeout`, `error`, `quarantined`
- Context budget report for the FE step is present

### 4.3 Independent Completion

In a FE-safe parallel run, BE branch completion and FE branch completion are evaluated independently. Neither branch's completion is blocked by the other branch's state.

---

## 5. QA Admission Conditions

QA may start after parallel execution only when ALL of the following are true:

1. **Both branches are complete**: BE branch completion (Section 4.1) and FE branch completion (Section 4.2) are both satisfied
2. **No branch is in a terminal failure state**: Neither branch may be in `failed`, `timeout`, `quarantined`, or `partial_failure`
3. **Artifacts are in merge-ready state**: BE and FE artifact namespaces are non-overlapping and both handoff artifacts are schema-valid
4. **Merge order is resolved**: The artifact merge order defined in Section 6 has been applied and confirmed before QA receives the combined artifact set

Partial branch success — where one branch is complete and the other is not — must NOT unlock QA. The QA gate is binary: both branches complete, or QA does not start.

---

## 6. Artifact Merge Order

After both branches reach completion, artifacts must be assembled in this order before QA admission:

1. BE artifacts from `impl/be_changes/` are written to the release staging area first
2. FE artifacts from `impl/fe_changes/` are written second
3. If any file path appears in both sets, the workflow is immediately moved to `partial_failure` — this condition must not be resolved automatically; it requires operator intervention
4. Both context budget reports are aggregated into a combined budget summary
5. Both handoff artifacts are included in the combined QA input package
6. The combined package is validated against the `coding_team_impl_to_qa_handoff` schema before QA step begins

---

## 7. Partial-Output State Definition

A partial-output state exists when:

- One branch has reached branch completion
- The other branch is still executing, has failed, or has timed out

**Behavior in partial-output state:**

| Scenario | Behavior |
|----------|----------|
| One branch complete, other still running | Wait for the other branch; do not advance to QA |
| One branch complete, other failed | Move workflow to `partial_failure`; do not advance to QA |
| One branch complete, other timed out | Move workflow to `partial_failure`; do not advance to QA |
| Both branches failed | Move workflow to `failed` |

`partial_failure` is a terminal state for the current run. It is not release-eligible. Retry must target only the failed branch and must be explicitly authorized.

---

## 8. What "Completion" Means for the Workflow

The workflow as a whole reaches `merge_ready` state when:

- Both branches satisfy their branch completion conditions (Section 4)
- Artifact merge order has been applied without conflict (Section 6)
- The combined artifact package has passed schema validation
- The workflow state machine records `effective_execution_path: gated_parallel`

Only a workflow in `merge_ready` state may proceed to QA.

---

## 9. Structural Impossibility Rule

If a workflow class is declared FE-safe in the eligibility policy but the runtime determines that FE completion structurally requires a `be_to_fe` handoff artifact that is not available under the parallel path, the runtime must:

1. Deny parallel dispatch with denial reason `structural_completion_impossible`
2. Force the workflow back to the sequential path
3. Log the structural denial with the workflow ID, class, and detected dependency

This rule takes precedence over any policy configuration. It cannot be overridden by policy alone.

---

## 10. Versioning and Amendment

This contract may only be amended through a documented governance review. Amendments must:

- Increment the version number
- Record the change rationale
- Be approved by the Architect before taking effect in any staging or production environment

---

## 11. References

- Design addendum: `docs/01_design/system/260308/260308_2330/OpenClaw_Nexus_Design_Document_v3.2.md` Sections 4.1–4.4
- Task list: WS-24-01, WS-24.5-02
- Related contracts: `docs/contracts/parallel_failure_handling_contract.md` (WS-24-02)
- Runtime implementation: WS-24.5-01, WS-24.5-02, WS-24.5-03
