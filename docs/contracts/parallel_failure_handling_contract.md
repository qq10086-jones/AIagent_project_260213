# Parallel Failure-Handling Contract

- Version: 1.0
- Date: 2026-03-09
- Milestone: M6
- Status: APPROVED
- Supersedes: none (first issue)

---

## 1. Purpose

This contract defines the user-visible and system-internal behavior for every failure mode that can occur during FE-safe parallel execution of `coding_team_v0`. It supplements the FE-safe Completion Contract and is a prerequisite for WS-24.5 runtime wiring.

---

## 2. Failure Mode Catalogue

### 2.1 BE Success + FE Failure

| Property | Value |
|----------|-------|
| Workflow state | `partial_failure` |
| Release eligible | NO |
| QA eligible | NO |
| BE artifacts | Quarantined — retained for 90 days, not merged into release |
| FE artifacts | Discarded |
| User-visible message | "The frontend implementation step failed. Backend work has been preserved but not released. The workflow must be retried." |
| Retry policy | FE branch only (see Section 4) |
| Rollback trigger | No — `partial_failure` is not a rollback trigger unless the failure rate exceeds the circuit-breaker threshold |

### 2.2 BE Failure + FE Success

| Property | Value |
|----------|-------|
| Workflow state | `partial_failure` |
| Release eligible | NO |
| QA eligible | NO |
| BE artifacts | Discarded |
| FE artifacts | Quarantined — retained for 90 days, not merged into release |
| User-visible message | "The backend implementation step failed. Frontend work has been preserved but not released. The workflow must be retried." |
| Retry policy | BE branch only (see Section 4) |
| Rollback trigger | No — same threshold condition as 2.1 |

### 2.3 Branch Timeout

| Property | Value |
|----------|-------|
| Workflow state | `partial_failure` (if one branch completes) or `failed` (if both timeout) |
| Release eligible | NO |
| QA eligible | NO |
| Timed-out branch artifacts | Discarded |
| Completed branch artifacts (if any) | Quarantined |
| User-visible message | "One or more implementation steps exceeded the time limit. The workflow has been stopped and must be retried." |
| Retry policy | Full workflow retry required (branch-only retry is not safe after timeout) |
| Rollback trigger | If timeout rate exceeds circuit-breaker threshold within rolling window |

### 2.4 Patch Failure After Diff-First Attempt

| Property | Value |
|----------|-------|
| Workflow state | Remains in the affected branch's step state; does not immediately become `partial_failure` |
| Fallback behavior | Automatic fallback to full-file output mode for the affected branch |
| Release eligible | Depends on whether fallback succeeds |
| Logged event | `diff_first_fallback` with anchor mismatch details |
| User-visible message | None for fallback (transparent); if fallback also fails: "The implementation could not be applied to the current codebase. Manual review is required." |
| Retry policy | If fallback fails, treat as branch failure; apply Section 2.1 or 2.2 accordingly |

### 2.5 Rollback-Triggering Incident

A rollback-triggering incident is defined as any of the following:

- Automated circuit-breaker activates (rolling window threshold breached)
- Manual operator rollback initiated
- `partial_failure` rate within a rolling window of 100 runs exceeds 25%
- Any uncontrolled scope expansion event (parallel path activated outside approved whitelist)

| Property | Value |
|----------|-------|
| Immediate action | Force-sequential mode activated for all subsequent runs |
| In-flight runs | Allowed to complete on their current execution path; not interrupted |
| Subsequent runs | Sequential path only until operator explicitly resets |
| Logged event | `rollback_trigger` with trigger type, timestamp, metric, and threshold |
| User-visible message | None directly; operator receives alert |
| Recovery | Manual reset by Architect-level operator only |

---

## 3. Partial Artifact Handling

Partial artifacts — those produced by a branch that did not result in a valid `merge_ready` workflow — must never be silently merged into release output.

| Artifact State | Handling |
|----------------|----------|
| From a completed branch in `partial_failure` | Quarantined: stored under `artifacts/quarantine/<workflow_id>/` for 90 days |
| From a failed branch | Discarded immediately |
| From a timed-out branch | Discarded immediately |
| Quarantined artifacts | May be inspected by Architect-level operators; may not be used as release input without a full workflow re-run |

Quarantine is not an intermediate state toward release. A quarantined artifact can only be used as a reference for debugging. Releasing from quarantine is explicitly prohibited.

---

## 4. Branch-Specific Retry Policy

### 4.1 When Branch-Only Retry Is Permitted

Branch-only retry (retrying only the failed branch without re-running the successful branch) is permitted when:

- The failure mode is 2.1 (BE success + FE failure) or 2.2 (BE failure + FE success)
- The timeout failure mode (2.3) has NOT occurred — timeout requires full workflow retry
- The successful branch's artifacts are still in quarantine and have not been modified or expired
- Retry is explicitly authorized by an Architect-level operator

### 4.2 Branch-Only Retry Procedure

1. Operator confirms the quarantined artifacts from the successful branch are intact
2. The failed branch is re-executed with the same input as the original run
3. If the re-executed branch succeeds, the two artifact sets are merged following the merge order in the FE-safe Completion Contract Section 6
4. If the re-executed branch fails again, the workflow moves to `failed` (no further automatic retry)

### 4.3 Maximum Retry Attempts

- Branch-only retry: maximum 1 attempt
- If the branch-only retry fails, the operator must initiate a full workflow re-run
- Full workflow re-runs reset all retry counters

### 4.4 What Is Not a Valid Retry

- Re-running only QA or release steps without re-running the failed implementation branch
- Manually patching quarantined artifacts and then releasing them
- Combining artifacts from two different workflow run IDs without explicit Architect approval

---

## 5. User-Visible Failure Communication

All user-facing failure messages must:

- State which step failed (BE or FE or both), without exposing internal stack traces
- State whether the workflow is retryable and by what mechanism
- Not expose system internals, file paths, or infrastructure names

| Failure Mode | Message Template |
|--------------|-----------------|
| BE success + FE failure | "The frontend implementation step failed. Backend work has been preserved but not released. The workflow must be retried." |
| BE failure + FE success | "The backend implementation step failed. Frontend work has been preserved but not released. The workflow must be retried." |
| Both failed | "Both implementation steps failed. The workflow must be restarted." |
| Branch timeout | "One or more implementation steps exceeded the time limit. The workflow has been stopped and must be retried." |
| Patch failure (no fallback) | "The implementation could not be applied to the current codebase. Manual review is required." |
| Rollback active | Not surfaced to users; operator-only |

---

## 6. Observability Requirements

Every failure event must emit a structured log entry containing:

| Field | Required |
|-------|----------|
| `workflow_id` | YES |
| `run_id` | YES |
| `failure_mode` | YES — one of: `be_failed`, `fe_failed`, `both_failed`, `branch_timeout`, `patch_failure`, `rollback_trigger` |
| `failed_branch` | YES where applicable |
| `workflow_state` | YES — final state after failure handling |
| `artifact_disposition` | YES — one of: `quarantined`, `discarded`, `intact` |
| `retry_eligible` | YES — boolean |
| `timestamp` | YES |

These log entries feed the circuit-breaker rolling window evaluation (WS-25-05).

---

## 7. Testability Requirements

Each failure mode in Section 2 must be covered by at least one integration test that:

- Simulates the failure condition
- Asserts the correct workflow state transition
- Asserts the correct artifact disposition (quarantined vs discarded)
- Asserts the correct user-visible message is emitted
- Asserts the correct structured log entry is written

Tests are implemented in WS-24-04.

---

## 8. References

- FE-safe Completion Contract: `docs/contracts/fe_safe_completion_contract.md`
- Design addendum: `docs/01_design/system/260308/260308_2330/OpenClaw_Nexus_Design_Document_v3.2.md` Sections 4.3, 4.4, 8
- Task list: WS-24-02, WS-24-04
- Circuit-breaker: WS-25-05
- Runtime implementation: WS-24.5-01, WS-24.5-02, WS-24.5-03
