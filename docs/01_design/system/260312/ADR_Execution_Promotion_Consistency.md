# Architecture Decision Record: Execution Promotion Consistency (M10 Phase 1)

**Date:** 2026-03-12
**Status:** Approved
**Context:** OpenClaw Nexus Worker-Coder execution is moving from `shadow` (isolated testing) to `promote` (merging verified isolated code back into the main workspace). Because we are preparing to run M7 enforced parallel workflows (where multiple worker-coders might run concurrently), we need strict consistency guarantees when promoting code to avoid dirty overwrites, silent data loss, and interleaved partial states.

## 1. Baseline Authority

**Question:** How do we define the "baseline" state of the workspace when a worker task begins, so we can detect if it has drifted before promotion?
- *Option A (Git Tree Hash):* Rely on `git rev-parse HEAD`. Pros: Native to git. Cons: Requires staging all changes; fails if the root workspace isn't clean or isn't a git repo; doesn't track unstaged drift during the run.
- *Option B (Snapshot Manifest Hash):* At task startup, the isolation guard already takes a file-system snapshot (size + mtime) of `target_paths`. We hash this manifest. Pros: Decoupled from git state; precisely scoped to the actual files the task cares about. Cons: Requires custom hashing logic.

**Decision:** **Option B (Snapshot Manifest Hash)**.
**Rationale:** We already generate a `workspace_manifest.json` during the `isolate` phase (M9). By hashing the contents of this manifest at `T0` (task start) and comparing it to a newly generated manifest at `T1` (promotion time) *only for the files within the `target_paths` scope*, we can reliably detect out-of-band drifts without polluting or depending on git staging area.

## 2. Conflict Unit

**Question:** What is the granularity of a "conflict" when drift is detected?
- *Option A (File-level):* If *any* byte in a targeted file changed since `T0`, reject the entire promotion.
- *Option B (Hunk-level):* Generate a standard diff patch. If the specific lines changed by the worker don't overlap with the lines drifted in the host file, attempt a 3-way merge.

**Decision:** **Option A (File-level strict rejection)**.
**Rationale:** We are building an AI orchestration system, not a generic source control tool. If the main workspace drifts out from under a running AI task, the safest and most predictable action is to fail closed (`PROMOTION_CONFLICT`). Attempting hunk-level merges introduces non-deterministic merge conflicts that the Orchestrator cannot resolve autonomously. We must prioritize zero data loss and zero silent corruption over maximizing throughput.

## 3. Promotion Atomicity

**Question:** How do we physically apply the isolated changes back to the root workspace in an atomic (all-or-nothing) manner?
- *Option A (In-place copy):* Loop over files and `fs.copyFileSync`.
- *Option B (Temp apply + Rename):* Copy to a `.nexus_promote_tmp` directory, then use `fs.renameSync` (which is atomic on POSIX).
- *Option C (Patch apply with rollback):* Generate a unified diff bundle. Attempt to apply it. If any file fails, use the bundle to reverse the already-applied files.

**Decision:** **Option C (Patch apply with rollback/journaling)** using the existing `patch_bundle` architecture.
**Rationale:** Node.js `fs.renameSync` across different file systems or Docker volumes can silently fall back to non-atomic copies. The worker-coder already has a `diff_summary` / `patch_bundle` mechanism (used for LLM tool outputs). By extending this:
1. We lock the write path.
2. We evaluate the pre-flight baseline check.
3. We generate a backup bundle of the *current* state of the target files.
4. We apply the worker's patch bundle.
5. If step 4 throws an error, we immediately apply the backup bundle (rollback) and throw `PROMOTION_PARTIAL_ABORT`.

## 4. Failure Semantics

**Question:** What happens to the task and workflow when a promotion fails?
- *Decision:* Emit a terminal error `PROMOTION_CONFLICT` indicating the workspace drifted. The task is marked as `failed`, and the standard Orchestrator DAG fail-closed mechanism takes over (halting downstream dependent tasks).
- We **DO NOT** attempt auto-retry on conflicts. A conflict means the world state has changed, and the LLM's previously generated code is potentially invalid. The task must be completely re-planned and re-executed from the new baseline.

## 5. Audit Artifacts

**Decision:** Every promotion attempt MUST write an audit artifact to the runtime release pack:
- `promotion_request.json`: The `T0` baseline hash and the requested diff.
- `promotion_result.json`: The `T1` pre-flight hash, the list of applied files, and the final status (`success` or `conflict`).
- If rollback occurs, a `rollback_journal.json` must be written.