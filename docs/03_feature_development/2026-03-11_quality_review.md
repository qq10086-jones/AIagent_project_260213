# 2026-03-11 Quality Review

## Scope

Review target:

- `worker-coder`
- next-stage mainline hardening outcomes landed on 2026-03-11

Review role:

- QA
- architecture

## Overall Assessment

Current code quality is `medium-high and improving`.

Current project quality is `stable for controlled rollout with remaining scale-up caution`.

Reasons:

- next-stage release gate is passing
- runtime boot source drift has been brought under control
- `brain -> orchestrator` typed contract has landed
- `worker-coder/coding_service.js` has been reduced materially and is now mostly orchestration logic
- targeted tests around decomposition slices are in place

Remaining concern:

- runtime lifecycle consistency must remain a first-class regression axis as the system scales
- single-file worker-coding scope semantics must remain locked down so no fallback path can reintroduce out-of-scope writes

## Findings

### P0 Fixed During Review

`worker-coder/worker.js` previously used `Promise.race()` for global timeout without a shared finalization guard. That allowed a timeout path to emit a failed result and acknowledge the message while the original task execution could still finish later and write fact/result/ack again.

This has now been fixed by introducing `worker-coder/task_lifecycle.js` and routing task completion, timeout failure, and ack through single-finalization semantics.

Validation after the fix:

- `node --check worker-coder/task_lifecycle.js` -> `PASS`
- `node --check worker-coder/worker.js` -> `PASS`
- `npm.cmd --prefix worker-coder run test:adapter` -> `PASS`

New targeted coverage:

- `worker-coder/tests/task_lifecycle.test.js`

### P1 Fixed During Review

`worker-coder/coding_service.js` previously performed optional auto-commit through shell string concatenation:

- `git add "<file_path>" && git commit -m "..."`

This has now been hardened by introducing `worker-coder/git_side_effects.js`.

What changed:

- git operations now run through structured argument arrays instead of shell string concatenation
- auto-commit success, skip, and failure outcomes are now written to structured artifact files
- `applyPatch` now exposes `diagnostics.auto_commit` instead of relying on console-only warnings

Validation after the fix:

- `node --check worker-coder/git_side_effects.js` -> `PASS`
- `node --check worker-coder/coding_service.js` -> `PASS`
- `npm.cmd --prefix worker-coder run test:adapter` -> `PASS`

New targeted coverage:

- `worker-coder/tests/git_side_effects.test.js`

### P1 Fixed During Review

`worker-coder/scoped_delta.js` previously emitted deterministic fallback stub files even when the allowed target scope was a single file such as `workspace/sandbox/crm_site/app.js` or `workspace/sandbox/crm_site/server.js`. Under truthful live cohort validation, that behavior produced `E_UNAUTHORIZED_WRITE` noise and distorted readiness evidence.

This has now been fixed by:

- mounting live `worker-coder` source into container `/app`
- preserving `verification_plan` and task-contract fields through orchestrator request building
- disabling stub fallback for single-file implementation targets
- correcting cohort result logic so verification supersets satisfy target tiers

Validation after the fix:

- `node worker-coder/tests/scoped_delta.test.js` -> `PASS`
- debug BE-only cohort -> `PASS`
- debug FE+BE cohort -> `PASS`
- full four-case worker-coding cohort -> `PASS 4 / 4`

## Architecture Assessment

### What Is Good

- mainline work has been kept aligned with governance rather than drifting into feature sprawl
- decomposition work reduced concentrated complexity in `worker-coder`
- validation and runbook coverage are now much closer to release-grade than before

### What Still Needs Discipline

- lifecycle control around long-running worker tasks
- keeping side effects structured and observable instead of console-only
- keeping runtime evidence as the source of truth for operational readiness

## Recommendation

Current recommendation is:

1. `WS-NEXT-04` can be treated as complete for this round of structural governance
2. treat worker lifecycle consistency as a permanent regression class with dedicated tests
3. open new capability work only behind the existing release-gate and evidence discipline
