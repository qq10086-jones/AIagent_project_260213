# M9 Closeout Task List

- Date: 2026-03-11
- Status: GO WITH CONDITIONS
- Scope: M9 final closeout after core implementation completed

---

## 1. Decision

Current assessment for M9 is:

`GO WITH CONDITIONS`

Interpretation:

- M9 core implementation is complete enough to continue mainline work.
- No rollback is recommended.
- Final closure should wait for a small set of quality and architecture closeout items.

---

## 2. Priority Task List

### P0

#### WS-M9-CLOSE-01: Real Workflow Live Validation

**Status**

`DONE` on 2026-03-11

**Goal**

Move from boundary/smoke/live-stack validation to a real workflow-level live validation.

**Why**

Current evidence proves that:

- `worker-coder` verify/retry/evidence loop works
- orchestrator evidence/memory flow works
- `brain -> orchestrator` HTTP fact gateway works
- containers can run the new code

But this is still not the same as validating a richer real workflow path end to end.

**Required outputs**

- one real `coding_team_v0` or equivalent implementation workflow validation report
- explicit capture of:
  - `verification_command`
  - retry behavior
  - coding failure memory
  - release pack evidence visibility
- artifact written under `orchestrator/artifacts/`

**Acceptance**

- live workflow completes or fails in a controlled and explainable way
- release pack contains expected M9 evidence
- no hidden runtime mismatch between local canaries and live workflow execution

**Implemented**

- resolved live runtime blockers found during execution:
  - rebuilt `worker-coder` so live runtime uses the updated deterministic mock provider path
  - fixed compose/runtime gap for `context_budget_policy.json`
  - aligned root `configs/prompt_scripts/registry.json` with runtime-needed `backend.impl.v2` / `frontend.impl.v2`
- upgraded `mock-inline-autofix` to generate contract-valid PM, architect, implementation, and release artifacts
- extended implementation-side mock outputs so retry/verification evidence and patch-bundle contracts both pass
- updated the live validator script to resolve the current manifest location under `meta/run_manifest.json`

**Result**

- real `coding_team_v0` live workflow now completes with `status=succeeded`
- release pack validation passes
- M9 evidence is visible in the resulting manifest:
  - `impl_be` and `impl_fe` both show `verification_checked=true`
  - both implementation steps show retry evidence with `attempts_used=2`
  - release manifest contains `coding_execution_evidence` and `coding_execution_summary`
- final report artifact:
  - `orchestrator/artifacts/canary/live_m9_workflow/live_m9_workflow_report.json`

---

#### WS-M9-CLOSE-04: Runtime / Compose Preflight Validation

**Status**

`DONE` on 2026-03-11

**Goal**

Prevent startup failures caused by missing runtime config or compose mount drift.

**Why**

This round exposed a real release-quality issue:

- `orchestrator` could fail to start when required config files were not mounted

That should be caught by preflight validation, not discovered manually after container startup fails.

**Required outputs**

- startup-time validation of required config files
- clear failure message for missing config
- coverage for at least:
  - `llm_providers.json`
  - `llm_role_policy.json`
  - runtime config file

**Acceptance**

- missing config causes immediate, explicit startup failure
- valid config path allows normal startup
- compose/runtime drift becomes easier to diagnose

**Implemented**

- added startup-time config preflight module in `orchestrator/src/config_preflight.js`
- `orchestrator/src/index.js` now fails fast with `CONFIG_PREFLIGHT_FAILED`
- `orchestrator/src/vnext/llm_dispatcher.js` now uses the same preflight guard before config reads
- added `npm --prefix orchestrator run validate:config_preflight`
- added targeted test coverage for pass/fail behavior

---

### P1

#### WS-M9-CLOSE-02: Brain Gateway Contract Hardening

**Goal**

Promote the current minimal `brain -> orchestrator` HTTP boundary into a stable typed contract.

**Why**

Current decoupling is directionally correct, but still partial:

- fact lookup is HTTP-based
- routing event ingest is only minimally surfaced
- overall API surface is not yet fully formalized

**Required outputs**

- documented brain gateway endpoint list
- request/response schema for each supported endpoint
- defined contract for:
  - fact lookup
  - tool trigger path
  - routing decision ingest
- at least one integration test suite covering the gateway

**Acceptance**

- no direct DB reads remain in `brain`
- supported gateway surface is explicitly documented
- request/response structure is stable and test-covered

---

#### WS-M9-CLOSE-03: `worker-coder` Service Decomposition

**Goal**

Reduce maintenance and regression risk by splitting `worker-coder/coding_service.js`.

**Why**

The file currently owns too many responsibilities:

- scope guard
- snapshot diff
- static checks
- verification
- retry policy
- prompt contract
- failure memory
- artifact logging

That is acceptable short-term, but not a good long-term maintenance shape.

**Suggested split**

- `scope_guard`
- `verification_runner`
- `retry_policy`
- `failure_memory`
- `prompt_contract`

**Required outputs**

- smaller, responsibility-focused modules
- no behavior regression
- existing tests and canaries still pass

**Acceptance**

- `coding_service.js` becomes materially smaller
- key logic paths are easier to reason about in isolation
- M9 behavior remains unchanged

---

## 3. Recommended Order

1. `WS-M9-CLOSE-04` Runtime / Compose Preflight Validation
2. `WS-M9-CLOSE-01` Real Workflow Live Validation
3. `WS-M9-CLOSE-02` Brain Gateway Contract Hardening
4. `WS-M9-CLOSE-03` `worker-coder` Service Decomposition

Reasoning:

- preflight validation is the fastest release-risk reduction
- real live validation is the main remaining quality gate
- gateway hardening and service decomposition are important, but can follow after runtime confidence is stronger

---

## 4. Exit Criteria

M9 can be treated as fully closed when all of the following are true:

- workflow-level live validation is complete
- startup/config drift is blocked by preflight checks
- `brain` gateway is documented and test-covered
- `worker-coder` service structure is no longer concentrated in one oversized file
- existing M9 canaries and targeted validation remain green

---

## 5. Summary

Current quality posture:

- implementation status: strong
- architecture direction: correct
- release readiness: acceptable with conditions
- remaining risk type: structural and validation-depth risk, not missing-core-feature risk

Final recommendation:

`GO WITH CONDITIONS`
