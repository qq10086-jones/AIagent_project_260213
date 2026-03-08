# OpenClaw Nexus Progress Update (AGENTS.md Style)

## Snapshot
- Date: `2026-03-06 21:31:29`
- Sprint context: `vnext_brain_first + m2_e2e_integration + guardrails + observability`
- North Star: `Discord/HTTP input -> Brain Router -> TaskEnvelope -> Guardrails -> OpenClaw orchestration -> Observability -> Discord`
- Current status: `Milestone 2 / E2E Integration + hardening` remains active. Live runtime evidence is now available; `WS-09` can be reviewed as `pass`, while `WS-10` still remains open pending notification-emission proof.

## Executive Summary
This session continued the allowed `Milestone 2` work under `OpenClaw_Nexus_vNext_Subtask_List_M2_E2E_20260306.md`.

The main result is that several previously contract-only slices are now wired into the real runtime path:
- key canary scripts now run from repository root
- real `/chat` non-quant requests now enter the vNext dispatch path
- workflow execution now enforces role/tool permission boundaries before dispatch
- workflow transitions and failures now have runtime observability wiring through Discord context
- entrypoint-level `node:test` coverage now exists for direct reply, single-agent approval response, and workflow context propagation
- `/chat` entrypoint-level `node:test` coverage now exists for direct chat bypass, coding approval response, and quant legacy-path preservation
- approval entrypoint-level `node:test` coverage now exists for approve, reject, and invalid-token paths
- workflow notification delivery `node:test` coverage now exists for success transition, failure chunked reply, and missing-context skip paths
- live validation script now exists for direct chat bypass, approval reject, and approval approve evidence capture when the local service is available
- live workflow validation script now exists for `workflow-runs/start -> poll -> timeline -> artifacts` evidence capture when the local service is available
- bundled live validation entrypoint and runbook now exist so final evidence collection is fixed and repeatable

The live environment is now reachable and the bundled live validation flow runs successfully. This closes the earlier environment blocker and upgrades the strongest evidence from canary/runtime-module level to actual live-service runtime evidence for `/chat`, approval/reject, workflow runtime polling, timeline, and artifact access.

## What Was Implemented
- **ST-04 / Integration script stability**
  - Added shared script path resolver:
    - `orchestrator/scripts/_paths.js`
  - Fixed root-relative path assumptions in key scripts:
    - `canary_coding_team_workflow_integration.js`
    - `canary_brain_router.js`
    - `canary_coding_team.js`
    - `canary_runtime_contract_hardening.js`
    - `validate_registry.js`
    - `validate_go_nogo.js`
    - `canary_tool_permission_guard.js`
    - `canary_observability_reporter.js`
    - `canary_approval_interceptor.js`
  - Added missing npm script entries in:
    - `orchestrator/package.json`

- **ST-01 / Live dispatch path wiring**
  - Updated `orchestrator/src/index.js` so `/chat` non-quant requests use `executeVNextDispatch(...)`.
  - Real Discord coding workflow startup now passes runtime context into workflow dispatch instead of dropping it.
  - Extracted `orchestrator/src/vnext/runtime_dispatch.js` so the entrypoint dispatch path can be tested without booting the full service.
  - Extracted `orchestrator/src/vnext/chat_entrypoint.js` so `/chat` routing behavior can be verified without starting the full HTTP service.

- **ST-02 / Guardrails runtime integration**
  - Updated `orchestrator/src/workflow_engine.js` to enforce `validateToolPermission(...)` before workflow step dispatch.
  - Updated `orchestrator/src/vnext/tool_permission_guard.js` role matrix to match current registry role names (`pm`, `architect`, `frontend`, `backend`, `qa`) while preserving legacy aliases.
  - Extracted `orchestrator/src/vnext/approval_entrypoint.js` so approval/reject HTTP behavior can be verified without starting the full service.
  - Expanded workflow integration canary to verify:
    - unauthorized role/tool dispatch fails with `TOOL_PERMISSION_DENIED`
    - risky first step can enter `waiting_approval`
    - approval on risky first step returns workflow step to `queued`
    - rejection on risky first step closes workflow with `APPROVAL_REJECTED`

- **ST-03 / Observability runtime wiring**
  - Updated `orchestrator/src/index.js` to call `formatTransitionNotification(...)` and `formatFailureReport(...)` from live workflow result handling.
  - Workflow notifications now depend on preserved Discord runtime context rather than isolated formatter-only tests.
  - Added `orchestrator/src/vnext/workflow_runtime_notifier.js` so workflow runtime state can be deterministically converted into transition/failure notifications.
  - Extracted `orchestrator/src/vnext/workflow_notification_delivery.js` so workflow result-consumer notification delivery can be verified without starting the full service.
  - Added `orchestrator/scripts/live_validate_vnext_runtime.js` so live runtime evidence can be captured with a single command once local orchestrator service is reachable.
  - Added `orchestrator/scripts/live_validate_workflow_runtime.js` so workflow runtime polling/timeline/artifact evidence can be captured with a single command once local orchestrator service is reachable.
  - Added `orchestrator/scripts/live_validate_m2_e2e_bundle.js` and `docs/03_feature_development/M2_E2E_LIVE_EVIDENCE_RUNBOOK_20260306.md` so the final evidence collection flow has one command and explicit pass criteria.

## Runtime Evidence
- `node orchestrator/scripts/canary_coding_team_workflow_integration.js -> ok`
- `node orchestrator/scripts/canary_brain_router.js -> ok (16/16)`
- `node orchestrator/scripts/canary_runtime_contract_hardening.js -> ok`
- `node orchestrator/scripts/canary_tool_permission_guard.js -> ok`
- `node orchestrator/scripts/canary_observability_reporter.js -> ok`
- `node orchestrator/scripts/canary_approval_interceptor.js -> ok`
- `node orchestrator/scripts/canary_workflow_runtime_notifier.js -> ok`
- `cmd /c npm --prefix orchestrator run canary:coding_team_workflow_integration -> ok`
- `cmd /c npm --prefix orchestrator run validate:registry -> ok`
- `cmd /c npm --prefix orchestrator run canary:observability_reporter -> ok`
- `cmd /c npm --prefix orchestrator run canary:workflow_runtime_notifier -> ok`
- `node --test orchestrator/test/runtime_dispatch.integration.test.js orchestrator/test/workflow_runtime_notifier.integration.test.js orchestrator/test/tool_permission_guard.integration.test.js orchestrator/test/vnext.test.js orchestrator/test/discord_dispatch.integration.test.js orchestrator/test/brain_router.integration.test.js -> ok (18/18)`
- `node --test orchestrator/test/chat_entrypoint.integration.test.js orchestrator/test/runtime_dispatch.integration.test.js orchestrator/test/workflow_runtime_notifier.integration.test.js orchestrator/test/tool_permission_guard.integration.test.js orchestrator/test/vnext.test.js orchestrator/test/discord_dispatch.integration.test.js orchestrator/test/brain_router.integration.test.js -> ok (21/21)`
- `node --test orchestrator/test/approval_entrypoint.integration.test.js orchestrator/test/chat_entrypoint.integration.test.js orchestrator/test/runtime_dispatch.integration.test.js orchestrator/test/workflow_runtime_notifier.integration.test.js orchestrator/test/tool_permission_guard.integration.test.js orchestrator/test/vnext.test.js orchestrator/test/discord_dispatch.integration.test.js orchestrator/test/brain_router.integration.test.js -> ok (24/24)`
- `node --test orchestrator/test/workflow_notification_delivery.integration.test.js orchestrator/test/approval_entrypoint.integration.test.js orchestrator/test/chat_entrypoint.integration.test.js orchestrator/test/runtime_dispatch.integration.test.js orchestrator/test/workflow_runtime_notifier.integration.test.js orchestrator/test/tool_permission_guard.integration.test.js orchestrator/test/vnext.test.js orchestrator/test/discord_dispatch.integration.test.js orchestrator/test/brain_router.integration.test.js -> ok (27/27)`
- `cmd /c npm --prefix orchestrator run validate:live_vnext_runtime -> ok (direct chat bypass, approval reject, and approval approve all passed in live runtime; report captured under orchestrator/artifacts/canary/live_vnext_runtime/live_vnext_runtime_report.json)`
- `cmd /c npm --prefix orchestrator run validate:live_workflow_runtime -> ok under tightened success criteria (live workflow now succeeds end-to-end; report captured under orchestrator/artifacts/canary/live_workflow_runtime/live_workflow_runtime_report.json)`
- `cmd /c npm --prefix orchestrator run validate:live_m2_e2e -> ok (live vNext path and live workflow path both pass under the tightened workflow-success criterion)`

## Subtask Review
- `ST-01 Wire vNext dispatch path into live entrypoints`
  - Status: `partial pass`
  - Evidence:
    - `/chat` non-quant path now uses vNext dispatch
    - Discord coding workflow path passes runtime context
    - entrypoint-level runtime dispatch test now verifies direct-reply, approval-request, and workflow context paths
    - `/chat` entrypoint test now verifies direct chat bypass and quant legacy-path preservation
    - live `/chat` validation now proves direct chat bypass creates no task records and approval/reject requests round-trip through the real entrypoint
  - Remaining:
    - explicit integration proof for Discord live entrypoints

- `ST-02 Integrate Guardrails into real dispatch and workflow execution`
  - Status: `partial pass`
  - Evidence:
    - runtime approval path already exists through `policy.js + enqueueTask`
    - workflow step permission boundary now blocks unauthorized role/tool pairs before dispatch
    - workflow canary now covers approval gate, approval-approved, approval-rejected, and permission deny paths
    - runtime dispatch test now verifies risky single-agent requests preserve `approval_request` response behavior
    - `/chat` entrypoint test now verifies coding requests can surface `approval_request` through vNext dispatch
    - approval entrypoint test now verifies approve requeue, reject failure closure, and invalid-token rejection
    - live `/chat` validation now proves approval-reject and approval-approve behavior through the real HTTP dispatch path
  - Remaining:
    - stronger live-service evidence for workflow-step tool permission denial beyond canary/integration scope

- `ST-03 Integrate Observability into real workflow transitions`
  - Status: `partial pass`
  - Evidence:
    - transition/failure formatting is wired into runtime result handling
    - Discord workflow context is now preserved for workflow-started tasks
    - observability canary remains green
    - workflow runtime notifier canary covers runtime-state-to-notification conversion
    - runtime dispatch test now verifies workflow context reaches workflow engine startup
    - workflow notification delivery test now verifies result-consumer-level transition send, failure chunked reply, and missing-context skip behavior
    - live workflow validation now proves the actual workflow succeeds and runtime status, timeline, and artifacts are queryable from the actual service
  - Remaining:
    - live-service proof that start/transition/failure notifications all emit during actual runs
    - live-service progress/failure notification evidence beyond result-consumer integration scope

- `ST-04 Fix and stabilize integration test execution`
  - Status: `pass`
  - Evidence:
    - root-relative path bug fixed
    - targeted canaries now run from repository root
    - npm command path verified via `cmd /c npm ...`

- `ST-05 Close WS-09 and WS-10 stage review with evidence`
  - Status: `in progress`
  - This document now records live runtime evidence closure for `WS-09` and a real workflow success path, but final milestone closure still depends on `WS-10` notification-emission proof.

## Stage Review
- `WS-09 Guardrails + Approval`
  - Judgment: `pass`
  - Reason:
    - runtime permission gate and approval evidence now exist
    - live `/chat` evidence now proves direct approval-request, reject, and approve behavior from the real service entrypoint

- `WS-10 Observability + UI`
  - Judgment: `conditional pass`
  - Reason:
    - runtime notification wiring now exists
    - live workflow success-path evidence now exists and status, timeline, and artifacts are queryable from the real service
    - live-service E2E evidence for progress/failure notification emission is still incomplete

## Governance Decision
- Allowed next work: continue `Milestone 2 / E2E Integration + hardening`
- Not allowed yet:
  - declaring `WS-10` complete
  - starting broader downstream feature expansion

## Remaining TODOs
- [ ] Add stronger integration evidence for progress/failure notification emission through real workflow transitions
- [ ] Update milestone summary so status language matches current runtime evidence

## Files Changed In This Session
- `orchestrator/src/index.js`
- `orchestrator/src/workflow_engine.js`
- `orchestrator/src/vnext/tool_permission_guard.js`
- `orchestrator/src/vnext/workflow_runtime_notifier.js`
- `orchestrator/src/vnext/runtime_dispatch.js`
- `orchestrator/src/vnext/chat_entrypoint.js`
- `orchestrator/src/vnext/approval_entrypoint.js`
- `orchestrator/src/vnext/workflow_notification_delivery.js`
- `orchestrator/scripts/_paths.js`
- `orchestrator/scripts/live_validate_vnext_runtime.js`
- `orchestrator/scripts/live_validate_workflow_runtime.js`
- `orchestrator/scripts/live_validate_m2_e2e_bundle.js`
- `orchestrator/scripts/canary_brain_router.js`
- `orchestrator/scripts/canary_coding_team.js`
- `orchestrator/scripts/canary_coding_team_workflow_integration.js`
- `orchestrator/scripts/canary_runtime_contract_hardening.js`
- `orchestrator/scripts/canary_tool_permission_guard.js`
- `orchestrator/scripts/canary_observability_reporter.js`
- `orchestrator/scripts/canary_approval_interceptor.js`
- `orchestrator/scripts/canary_workflow_runtime_notifier.js`
- `orchestrator/scripts/validate_registry.js`
- `orchestrator/scripts/validate_go_nogo.js`
- `orchestrator/package.json`
- `docs/03_feature_development/M2_E2E_LIVE_EVIDENCE_RUNBOOK_20260306.md`
- `orchestrator/test/chat_entrypoint.integration.test.js`
- `orchestrator/test/approval_entrypoint.integration.test.js`
- `orchestrator/test/workflow_notification_delivery.integration.test.js`
- `orchestrator/test/runtime_dispatch.integration.test.js`
