# OpenClaw Nexus vNext
## Milestone 2 Subtask List
## Focus: E2E Integration + DoD Closure

---

## 1. Purpose

This file defines the next allowed sub tasks after the Milestone 1 contract/data-layer slice.

It follows:
- `OpenClaw_Nexus_vNext_Design_Document.md`
- `OpenClaw_Nexus_vNext_Engineering_Task_List.md`
- `OpenClaw_Execution_Governance_Scope_Control.md`

Primary objective:
- close the gap between isolated contract modules and the live North Star pipeline
- reach Definition of Done for the highest-priority unfinished critical-path slices

Non-scope:
- no dashboard/UI build
- no new agent teams
- no quant expansion
- no memory/system expansion

---

## 2. Current QA-Based Stage Judgment

Current state is best treated as:
- `WS-01` to `WS-10` minimum contract slices: mostly present
- production integration status: incomplete
- governance status: next allowed phase is `Milestone 2 / E2E Integration + hardening`

This means the next work should focus on:
- production wiring
- workflow transition validation
- guardrail enforcement in real dispatch paths
- observability messages in real runtime paths
- stable integration test execution

---

## 3. Priority Rules

Priority order:
1. Type A tasks that unblock the North Star pipeline
2. Type A test and validation hardening for DoD closure
3. Type B improvements only if directly required by Type A validation

Definition of complete for this phase:
- integrated into the live runtime path
- has contract validation
- has integration/runtime evidence
- downstream compatibility verified

---

## 4. Subtask Checklist

### ST-01 Wire vNext dispatch path into live entrypoints

Type:
- Type A / Critical Path

Pipeline node:
- Discord Gateway -> Brain Router -> TaskEnvelope -> OpenClaw Orchestration

Why now:
- design and governance both require the North Star path to be truly executable, not only modeled in isolated modules

Scope:
- verify all real Discord/HTTP entrypoints use `normalizeInputRequest`
- verify routing consistently uses `routeTaskRequest`
- verify direct chat bypass does not create unnecessary workflow records
- verify coding workflow path consistently calls `workflowEngine.startWorkflowRun`

Deliverables:
- runtime integration checklist
- fixed entrypoint wiring where missing
- integration tests for live dispatch path

Acceptance criteria:
- Journey A direct chat works through the real entrypoint
- Journey C coding workflow starts through the real entrypoint
- dispatch responses remain schema-valid

Non-scope:
- no new workflow types

---

### ST-02 Integrate Guardrails into real dispatch and workflow execution

Type:
- Type A / Critical Path

Pipeline node:
- TaskEnvelope -> Guardrails -> OpenClaw Orchestration

Why now:
- current guardrail modules exist, but critical-path evidence is insufficient until approval and permission checks are enforced in production paths

Scope:
- connect `risk_classifier` and approval interception to actual single-agent/workflow dispatch
- connect role/tool allowlist checks before execution
- expand risky coding-action detection so `coding.delegate` requests are not implicitly treated as safe

Deliverables:
- enforced approval path in runtime flow
- enforced tool permission boundary in workflow execution
- canary/integration tests for rejection/approval cases

Acceptance criteria:
- destructive or high-risk requests yield `approval_request`
- unauthorized role/tool pair is blocked before execution
- safe requests still proceed without regression

Non-scope:
- no generalized policy engine redesign

---

### ST-03 Integrate Observability into real workflow transitions

Type:
- Type A / Critical Path

Pipeline node:
- OpenClaw Orchestration -> Observability Reporter -> Discord

Why now:
- current observability slice is only formatter-level; design success requires queryable/logged stage visibility and user-visible progress/failure reporting

Scope:
- call `observability_reporter` from real workflow transition points
- send progress updates at major step changes
- send failure summaries on terminal error paths
- ensure emitted content is safe and redacted

Deliverables:
- runtime wiring for transition notifications
- runtime wiring for failure reports
- integration evidence from actual workflow transitions

Acceptance criteria:
- workflow start, step transition, and failure each emit standardized messages
- failure payloads are schema-valid
- secrets are not exposed in emitted logs

Non-scope:
- no dashboard UI

---

### ST-04 Fix and stabilize integration test execution

Type:
- Type A / Critical Path

Pipeline node:
- validation layer across the North Star pipeline

Why now:
- governance DoD requires integration tests; current key canary execution is fragile and path-dependent

Scope:
- fix path assumptions in canary/integration scripts
- make test execution independent of invocation directory
- document required local test commands
- separate sandbox/environment failures from true code failures

Deliverables:
- corrected path resolution in integration scripts
- stable integration test command set
- test runbook

Acceptance criteria:
- coding-team integration canary runs from repository root
- integration tests have deterministic pass/fail semantics
- failure messages identify environment/setup issues clearly

Non-scope:
- no broad test framework migration

---

### ST-05 Close WS-09 and WS-10 stage review with evidence

Type:
- Type A / Critical Path

Pipeline node:
- Guardrails / Observability stage acceptance

Why now:
- current progress claims are ahead of verified evidence; governance requires explicit pass/fail decision before moving on

Scope:
- rerun stage review for `WS-09`
- rerun stage review for `WS-10`
- record pass / conditional pass / fail using runtime evidence
- update progress reports to distinguish `contract-complete` from `workflow-complete`

Deliverables:
- QA review note
- revised progress report
- explicit go/no-go for next workstream

Acceptance criteria:
- status language matches evidence
- no workstream is marked complete without runtime integration evidence

Non-scope:
- no cosmetic doc cleanup unrelated to status accuracy

---

## 5. Suggested Execution Order

1. `ST-04` Fix integration test stability first
2. `ST-01` Confirm live dispatch path wiring
3. `ST-02` Enforce guardrails in runtime
4. `ST-03` Enforce observability in runtime
5. `ST-05` Run stage review and update milestone status

Reason:
- testing must be trustworthy before stage closure
- upstream dispatch/guardrail/observability wiring must be real before declaring completion

---

## 6. Immediate TODO Snapshot

- [ ] Fix root-relative path bug in `orchestrator/scripts/canary_coding_team_workflow_integration.js`
- [ ] Verify real Discord/HTTP entrypoints fully use vNext normalization and routing
- [ ] Wire approval interception into actual dispatch path
- [ ] Wire tool permission checks into workflow execution path
- [ ] Wire observability reporter into workflow transition and failure paths
- [ ] Add integration tests covering approval, rejection, and progress notification paths
- [ ] Reissue WS-09 and WS-10 stage review with runtime evidence
- [ ] Update progress wording to avoid treating contract slices as full workstream completion

---

## 7. Exit Condition For This Subtask List

This subtask list is complete when:
- the North Star pipeline is executable end-to-end for direct chat and coding workflow journeys
- guardrails and observability are present in the real runtime path
- integration evidence exists and is repeatable
- `WS-09` and `WS-10` can be honestly marked complete
