# OpenClaw Nexus vNext
## Engineering Task List
## Focus: Brain Router + Coding Team + Execution Contracts

---

## 1. Objective

Refactor the current system from a report-centric OpenClaw stack into a Brain-routed multi-agent execution platform with a production-grade Coding Team workflow.

Primary goal:
- Discord input
- Brain intent routing
- conditional OpenClaw orchestration
- structured role-based execution
- artifact output and traceability

---

## 2. Workstream Overview

### WS-01 Input Gateway
### WS-02 Brain Router
### WS-03 Canonical Task Schema
### WS-04 OpenClaw Orchestration Refactor
### WS-05 Prompt Script Registry
### WS-06 Coding Team Workflow
### WS-07 Tool Adapter Layer
### WS-08 Artifact + State Tracking
### WS-09 Guardrails + Approval
### WS-10 Observability + UI

---

## 3. Detailed Task List

---

## WS-01 Input Gateway

### Task 01-01 Discord message normalization
Deliverable:
- normalize Discord message/event into internal request object
Requirements:
- extract text
- extract attachments
- extract user/channel/thread metadata
- preserve original raw payload for debugging

### Task 01-02 Discord response protocol
Deliverable:
- standard response modes
Modes:
- direct reply
- progress update
- artifact summary
- approval request
- final completion reply

### Task 01-03 Input source abstraction
Deliverable:
- interface for future Web UI / CLI / API gateways
Requirements:
- Discord must become one adapter, not hardcoded core logic

---

## WS-02 Brain Router

### Task 02-01 Intent taxonomy
Deliverable:
- stable enum definitions
Suggested values:
- chat
- coding
- quant
- docs
- research
- ops
- unknown

### Task 02-02 Router inference module
Deliverable:
- Brain Router service/module
Requirements:
- classify intent
- estimate complexity
- determine orchestration requirement
- assign target team
- output structured JSON only

### Task 02-03 Routing policy engine
Deliverable:
- deterministic routing rules on top of model output
Requirements:
- model decides candidate class
- policy layer validates / overrides edge cases
- high-risk or ambiguous tasks require clarification or approval

### Task 02-04 Direct chat bypass
Deliverable:
- path that returns Brain reply directly without OpenClaw
Success criteria:
- chat requests do not create unnecessary workflow records

---

## WS-03 Canonical Task Schema

### Task 03-01 TaskEnvelope schema definition
Deliverable:
- JSON schema / pydantic / zod schema
Fields:
- task_id
- source
- raw_input
- normalized_input
- intent
- sub_intent
- requires_orchestration
- target_team
- expected_outputs
- constraints
- context

### Task 03-02 Validation layer
Deliverable:
- schema validation before entering orchestrator
Requirements:
- invalid task envelope must fail early
- failure reason must be logged

### Task 03-03 Context packing
Deliverable:
- standard context packer
Requirements:
- include relevant conversation context
- attachment references
- prior task references if available

---

## WS-04 OpenClaw Orchestration Refactor

### Task 04-01 OpenClaw boundary cleanup
Deliverable:
- explicit contract for what OpenClaw receives
Requirements:
- OpenClaw should only receive normalized task envelopes
- remove first-pass intent recognition responsibilities from OpenClaw

### Task 04-02 Workflow planner
Deliverable:
- workflow planner module
Requirements:
- map task type to workflow template
Examples:
- chat → no workflow
- coding/simple → single execution path
- coding/project → multi-role workflow
- quant → quant workflow

### Task 04-03 State machine implementation
Deliverable:
- stable task lifecycle model
States:
- received
- classified
- planned
- dispatched
- running
- verifying
- completed
- failed

### Task 04-04 Retry and failure policy
Deliverable:
- retry strategy per workflow step
Requirements:
- distinguish transient tool failure vs logic failure
- maintain error provenance

---

## WS-05 Prompt Script Registry

### Task 05-01 Registry format
Deliverable:
- YAML/JSON spec for prompt scripts
Fields:
- script_id
- role
- model
- input_schema
- output_schema
- tool_permissions
- artifact_type
- validation

### Task 05-02 PM script
Deliverable:
- `pm.design_doc.v1`
Output:
- structured design document

### Task 05-03 Architect script
Deliverable:
- `architect.system_spec.v1`
Output:
- modules
- boundaries
- contracts
- implementation sequence

### Task 05-04 UI script
Deliverable:
- `ui.component_spec.v1`
Output:
- page map
- component list
- state map
- interaction notes

### Task 05-05 QA planning script
Deliverable:
- `qa.test_plan.v1`
Output:
- test matrix
- verification steps
- release checklist

---

## WS-06 Coding Team Workflow

### Task 06-01 Workflow template: coding_team.standard
Deliverable:
- standard multi-role coding workflow
Stages:
- PM
- Architect
- UI/UX optional
- Backend
- Frontend
- Integration
- QA

### Task 06-02 Role handoff contracts
Deliverable:
- contract definitions between roles
Examples:
- PM → Architect
- Architect → Backend/Frontend
- Backend/Frontend → QA

### Task 06-03 PM output validator
Deliverable:
- validator ensuring PM output contains:
  - scope
  - user stories
  - acceptance criteria
  - non-goals
  - artifact list

### Task 06-04 Architect output validator
Deliverable:
- validator ensuring architecture contains:
  - module breakdown
  - interfaces
  - dependency choices
  - risk notes

### Task 06-05 Backend execution adapter
Deliverable:
- backend execution wrapper around coding tool
Requirements:
- accept structured task packet
- return code diff / changed files / logs

### Task 06-06 Frontend execution adapter
Deliverable:
- frontend execution wrapper around coding tool
Requirements:
- same contract as backend adapter

### Task 06-07 QA verifier
Deliverable:
- QA stage implementation
Requirements:
- static checklist
- test generation/execution hook
- acceptance summary artifact

---

## WS-07 Tool Adapter Layer

### Task 07-01 Unified tool adapter interface
Deliverable:
- common interface for Codex / OpenCode / local executors / workers

### Task 07-02 Coding executor abstraction
Deliverable:
- `CodingExecutor` abstraction
Methods:
- prepare()
- execute()
- collect_artifacts()
- summarize()

### Task 07-03 Quant executor abstraction
Deliverable:
- `QuantExecutor` abstraction

### Task 07-04 Tool capability manifest
Deliverable:
- each tool declares:
  - supported task types
  - supported artifact types
  - limits
  - risk level

---

## WS-08 Artifact + State Tracking

### Task 08-01 Artifact model
Deliverable:
- artifact metadata schema
Fields:
- artifact_id
- task_id
- role
- type
- path
- mime
- created_at
- summary

### Task 08-02 Artifact persistence
Deliverable:
- save docs, patches, logs, screenshots, reports consistently

### Task 08-03 Final result packager
Deliverable:
- aggregate workflow outputs into final Discord reply package

### Task 08-04 Replay support
Deliverable:
- workflow replay/debug view
Requirements:
- inspect each step input/output

---

## WS-09 Guardrails + Approval

### Task 09-01 Risk classification
Deliverable:
- low / medium / high risk classification rules

### Task 09-02 Approval checkpoints
Deliverable:
- approval workflow for risky actions
Examples:
- file deletion
- wide code rewrite
- quant execution with market impact
- secret/config mutation

### Task 09-03 Tool permission boundaries
Deliverable:
- role-based tool allowlist

---

## WS-10 Observability + UI

### Task 10-01 Task dashboard
Deliverable:
- UI page for task status, current stage, assigned role, artifacts

### Task 10-02 Workflow timeline
Deliverable:
- visual execution timeline

### Task 10-03 Discord progress notifications
Deliverable:
- standardized progress messages at major workflow transitions

### Task 10-04 Failure reporting
Deliverable:
- user-friendly failure summary + engineer-readable logs

---

## 4. Non-Functional Requirements

### NFR-01 Determinism
Where possible, routing and workflow selection must be deterministic and policy-backed.

### NFR-02 Traceability
Every artifact and state transition must be attributable to a task and role.

### NFR-03 Replaceability
Tool providers must be swappable without changing top-level workflow contracts.

### NFR-04 Local-first
No workflow should require cloud-only dependencies unless explicitly approved.

### NFR-05 Recoverability
Workflow failure must not destroy task history or artifacts.

---

## 5. Acceptance Criteria

The refactor is accepted when the following user journeys work end-to-end:

### Journey A: Direct chat
User sends a pure chat request in Discord.
Expected:
- Brain answers directly
- no OpenClaw orchestration triggered

### Journey B: PM/design request
User asks for a product design document.
Expected:
- Brain routes to document-oriented workflow
- PM/Architect output is generated as structured artifact

### Journey C: Full coding workflow
User asks for a project feature implementation.
Expected:
- Brain classifies as coding
- OpenClaw launches coding team workflow
- PM/Architect outputs produced
- coding executor invoked
- QA artifact returned

### Journey D: Quant request
User asks for quant analysis/report.
Expected:
- Brain routes to quant path
- quant worker executes without contaminating coding workflow

---

## 6. Priority Order

P0:
- Brain Router
- TaskEnvelope schema
- direct chat bypass
- OpenClaw boundary cleanup

P1:
- Prompt Script Registry
- Coding Team workflow template
- role handoff contracts
- tool adapter abstraction

P2:
- artifact replay
- approval flows
- dashboard upgrades
- quant team formalization

---

## 7. Engineering Standard

All modules must include:
- typed schemas
- structured logs
- explicit error messages
- unit tests for routing/contracts
- integration tests for workflow transitions
- sample payloads for manual debugging

No agent module should be merged if:
- input/output contract is undefined
- artifact type is unspecified
- failure mode is undocumented
- role ownership is ambiguous
