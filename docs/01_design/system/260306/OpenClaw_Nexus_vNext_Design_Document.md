# OpenClaw Nexus vNext
## Brain-Routed Multi-Agent Execution System
## Design Document

---

## 1. Vision

OpenClaw Nexus is a local-first multi-agent execution system designed to convert human natural-language input into structured execution workflows.

The system uses Discord as the primary interaction gateway. Incoming requests are first analyzed by a Brain Router layer, which determines user intent, task complexity, and whether orchestration is required. Simple conversational tasks are answered directly by the Brain layer. Execution-oriented tasks are transformed into structured task envelopes and passed to OpenClaw, which acts as the orchestration trunk and dispatches specialized agents, workers, and tools.

The long-term goal is to build a central AI operating system that can manage multiple specialized teams, beginning with a Coding Team and later expanding to Quant, E-commerce, Short-video, Research, and other vertical assistants.

---

## 2. Core Principles

### 2.1 Local-first
All core inference, orchestration, artifact generation, and task execution should be runnable in the local environment or local Docker stack.

### 2.2 Brain before orchestration
Not every request should enter OpenClaw. The Brain layer must classify requests first and only escalate requests that require workflow execution.

### 2.3 OpenClaw as trunk, not brain
OpenClaw is the orchestration trunk of the system. It should not become a monolithic reasoning engine for all requests.

### 2.4 Agent = contract, not persona
Each agent must be defined by:
- input schema
- output schema
- allowed tools
- success criteria
- failure handling
- escalation rules

### 2.5 Document-first for non-coding roles
For PM, Architect, UI, Analyst, Research, QA-planning and other non-coding tasks, the primary output is structured documentation, not immediate code generation.

### 2.6 Execution-first for coding roles
For frontend, backend, integration, testing and patch delivery roles, the primary output is executable changes, diffs, code files, tests, and runbooks.

---

## 3. High-Level Architecture

### 3.1 Input Layer
- Human natural-language input
- Primary gateway: Discord
- Future gateways:
  - Web UI
  - CLI
  - API
  - Scheduled triggers

### 3.2 Brain Router Layer
Responsibilities:
- parse user input
- classify intent
- detect complexity
- detect required execution mode
- detect whether OpenClaw orchestration is necessary
- normalize user request into structured task object

Possible intent classes:
- chat
- coding
- quant
- docs
- research
- ops
- unknown

Possible decision outcomes:
- direct_reply
- single_agent
- orchestrated_workflow
- human_review_required

### 3.3 OpenClaw Orchestration Layer
Responsibilities:
- receive structured task envelope
- generate workflow/DAG
- assign role-based agents
- manage state transitions
- invoke tools/workers
- collect artifacts
- return execution result to Discord

### 3.4 Execution Layer
Execution is performed by specialized workers and tool adapters.

Current / planned executors:
- worker-coder
- worker-quant
- prompt-script agents
- Codex adapter
- OpenCode adapter
- browser / evidence tools
- shell / sandbox tools
- future vertical workers

### 3.5 Artifact Layer
Stores:
- design docs
- task lists
- code patches
- screenshots
- reports
- QA summaries
- workflow logs

### 3.6 Memory / Context Layer
Stores:
- task history
- project context
- reusable constraints
- prior design docs
- reusable prompts / prompt scripts
- execution traces for replay and debugging

---

## 4. Canonical Task Object

All non-trivial tasks must be normalized into a canonical task envelope before execution.

Example schema:

```json
{
  "task_id": "uuid",
  "source": "discord",
  "user_input": "Build a CRM MVP with login, customer list and notes",
  "intent": "coding",
  "sub_intent": "project_bootstrap",
  "requires_orchestration": true,
  "target_team": "coding_team",
  "expected_outputs": [
    "design_doc",
    "task_breakdown",
    "repo_changes"
  ],
  "constraints": {
    "local_only": true,
    "approval_mode": "manual",
    "risk_level": "medium"
  },
  "context": {
    "channel_id": "discord-channel-id",
    "thread_id": "discord-thread-id",
    "attachments": []
  }
}
```

---

## 5. Brain Router Design

### 5.1 Responsibilities
The Brain Router is the first decision layer after user input ingestion.

It must answer:
1. What is the user trying to do?
2. Is this a direct-answer task or an execution task?
3. Does the request need orchestration?
4. Which team should handle it?
5. What artifacts are expected?

### 5.2 Router Output
The router output must be structured JSON, not free text.

### 5.3 Routing Policy
- `chat` → direct Brain response
- `coding` + simple patch → single coding path
- `coding` + multi-role project → OpenClaw orchestration
- `quant` → quant pipeline / quant worker
- `docs/research` → document-oriented agent workflow
- `unknown` → clarification or fallback

### 5.4 Escalation Rules
Escalate to OpenClaw when:
- multiple roles are needed
- multiple artifacts are expected
- approval checkpoints are required
- the task spans design + implementation + verification
- the task includes external tool invocations

---

## 6. OpenClaw Role in the System

OpenClaw serves as the orchestration trunk.

OpenClaw responsibilities:
- workflow planning
- role dispatch
- tool invocation
- execution tracking
- artifact aggregation
- retry/recovery hooks
- audit trail

OpenClaw should not be responsible for:
- all first-pass intent recognition
- all direct chat handling
- unrestricted tool execution without policy
- role definition itself

---

## 7. Agent Model

### 7.1 Agent Categories
#### A. Planning Agents
- PM Agent
- Architect Agent
- Analyst Agent
- Research Agent

#### B. Design Agents
- UI/UX Agent
- API Design Agent
- Data Model Agent

#### C. Execution Agents
- Frontend Agent
- Backend Agent
- Integration Agent
- Quant Execution Agent

#### D. Verification Agents
- QA Agent
- Test Agent
- Reviewer Agent
- Risk/Guard Agent

### 7.2 Agent Contract Template
Each agent must define:
- name
- mission
- input schema
- output schema
- tools allowed
- forbidden actions
- dependencies
- success criteria
- retry policy
- escalation policy

---

## 8. Coding Team Design

Coding Team is the first fully realized vertical team in the system.

### 8.1 Mission
Convert a user’s product request into:
- structured design documentation
- implementation plan
- code changes
- tests
- verification artifacts

### 8.2 Standard Workflow
1. PM Agent
2. Architect Agent
3. UI/UX Agent (optional)
4. Backend Agent
5. Frontend Agent
6. Integration Agent
7. QA Agent
8. Release / Runbook Agent

### 8.3 Role Definitions

#### PM Agent
Input:
- user request
- existing project context

Output:
- clarified problem statement
- scope
- user stories
- acceptance criteria
- milestone breakdown

#### Architect Agent
Input:
- PM output
- existing codebase/project constraints

Output:
- system design
- module boundaries
- interface contracts
- dependency decisions
- implementation sequence

#### UI/UX Agent
Input:
- PM and Architect outputs

Output:
- page map
- component inventory
- state transitions
- interaction notes
- design constraints

#### Backend Agent
Input:
- architecture + API/data contracts

Output:
- backend code or patch
- migrations
- service logic
- tests

#### Frontend Agent
Input:
- UI spec + API contract

Output:
- UI code or patch
- page/component implementation
- interaction logic
- tests

#### QA Agent
Input:
- implementation outputs

Output:
- test plan
- test execution checklist
- defect report
- acceptance result

---

## 9. Non-Coding Agent Workflows

### 9.1 PM/Planning Workflow
Used when the task requires:
- project planning
- documentation
- proposal writing
- task decomposition
- milestone design

Primary output:
- design doc
- task list
- acceptance matrix

### 9.2 UI Workflow
Used when the task requires:
- wireframes
- component planning
- frontend information architecture

Primary output:
- UI spec
- component map
- state/event matrix

### 9.3 Research Workflow
Used when the task requires:
- technical comparison
- literature mapping
- decision support

Primary output:
- research brief
- options matrix
- recommendation memo

---

## 10. Tooling Strategy

### 10.1 Tool Classes
- reasoning models
- code generation tools
- browser tools
- shell/sandbox tools
- quant analysis tools
- artifact generation tools

### 10.2 Current Tool Mapping
- direct conversation → Brain LLM
- structured planning docs → prompt-script agents + LLM
- coding execution → Codex / OpenCode / worker-coder
- quant execution → worker-quant
- browser evidence → OpenClaw browser tools

### 10.3 Tool Abstraction Requirement
Tools must be wrapped behind stable interfaces so that tool providers can be swapped without changing upstream orchestration logic.

---

## 11. Prompt Script Registry

A Prompt Script Registry must be introduced for non-coding and semi-structured tasks.

Each script definition should include:
- script_id
- target_agent
- input schema
- output schema
- preferred model
- temperature / reasoning mode
- allowed tools
- artifact type
- validation rules

Examples:
- `pm.design_doc.v1`
- `architect.system_spec.v1`
- `ui.component_spec.v1`
- `qa.test_plan.v1`

This registry is critical to making agent behavior reproducible and controllable.

---

## 12. Workflow Patterns

### 12.1 Direct Reply Pattern
Used for:
- chat
- quick Q&A
- no artifact requirement

### 12.2 Single-Agent Pattern
Used for:
- simple doc generation
- simple patch generation
- narrow analysis tasks

### 12.3 Multi-Agent Workflow Pattern
Used for:
- project implementation
- multi-role decomposition
- design → build → verify flows

### 12.4 Human Approval Pattern
Used for:
- destructive actions
- risky patches
- production deployments
- financial actions
- external communication

---

## 13. State Machine

Suggested task lifecycle:
- received
- classified
- normalized
- planned
- dispatched
- running
- waiting_for_dependency
- awaiting_approval
- verifying
- completed
- failed
- canceled

Each transition must be logged and queryable.

---

## 14. Quality Gates

No workflow should be considered complete without explicit quality gates.

Minimum gates:
- schema-valid output
- artifact generated
- execution log saved
- validation passed
- failure reason captured if unsuccessful

For coding workflows:
- patch generated
- tests generated or executed
- run instructions produced
- rollback note produced where applicable

---

## 15. Guardrails

### 15.1 Risk Policy
Actions must be labeled:
- low risk
- medium risk
- high risk

### 15.2 High-Risk Actions
Require manual approval:
- deleting files
- modifying secrets
- production deployment
- broker / trading execution
- external publishing
- system-wide config changes

### 15.3 Local Safety
The system must respect local-only execution constraints and avoid unintended cloud dependency.

---

## 16. Observability

The system must expose:
- task status
- workflow stage
- current agent
- artifact links
- error logs
- execution timeline

Recommended views:
- Discord progress updates
- local UI dashboard
- artifact browser
- replay/debug console

---

## 17. Current Gap Analysis

The current implementation already contains useful infrastructure:
- OpenClaw
- orchestrator
- worker-quant
- worker-coder
- Docker stack
- UI
- artifact storage

However, the system is still overly centered on specific report pipelines and has not yet fully realized:
- Brain-first routing
- canonical task envelopes
- agent contracts
- prompt script registry
- coding team workflow standardization
- explicit separation between chat / routing / orchestration / execution

---

## 18. vNext Implementation Priorities

Priority 1:
- Brain Router
- canonical task schema
- OpenClaw routing boundary
- coding team workflow

Priority 2:
- prompt script registry
- PM / Architect / UI / QA document agents
- worker-coder integration contracts

Priority 3:
- quant team formalization
- artifact replay
- multi-team expansion
- memory/context layer

---

## 19. Definition of Success

The system is considered successful when:
1. A Discord request can be correctly classified by Brain.
2. Chat requests are answered directly without orchestration.
3. Coding requests can trigger a structured Coding Team workflow.
4. PM/Architect/UI tasks produce reproducible high-quality documents.
5. Backend/Frontend tasks can call coding executors through stable interfaces.
6. Quant tasks can route into quant worker without contaminating coding workflows.
7. All artifacts, states, and logs are traceable end-to-end.

---

## 20. Final Product Positioning

OpenClaw Nexus is not a single agent.
It is a local-first AI operating system composed of:
- one input plane
- one routing brain
- one orchestration trunk
- multiple specialized execution teams

The first production-grade team to be completed is the Coding Team.
