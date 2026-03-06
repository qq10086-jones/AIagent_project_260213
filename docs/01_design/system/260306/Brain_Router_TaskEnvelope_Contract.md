# Brain Router and TaskEnvelope Contract

## Scope

This contract closes the minimum upstream Definition of Done for:
- Discord Gateway normalization
- Brain Router output
- TaskEnvelope normalization
- Discord to OpenClaw dispatch boundary

This contract supports the North Star path:

Human Input  
-> Discord Gateway  
-> Brain Router  
-> TaskEnvelope  
-> OpenClaw Orchestration  
-> Coding Team Workflow  
-> Artifacts

## Contract Set

### 1. Discord Request Contract

Schema:
- `orchestrator/contracts/discord_request.schema.json`

Required fields:
- `source=discord`
- `raw_input`

Optional dispatch controls:
- `provider`
- `model`
- `fast_mode`
- `model_preference`

### 2. Brain Router Output Contract

Schema:
- `orchestrator/contracts/brain_router.output.schema.json`

Required fields:
- `decision`
- `route`
- `task_envelope`

Allowed decisions:
- `direct_reply`
- `single_agent`
- `orchestrated_workflow`
- `human_review_required`

### 3. TaskEnvelope Contract

Schema:
- `orchestrator/contracts/task_envelope.schema.json`

Required fields:
- `task_id`
- `source`
- `raw_input`
- `normalized_input`
- `intent`
- `requires_orchestration`
- `target_team`
- `expected_outputs`
- `constraints`
- `context`

### 4. Dispatch Success Response Contract

Schema:
- `orchestrator/contracts/dispatch_success_response.schema.json`

Allowed response modes:
- `direct_reply`
- `progress_update`
- `approval_request`
- `final_completion_reply`

### 5. Dispatch Error Response Contract

Schema:
- `orchestrator/contracts/dispatch_error_response.schema.json`

Required error fields:
- `ok=false`
- `response_mode=final_completion_reply`
- `error`
- `error_code`

## Routing Rules

### Chat
- output decision: `direct_reply`
- OpenClaw orchestration must not start

### Coding Simple
- output decision: `single_agent`
- tool path defaults to `coding.delegate`

### Coding Complex
- output decision: `orchestrated_workflow`
- workflow defaults to `coding_team_v0`

### Coder Directive
- `/coder` bypasses normal ambiguity
- forced intent: `coding`
- forced decision: `orchestrated_workflow`
- default model: `qwen-coder-next`
- explicit override allowed via:
  - `@model=<name>`
  - `model=<name>`
- explicit provider override allowed via:
  - `@provider=<name>`
  - `provider=<name>`

## Failure Conditions

Failure codes currently defined at the boundary:
- `BAD_REQUEST`
- `TASK_ENVELOPE_INVALID`
- `REGISTRY_INVALID`

Boundary rejection conditions:
- empty `raw_input`
- invalid task envelope shape
- invalid registry tool/workflow contract

Boundary response guarantees:
- chat responses return `direct_reply`
- queued task/workflow responses return `progress_update` or `approval_request`
- boundary errors return `final_completion_reply`

### 6. Route API Response Contract

Schema:
- `orchestrator/contracts/route_response.schema.json`

Guaranteed fields:
- `normalized`
- `decision`
- `route`
- `task_envelope`

### 7. Dispatch Preview Contract

Schema:
- `orchestrator/contracts/dispatch_preview_response.schema.json`

Preview guarantees:
- chat request previews resolve to `direct_reply`
- simple coding previews resolve to `progress_update`
- coder directive previews resolve to workflow `progress_update`
- invalid input previews resolve to `final_completion_reply`

## Definition of Done Coverage

Covered:
- input schema defined
- output schema defined
- contract document exists
- integration tests added
- failure conditions defined

Not covered yet:
- full live Discord end-to-end test against running worker stack
- prompt script registry contracts
- downstream role artifact validators beyond current coding workflow
