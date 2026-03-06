# Task Specification
## brain_router_dod_close_v1

### Task Name
`brain_router_dod_close_v1`

### Pipeline Node
- Discord Gateway
- Brain Router
- TaskEnvelope Normalization
- OpenClaw Orchestration Boundary

### Task Type
Type A - Critical Path

### Upstream Dependency
- Intent taxonomy defined
- Discord input normalization exists
- TaskEnvelope base structure exists

### Deliverables
- `orchestrator/contracts/discord_request.schema.json`
- `orchestrator/contracts/task_envelope.schema.json`
- `orchestrator/contracts/brain_router.output.schema.json`
- `orchestrator/contracts/dispatch_success_response.schema.json`
- `orchestrator/contracts/dispatch_error_response.schema.json`
- `orchestrator/contracts/route_response.schema.json`
- `orchestrator/contracts/dispatch_preview_response.schema.json`
- `docs/01_design/system/260306/Brain_Router_TaskEnvelope_Contract.md`
- `orchestrator/test/brain_router.integration.test.js`
- `orchestrator/test/discord_dispatch.integration.test.js`

### Non-Scope Declaration
- No prompt script registry
- No PM / Architect / UI / QA agent expansion
- No dashboard work
- No quant team formalization
- No memory system expansion

### Acceptance Criteria
- Discord request contract is schema-defined
- Brain Router output contract is schema-defined
- TaskEnvelope contract is schema-defined
- Chat requests resolve to `direct_reply`
- Complex coding requests resolve to `orchestrated_workflow`
- `/coder` defaults to `qwen-coder-next`
- `/coder` supports manual model override
- dispatch responses resolve to schema-stable response modes
- route API responses resolve to schema-stable payloads
- dispatch API responses resolve to schema-stable payloads
- Integration tests cover chat, coding, and coder directive paths
