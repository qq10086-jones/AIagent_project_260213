# Tool Adapter Interface Contract

## Scope

This contract defines the minimum `WS-07-01` / `WS-07-02` slice for coding executors.

Current phase scope:
- `coding_executor` only
- providers:
  - `opencode`
  - `codex`
  - `qwen`
- no quant executor in this task

## Request Contract

Schema:
- [tool_adapter_request.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\tool_adapter_request.schema.json)

Current request shape:
- `adapter_type`
- `provider`
- `task_type`
- `payload`
- `context`

## Result Contract

Schema:
- [tool_adapter_result.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\tool_adapter_result.schema.json)

Current result shape:
- `ok`
- `adapter_type`
- `provider`
- `result`
- `error`

## Runtime Behavior

Current runtime hard checks:
- `impl_be` and `impl_fe` execution packets must map to a valid `coding_executor` request
- provider can vary without changing workflow contract shape
- adapter request is attached to workflow payload for execution steps

## Non-Scope

- no generic cross-domain executor framework yet
- no quant executor in this task
- no worker-side provider refactor in this task
