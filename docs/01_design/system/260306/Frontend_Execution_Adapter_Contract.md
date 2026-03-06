# Frontend Execution Adapter Contract

## Scope

This contract defines the minimum frontend execution wrapper for `WS-06-06`.

Current phase scope:
- `impl_fe` only
- no generic executor abstraction beyond the minimum frontend packet
- no QA verifier changes in this task

## Runtime Packet

Schema:
- [frontend_execution_packet.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\frontend_execution_packet.schema.json)

Current packet fields:
- `adapter_id`
- `role`
- `step_id`
- `target_paths`
- `required_outputs`
- `input_artifacts`
- `execution_mode`
- `verification_hint`
- `provider_hint`
- `model_hint`

## Runtime Behavior

Current runtime hard checks:
- `impl_fe` must receive a valid frontend execution packet
- invalid execution packet blocks workflow dispatch
- packet is passed through to worker-coder delegate diagnostics

## Non-Scope

- no backend changes in this task beyond existing compatibility
- no full CodingExecutor abstraction yet
- no QA verifier changes in this task
