# Backend Execution Adapter Contract

## Scope

This contract defines the minimum backend execution wrapper for `WS-06-05`.

Current phase scope:
- `impl_be` only
- no frontend adapter in this task
- no generic tool abstraction beyond the minimum backend packet

## Runtime Packet

Schema:
- [backend_execution_packet.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\backend_execution_packet.schema.json)

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
- `impl_be` must receive a valid backend execution packet
- invalid execution packet blocks workflow dispatch
- packet is passed through to worker-coder delegate diagnostics

## Non-Scope

- no frontend execution adapter in this task
- no full CodingExecutor abstraction yet
- no QA verifier changes in this task
