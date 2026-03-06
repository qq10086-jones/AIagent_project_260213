# Runtime Contract Hardening Contract

## Scope

This contract hardens the current runtime so that prompt script and agent contracts are enforced during orchestrator startup and workflow execution.

The goal is to move from canary-only validation to runtime blocking validation.

## Runtime Guarantees

### 1. Startup Guarantees

At orchestrator startup:
- prompt script registry must load successfully
- agent contract registry must load successfully
- prompt script to agent binding validation must pass

If any check fails:
- startup must fail immediately

### 2. Workflow Guarantees

At workflow start:
- each `prompt_script_id` referenced by a workflow step must exist
- prompt script role must match workflow step role

At step payload build:
- missing prompt script must raise hard error
- role mismatch must raise hard error

## Failure Codes

- `PROMPT_SCRIPT_REGISTRY_MISSING`
- `PROMPT_SCRIPT_NOT_FOUND`
- `PROMPT_SCRIPT_ROLE_MISMATCH`
- `PROMPT_SCRIPT_BINDING_INVALID`

## Non-Scope

- no prompt runtime executor
- no additional agent teams
- no validator engine for PM/Architect output content yet
