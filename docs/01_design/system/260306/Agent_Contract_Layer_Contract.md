# Agent Contract Layer Contract

## Scope

This contract introduces the minimal MetaGPT-inspired agent contract layer required by the current phase.

The layer does not implement agent runtime execution.
It only defines:
- agent contract files
- contract schema
- prompt script to agent binding rules
- canary validation

## Agent Contract Schema

Schema:
- `orchestrator/contracts/agent_contract.schema.json`

Required fields:
- `agent_id`
- `role`
- `mission`
- `input_schema`
- `output_schema`
- `allowed_tools`
- `forbidden_actions`
- `SOP`

## Minimal Agent Set

Files:
- `configs/agents/pm_agent.json`
- `configs/agents/architect_agent.json`
- `configs/agents/ui_agent.json`
- `configs/agents/backend_agent.json`
- `configs/agents/frontend_agent.json`
- `configs/agents/qa_agent.json`

## Binding Rules

Prompt scripts must define:
- `agent_id`
- `role`

Binding is valid only if:
- referenced agent exists
- script role matches agent role
- script tool permissions are a subset of agent allowed tools

## Current Workflow Impact

Current runtime integration remains minimal:
- workflow payload carries prompt script metadata
- prompt scripts now identify target agents
- agent contract registry is validated independently

No new runtime executor is introduced in this patch.
