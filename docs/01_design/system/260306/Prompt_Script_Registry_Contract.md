# Prompt Script Registry Contract

## Scope

This contract introduces the minimal Prompt Script Registry required to unlock the Coding Team workflow without expanding into new teams or non-critical subsystems.

North Star alignment:

Human Input  
-> Discord Gateway  
-> Brain Router  
-> TaskEnvelope  
-> OpenClaw Orchestration  
-> Coding Team Workflow  
-> Artifacts

## Registry Contract

Schema:
- `orchestrator/contracts/prompt_script_registry.schema.json`

Registry file:
- `configs/prompt_scripts/registry.json`

Required fields per script:
- `script_id`
- `role`
- `model`
- `input_schema`
- `output_schema`
- `tool_permissions`
- `artifact_type`
- `validation`

Optional fields:
- `system_prompt`

## Minimal Scripts in Scope

### 1. `pm.design_doc.v1`
- role: `pm`
- artifact_type: `design_doc`
- tool_permissions: `coding.delegate`

### 2. `architect.system_spec.v1`
- role: `architect`
- artifact_type: `system_spec`
- tool_permissions: `coding.delegate`

### 3. `qa.test_plan.v1`
- role: `qa`
- artifact_type: `test_plan`
- tool_permissions: `coding.execute`, `coding.delegate`

## Coding Workflow Integration

The following `coding_team_v0` steps now carry `prompt_script_id`:
- `pm_spec -> pm.design_doc.v1`
- `arch_design -> architect.system_spec.v1`
- `qa_verify -> qa.test_plan.v1`

Runtime guarantees:
- workflow step payload includes `prompt_script_id`
- workflow step payload includes `prompt_script` contract object when available
- `coding.delegate` prompt generation includes prompt script metadata
- `qa_verify` acceptance context includes prompt script metadata

## Failure Conditions

- missing prompt script registry file
- invalid prompt script registry structure
- workflow step uses invalid `prompt_script_id` type
- runtime payload cannot attach required prompt script metadata

## Non-Scope

- no UI prompt script
- no quant prompt script
- no multi-team prompt registry
- no prompt execution engine abstraction beyond current workflow payload attachment
