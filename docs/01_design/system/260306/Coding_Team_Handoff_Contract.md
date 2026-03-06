# Coding Team Handoff Contract

## Scope

This contract defines the minimum handoff rules between critical Coding Team stages.

Current phase scope:
- PM -> Architect
- Architect -> Backend / Frontend / QA

## Registry

File:
- `orchestrator/configs/contracts/coding_team_v0_handoffs.json`

Required fields per handoff:
- `from_step`
- `to_steps`
- `required_artifacts`
- `required_sections`
- optional `typed_handoff`
  - `file`
  - `required_fields`

## Current Handoffs

### PM -> Architect
- from_step: `pm_spec`
- to_steps: `arch_design`
- required_artifacts:
  - `plan/spec.md`
  - `plan/acceptance.json`
  - `plan/milestones.md`
  - `handoff/pm_to_architect.json`
- typed_handoff:
  - file: `handoff/pm_to_architect.json`
  - required fields:
    - `from_step`
    - `to_steps`
    - `scope_summary`
    - `artifacts`
    - `acceptance.criteria`

### Architect -> Backend / Frontend / QA
- from_step: `arch_design`
- to_steps:
  - `impl_be`
  - `impl_fe`
  - `qa_verify`
- required_artifacts:
  - `plan/arch.md`
  - `risk/risk_report.json`
  - `plan/workplan.md`
  - `handoff/architect_to_impl.json`
- typed_handoff:
  - file: `handoff/architect_to_impl.json`
  - required fields:
    - `from_step`
    - `to_steps`
    - `modules`
    - `interfaces`
    - `decisions`
    - `risks`

## Runtime Validation

Current runtime hard checks:
- required artifacts must exist
- required content sections must be detectable
- typed handoff manifest must exist when configured
- typed handoff manifest required fields must be present
- typed handoff manifest must satisfy its runtime schema contract:
  - [coding_team_pm_handoff.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\coding_team_pm_handoff.schema.json)
  - [coding_team_arch_handoff.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\coding_team_arch_handoff.schema.json)
- invalid handoff blocks downstream execution

This contract intentionally does not yet introduce full semantic validation of generated design quality.
