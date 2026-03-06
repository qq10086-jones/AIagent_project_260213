# Coding Team Output Validators Contract

## Scope

This contract introduces the minimum runtime validators for:
- PM output
- Architect output

These validators are required before broader role handoff validation can be considered reliable.

## PM Output Validator

Checks:
- required files exist:
  - `plan/spec.md`
  - `plan/acceptance.json`
  - `plan/milestones.md`
- required markdown headings are detectable in `plan/spec.md`:
  - scope
  - user stories
  - acceptance criteria
  - non-goals
  - artifact list
- `acceptance.json` must contain:
  - non-empty `criteria`
  - non-empty `artifacts`
  - non-empty `owner`
  - non-empty `version`

Failure codes:
- `PM_ARTIFACT_ROOT_MISSING`
- `PM_REQUIRED_FILES_MISSING`
- `PM_REQUIRED_SECTIONS_MISSING`

Runtime schema contract:
- [coding_team_pm_acceptance.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\coding_team_pm_acceptance.schema.json)

## Architect Output Validator

Checks:
- required files exist:
  - `plan/arch.md`
  - `risk/risk_report.json`
  - `plan/workplan.md`
- required markdown headings are detectable in `plan/arch.md`:
  - module breakdown
  - interfaces
  - dependency choices
  - risk notes
- `risk_report.json` must contain:
  - non-empty `risks`
  - each risk must include `title` and `mitigation`
  - non-empty `decision_log`

Failure codes:
- `ARCH_ARTIFACT_ROOT_MISSING`
- `ARCH_REQUIRED_FILES_MISSING`
- `ARCH_REQUIRED_SECTIONS_MISSING`

Runtime schema contract:
- [coding_team_arch_risk_report.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\coding_team_arch_risk_report.schema.json)

## Runtime Integration

Runtime behavior:
- `pm_spec` success triggers PM validator
- `arch_design` success triggers Architect validator
- validator failure blocks downstream workflow execution

## Non-Scope

- no UI validator
- no backend validator
- no frontend validator
- no semantic LLM-based quality scoring
