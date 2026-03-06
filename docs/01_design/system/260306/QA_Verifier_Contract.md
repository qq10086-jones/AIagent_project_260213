# QA Verifier Contract

## Scope

This contract defines the minimum `WS-06-07` QA verifier slice for `qa_verify`.

Current phase scope:
- static artifact contract validation
- no real test runner orchestration in this task
- no release-pack redesign in this task

## Required QA Artifacts

- `tests/test_plan.md`
- `qa/smoke_report.md`
- `qa/verification.json`

## Runtime Contract

Schema:
- [qa_verification.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\qa_verification.schema.json)

Current hard checks:
- `test_plan.md` must include headings:
  - `test plan`
  - `verification steps`
  - `release checklist`
- `smoke_report.md` must include headings:
  - `smoke report`
  - `executed checks`
  - `result summary`
- `verification.json` must be schema-valid
- upstream implementation delta must exist for `qa_verify`

## Failure Codes

- `QA_ARTIFACT_ROOT_MISSING`
- `QA_REQUIRED_FILES_MISSING`
- `QA_VERIFICATION_INVALID`
- `STEP_QA_NO_IMPL_DELTA`
- `STEP_QA_EVIDENCE_MISSING`

## Non-Scope

- no browser/UI evidence collection
- no real test execution hook in this task
- no go/no-go redesign in this task
