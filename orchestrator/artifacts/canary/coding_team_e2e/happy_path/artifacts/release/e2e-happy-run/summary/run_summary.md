# Run Summary

- run_id: e2e-happy-run
- workflow_run_id: 28ea15f6-62ba-47cc-9f16-a024934c2c0f
- workflow_id: coding_team_v0
- project_type: webapp_crm
- status: succeeded
- strict_canary_verdict: PASS
- preview_validation: PREVIEW_MISSING
- preview_warning: YES
- product_fidelity: ARTIFACT_COMPLETE_BUT_SHALLOW
- product_fidelity_warning: YES
- generated_at: 2026-03-29T15:25:08.652Z

## Context Budget
- total_steps: 8
- ok: 8
- warning: 0
- overflow_risk: 0
- missing: 0

## Context Artifacts
- context_packets: 3
- repo_maps: 3

## Coding Execution Evidence
- verification_checked: 0
- verification_passed: 0
- verification_failed: 0
- retry_enabled_steps: 0
- retry_attempted_steps: 0
- failure_memory_entries: 0

## Preview Validation
- classification: preview_missing
- should_warn: true
- report_path: artifacts/release/e2e-happy-run/qa/preview_validation_report.json

## Product Fidelity
- classification: artifact_complete_but_shallow
- should_warn: true
- perceptual_quality_score: low
- fidelity_gate_mode: warning
- report_path: artifacts/release/e2e-happy-run/qa/product_fidelity_report.json

## Execution Config
- execution_lane: not_configured
- provider: not_configured
- model: not_configured

## Steps
- [OK] 0:pm_spec (coding.delegate)
- [OK] 1:arch_design (coding.delegate)
- [OK] 2:impl_be (coding.delegate)
- [OK] 3:impl_fe (coding.delegate)
- [OK] 4:smoke_test (coding.execute)
- [OK] 5:qa_verify (coding.delegate)
- [OK] 6:release_pack (coding.delegate)
- [OK] 7:deploy_preview (ops.deploy_preview)
## Artifact Quality
- contract_found: true
- required: 3
- present: 3
- missing: 0
- invalid: 0
