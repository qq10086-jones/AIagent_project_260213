# Discord Fidelity Canary Report

- generated_at: 2026-03-28T04:11:25.868Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_fidelity_suite_v1.json
- base_url: http://localhost:3000
- scenarios: 5
- fidelity_report_rate: 60%
- fidelity_warn_rate: 100%
- verdict: FAIL

## Results Per Scenario

### #1 ecommerce_product_grid (high)
- workflow: succeeded | total_ms: 18262 | dispatch_ms: 8233
- fidelity_exists: true
- fidelity_classification: artifact_complete_but_shallow
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_missing
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow,preview_mismatch] actual=artifact_complete_but_shallow)

### #2 crm_customer_list (medium)
- workflow: failed | total_ms: 288191 | dispatch_ms: 7493
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=none)

### #3 todo_webapp_simple (low)
- workflow: partial_failure | total_ms: 19275 | dispatch_ms: 9229
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,preview_mismatch] actual=none)

### #4 landing_page_static (low)
- workflow: succeeded | total_ms: 20224 | dispatch_ms: 10199
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,visually_incomplete,preview_mismatch] actual=preview_mismatch)

### #5 utility_script_only (medium)
- workflow: succeeded | total_ms: 19114 | dispatch_ms: 9082
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,visually_incomplete,artifact_complete_but_shallow,preview_mismatch] actual=preview_mismatch)

## Fidelity Classification Distribution
- artifact_complete_but_shallow: 1
- preview_mismatch: 2

## Perceptual Quality Distribution
- low: 3

## Timing
- min_total_ms: 18262
- max_total_ms: 288191
- avg_total_ms: 73013
