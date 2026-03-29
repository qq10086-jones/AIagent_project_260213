# Discord Fidelity Canary Report

- generated_at: 2026-03-28T04:42:36.924Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_fidelity_suite_v1.json
- base_url: http://localhost:3000
- scenarios: 5
- fidelity_report_rate: 80%
- fidelity_warn_rate: 100%
- verdict: FAIL

## Results Per Scenario

### #1 ecommerce_product_grid (high)
- workflow: succeeded | total_ms: 18254 | dispatch_ms: 8233
- fidelity_exists: true
- fidelity_classification: artifact_complete_but_shallow
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_missing
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow,preview_mismatch] actual=artifact_complete_but_shallow)

### #2 crm_customer_list (medium)
- workflow: failed | total_ms: 483968 | dispatch_ms: 7678
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=none)

### #3 todo_webapp_simple (low)
- workflow: succeeded | total_ms: 22176 | dispatch_ms: 12144
- fidelity_exists: true
- fidelity_classification: artifact_complete_but_shallow
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_missing
- go_no_go_verdict: GO
- classification_check: FAIL (expected=[demo_usable,domain_misaligned,preview_mismatch] actual=artifact_complete_but_shallow)

### #4 landing_page_static (low)
- workflow: succeeded | total_ms: 18463 | dispatch_ms: 8435
- fidelity_exists: true
- fidelity_classification: artifact_complete_but_shallow
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_missing
- go_no_go_verdict: GO
- classification_check: FAIL (expected=[demo_usable,visually_incomplete,preview_mismatch] actual=artifact_complete_but_shallow)

### #5 utility_script_only (medium)
- workflow: succeeded | total_ms: 17450 | dispatch_ms: 7409
- fidelity_exists: true
- fidelity_classification: artifact_complete_but_shallow
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_missing
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,visually_incomplete,artifact_complete_but_shallow,preview_mismatch] actual=artifact_complete_but_shallow)

## Fidelity Classification Distribution
- artifact_complete_but_shallow: 4

## Perceptual Quality Distribution
- low: 4

## Timing
- min_total_ms: 17450
- max_total_ms: 483968
- avg_total_ms: 112062
