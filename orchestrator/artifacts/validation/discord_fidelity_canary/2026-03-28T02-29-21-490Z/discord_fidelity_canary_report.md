# Discord Fidelity Canary Report

- generated_at: 2026-03-28T02:39:41.092Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_fidelity_suite_v1.json
- base_url: http://localhost:3000
- scenarios: 5
- fidelity_report_rate: 80%
- fidelity_warn_rate: 100%
- verdict: FAIL

## Results Per Scenario

### #1 ecommerce_product_grid (high)
- workflow: succeeded | total_ms: 21932 | dispatch_ms: 11886
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: FAIL (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=preview_mismatch)

### #2 crm_customer_list (medium)
- workflow: failed | total_ms: 535747 | dispatch_ms: 9110
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=none)

### #3 todo_webapp_simple (low)
- workflow: succeeded | total_ms: 20186 | dispatch_ms: 10149
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: FAIL (expected=[demo_usable,domain_misaligned] actual=preview_mismatch)

### #4 landing_page_static (low)
- workflow: succeeded | total_ms: 22602 | dispatch_ms: 12567
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: FAIL (expected=[demo_usable,visually_incomplete] actual=preview_mismatch)

### #5 utility_script_only (medium)
- workflow: succeeded | total_ms: 19092 | dispatch_ms: 9060
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: FAIL (expected=[demo_usable,visually_incomplete,artifact_complete_but_shallow] actual=preview_mismatch)

## Fidelity Classification Distribution
- preview_mismatch: 4

## Perceptual Quality Distribution
- low: 4

## Timing
- min_total_ms: 19092
- max_total_ms: 535747
- avg_total_ms: 123912
