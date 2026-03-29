# Discord Fidelity Canary Report

- generated_at: 2026-03-28T03:29:27.208Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_fidelity_suite_v1.json
- base_url: http://localhost:3000
- scenarios: 5
- fidelity_report_rate: 40%
- fidelity_warn_rate: 100%
- verdict: FAIL

## Results Per Scenario

### #1 ecommerce_product_grid (high)
- workflow: failed | total_ms: 16540 | dispatch_ms: 6490
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow,preview_mismatch] actual=none)

### #2 crm_customer_list (medium)
- workflow: failed | total_ms: 440186 | dispatch_ms: 8648
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=none)

### #3 todo_webapp_simple (low)
- workflow: failed | total_ms: 19196 | dispatch_ms: 9168
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,preview_mismatch] actual=none)

### #4 landing_page_static (low)
- workflow: succeeded | total_ms: 20018 | dispatch_ms: 9983
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,visually_incomplete,preview_mismatch] actual=preview_mismatch)

### #5 utility_script_only (medium)
- workflow: succeeded | total_ms: 17767 | dispatch_ms: 7744
- fidelity_exists: true
- fidelity_classification: preview_mismatch
- perceptual_quality: low
- fidelity_should_warn: true
- reasoning_criteria: 7
- preview_classification: preview_mismatch
- go_no_go_verdict: GO
- classification_check: PASS (expected=[demo_usable,visually_incomplete,artifact_complete_but_shallow,preview_mismatch] actual=preview_mismatch)

## Fidelity Classification Distribution
- preview_mismatch: 2

## Perceptual Quality Distribution
- low: 2

## Timing
- min_total_ms: 16540
- max_total_ms: 440186
- avg_total_ms: 102741
