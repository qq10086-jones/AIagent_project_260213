# Discord Fidelity Canary Report

- generated_at: 2026-03-28T02:08:40.971Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_fidelity_suite_v1.json
- base_url: http://localhost:3000
- scenarios: 5
- fidelity_report_rate: 0%
- fidelity_warn_rate: n/a
- verdict: FAIL

## Results Per Scenario

### #1 ecommerce_product_grid (high)
- workflow: succeeded | total_ms: 22725 | dispatch_ms: 12690
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: GO
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=none)

### #2 crm_customer_list (medium)
- workflow: failed | total_ms: 579819 | dispatch_ms: 8321
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: n/a
- classification_check: n/a (expected=[demo_usable,domain_misaligned,artifact_complete_but_shallow] actual=none)

### #3 todo_webapp_simple (low)
- workflow: succeeded | total_ms: 24075 | dispatch_ms: 14034
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: GO
- classification_check: n/a (expected=[demo_usable,domain_misaligned] actual=none)

### #4 landing_page_static (low)
- workflow: succeeded | total_ms: 17748 | dispatch_ms: 7723
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: GO
- classification_check: n/a (expected=[demo_usable,visually_incomplete] actual=none)

### #5 utility_script_only (medium)
- workflow: succeeded | total_ms: 17763 | dispatch_ms: 7726
- fidelity_exists: false
- preview_classification: n/a
- go_no_go_verdict: GO
- classification_check: n/a (expected=[demo_usable,visually_incomplete,artifact_complete_but_shallow] actual=none)

## Fidelity Classification Distribution

## Perceptual Quality Distribution

## Timing
- min_total_ms: 17748
- max_total_ms: 579819
- avg_total_ms: 132426
