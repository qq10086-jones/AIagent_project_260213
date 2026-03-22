# Discord Coding Load Test Report

- generated_at: 2026-03-22T13:54:01.006Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_coding_real_e2e_suite_v1.json
- total_runs: 3
- success_count: 2
- failure_count: 1
- workflow_success_count: 2
- dispatch_p50_ms: 9350
- dispatch_p95_ms: 10023
- total_p50_ms: 3240570
- total_p95_ms: 18830378
- verdict: FAIL

## Dispatch Modes
- progress_update: 3

## Workflow Statuses
- timeout: 1
- succeeded: 2

## Scenario Counts
- fashion_brand_site: 2
- coffee_shop_site: 1

## Runs
- #1 scenario=fashion_brand_site class=medium mode=progress_update workflow_status=timeout dispatch_ms=7336 total_ms=3609156 error=LOAD_TEST_TIMEOUT
- #2 scenario=fashion_brand_site class=medium mode=progress_update workflow_status=timeout dispatch_ms=10023 total_ms=18830378 error=LOAD_TEST_TIMEOUT
- #3 scenario=fashion_brand_site class=medium mode=progress_update workflow_status=succeeded dispatch_ms=9350 total_ms=3240570 error=none
- #4 scenario=coffee_shop_site class=medium mode=progress_update workflow_status=succeeded dispatch_ms=4592 total_ms=1018775 error=none
