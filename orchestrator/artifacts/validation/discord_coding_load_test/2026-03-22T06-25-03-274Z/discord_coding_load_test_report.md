# Discord Coding Load Test Report

- generated_at: 2026-03-22T06:27:56.817Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_coding_real_e2e_suite_v1.json
- total_runs: 3
- success_count: 0
- failure_count: 3
- workflow_success_count: 0
- dispatch_p50_ms: 9080
- dispatch_p95_ms: 10558
- total_p50_ms: 14100
- total_p95_ms: 15574
- verdict: FAIL

## Dispatch Modes
- progress_update: 3

## Workflow Statuses
- failed: 3

## Scenario Counts
- fashion_brand_site: 2
- coffee_shop_site: 1

## Runs
- #1 scenario=fashion_brand_site class=medium mode=progress_update workflow_status=failed dispatch_ms=5739 total_ms=131108 error=OpenCode authentication failed
- #2 scenario=fashion_brand_site class=medium mode=progress_update workflow_status=failed dispatch_ms=10558 total_ms=15574 error=OpenCode authentication failed
- #3 scenario=fashion_brand_site class=medium mode=progress_update workflow_status=failed dispatch_ms=7686 total_ms=12700 error=OpenCode authentication failed
- #4 scenario=coffee_shop_site class=medium mode=progress_update workflow_status=failed dispatch_ms=9080 total_ms=14100 error=OpenCode authentication failed
