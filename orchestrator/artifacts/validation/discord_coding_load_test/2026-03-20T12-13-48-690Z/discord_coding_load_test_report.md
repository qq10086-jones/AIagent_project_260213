# Discord Coding Load Test Report

- generated_at: 2026-03-20T14:51:55.054Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_coding_route_queue_suite_v1.json
- total_runs: 18
- success_count: 15
- failure_count: 3
- workflow_success_count: 15
- dispatch_p50_ms: 5826
- dispatch_p95_ms: 11037
- total_p50_ms: 1912171
- total_p95_ms: 4282857
- verdict: FAIL

## Dispatch Modes
- progress_update: 18

## Workflow Statuses
- succeeded: 15
- partial_failure: 1
- failed: 2

## Scenario Counts
- promo_site_fast: 3
- landing_page_fast: 6
- portfolio_fast: 5
- pricing_fast: 4

## Runs
- #1 scenario=promo_site_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=7042 total_ms=1315511 error=none
- #2 scenario=promo_site_fast class=short mode=progress_update workflow_status=partial_failure dispatch_ms=6734 total_ms=673525 error=PARTIAL_FAILURE
- #3 scenario=promo_site_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=11060 total_ms=1765990 error=none
- #4 scenario=promo_site_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=6129 total_ms=1429969 error=none
- #5 scenario=promo_site_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=5826 total_ms=1653655 error=none
- #6 scenario=promo_site_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=5525 total_ms=1547259 error=none
- #7 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=6343 total_ms=2036528 error=none
- #8 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=5679 total_ms=1833901 error=none
- #9 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=8962 total_ms=1915556 error=none
- #10 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=7388 total_ms=1903735 error=none
- #11 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=4630 total_ms=1906948 error=none
- #12 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=11037 total_ms=1912171 error=none
- #13 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=7387 total_ms=2153196 error=none
- #14 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=7425 total_ms=1927607 error=none
- #15 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=4595 total_ms=1854319 error=none
- #16 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=3752 total_ms=2188773 error=none
- #17 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=5126 total_ms=1944648 error=none
- #18 scenario=pricing_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=9023 total_ms=2184257 error=none
- #19 scenario=pricing_fast class=short mode=progress_update workflow_status=partial_failure dispatch_ms=8315 total_ms=1310887 error=PARTIAL_FAILURE
- #20 scenario=pricing_fast class=short mode=progress_update workflow_status=failed dispatch_ms=4198 total_ms=2753919 error=GLOBAL_TASK_TIMEOUT
- #21 scenario=pricing_fast class=short mode=progress_update workflow_status=failed dispatch_ms=3597 total_ms=4282857 error=GLOBAL_TASK_TIMEOUT
