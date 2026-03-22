# Discord Coding Load Test Report

- generated_at: 2026-03-21T15:51:36.149Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_coding_route_queue_suite_v1.json
- total_runs: 10
- success_count: 0
- failure_count: 10
- workflow_success_count: 0
- dispatch_p50_ms: 9662
- dispatch_p95_ms: 16006
- total_p50_ms: 1810525
- total_p95_ms: 1813979
- verdict: FAIL

## Dispatch Modes
- progress_update: 10

## Workflow Statuses
- timeout: 9
- failed: 1

## Scenario Counts
- promo_site_fast: 4
- landing_page_fast: 6

## Runs
- #1 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=9820 total_ms=1814591 error=LOAD_TEST_TIMEOUT
- #2 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=9500 total_ms=1814274 error=LOAD_TEST_TIMEOUT
- #3 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=9207 total_ms=1813979 error=LOAD_TEST_TIMEOUT
- #4 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=10503 total_ms=1813472 error=LOAD_TEST_TIMEOUT
- #5 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=10795 total_ms=1813766 error=LOAD_TEST_TIMEOUT
- #6 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=6074 total_ms=1809052 error=LOAD_TEST_TIMEOUT
- #7 scenario=landing_page_fast class=short mode=progress_update workflow_status=failed dispatch_ms=16006 total_ms=1039921 error=OpenCode authentication failed
- #8 scenario=landing_page_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=10393 total_ms=1812083 error=LOAD_TEST_TIMEOUT
- #9 scenario=landing_page_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=10685 total_ms=1812376 error=LOAD_TEST_TIMEOUT
- #10 scenario=landing_page_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=9662 total_ms=1810525 error=LOAD_TEST_TIMEOUT
- #11 scenario=landing_page_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=5011 total_ms=1808760 error=LOAD_TEST_TIMEOUT
- #12 scenario=landing_page_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=5304 total_ms=1809053 error=LOAD_TEST_TIMEOUT
