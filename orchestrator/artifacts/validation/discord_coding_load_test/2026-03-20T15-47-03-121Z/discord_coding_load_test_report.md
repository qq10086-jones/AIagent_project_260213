# Discord Coding Load Test Report

- generated_at: 2026-03-21T13:26:24.763Z
- suite_file: E:\AIagent_project_260213\orchestrator\canary_inputs\discord_coding_route_queue_suite_v1.json
- total_runs: 18
- success_count: 10
- failure_count: 8
- workflow_success_count: 10
- dispatch_p50_ms: 5200
- dispatch_p95_ms: 11754
- total_p50_ms: 3744775
- total_p95_ms: 37270479
- verdict: FAIL

## Dispatch Modes
- progress_update: 18

## Workflow Statuses
- timeout: 6
- succeeded: 10
- failed: 2

## Scenario Counts
- promo_site_fast: 3
- landing_page_fast: 6
- portfolio_fast: 5
- pricing_fast: 4

## Runs
- #1 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=12666 total_ms=35940704 error=LOAD_TEST_TIMEOUT
- #2 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=12359 total_ms=35940398 error=LOAD_TEST_TIMEOUT
- #3 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=12048 total_ms=35940090 error=LOAD_TEST_TIMEOUT
- #4 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=11754 total_ms=35939798 error=LOAD_TEST_TIMEOUT
- #5 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=11461 total_ms=35939502 error=LOAD_TEST_TIMEOUT
- #6 scenario=promo_site_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=11149 total_ms=35939192 error=LOAD_TEST_TIMEOUT
- #7 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=859 total_ms=3957164 error=none
- #8 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=1159 total_ms=3484547 error=none
- #9 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=562 total_ms=3744775 error=none
- #10 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=791 total_ms=4164685 error=none
- #11 scenario=landing_page_fast class=short mode=progress_update workflow_status=failed dispatch_ms=189 total_ms=3483353 error=OpenCode authentication failed
- #12 scenario=landing_page_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=274 total_ms=3247237 error=none
- #13 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=5200 total_ms=2943161 error=none
- #14 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=5436 total_ms=3103920 error=none
- #15 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=4845 total_ms=3884741 error=none
- #16 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=8175 total_ms=3103146 error=none
- #17 scenario=portfolio_fast class=short mode=progress_update workflow_status=succeeded dispatch_ms=6404 total_ms=3673367 error=none
- #18 scenario=pricing_fast class=short mode=progress_update workflow_status=failed dispatch_ms=8223 total_ms=581198 error=OpenCode authentication failed
- #19 scenario=pricing_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=6674 total_ms=37270479 error=LOAD_TEST_TIMEOUT
- #20 scenario=pricing_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=6548 total_ms=35827448 error=LOAD_TEST_TIMEOUT
- #21 scenario=pricing_fast class=short mode=progress_update workflow_status=timeout dispatch_ms=5151 total_ms=35426242 error=LOAD_TEST_TIMEOUT
