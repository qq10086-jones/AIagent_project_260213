# Discord Coding Load Test Strategy

## Objective

Validate the real Discord-style coding intake path rather than only the downstream `coding.delegate` worker path.

The test target is:

- Discord-style message normalization
- `/vnext/dispatch` routing and queue admission
- workflow creation for light coding projects
- end-to-end workflow completion and artifact packaging
- failure visibility under controlled load

## Why This Test Is Split

The current system does not have true high execution parallelism for coding.

Observed constraints:

- `worker-coder` is deployed as a single service instance
- coding execution is effectively serialized at the worker layer
- workflow submission can be concurrent, but implementation throughput is close to one active coding task at a time

Because of that, one large mixed test would hide whether failures come from:

- intake and queue pressure
- workflow orchestration pressure
- coding execution backlog
- artifact finalization instability

So the load test is split into two suites.

## Suite A: Route Queue

File:

- `orchestrator/canary_inputs/discord_coding_route_queue_suite_v1.json`

Purpose:

- stress Discord-style intake
- stress router and workflow creation
- stress queue growth
- keep task body light so the signal reflects admission pressure instead of deep implementation complexity

Task profile:

- short landing pages
- short marketing sites
- small utility scripts
- very light static or script-oriented work

Recommended load:

- warmup: 3
- measured runs: 12 to 18
- concurrency: 4 to 6

Why:

- this is enough to create backlog without causing multi-hour tail latency
- at current single-worker coding throughput, higher concurrency mostly measures queue length, not useful execution capacity

Suggested gate:

- workflow success rate >= 0.70
- dispatch p95 <= 15000 ms
- no unexpected dispatch mode spikes

## Suite B: Real E2E

File:

- `orchestrator/canary_inputs/discord_coding_real_e2e_suite_v1.json`

Purpose:

- measure real end-to-end completion quality for lightweight but realistic coding requests
- validate artifact generation and QA/release closure
- surface business failures such as `ARTIFACT_INCOMPLETE`

Task profile:

- small promotional websites
- event registration pages
- restaurant reservation pages
- simple automation or formatting scripts

Recommended load:

- warmup: 2
- measured runs: 6 to 10
- concurrency: 2 to 3

Why:

- these scenarios actually traverse more of the workflow
- if average workflow service time is 5 to 8 minutes, then:
- at 6 runs the last task waits about 25 to 40 minutes
- at 10 runs the last task waits about 45 to 72 minutes
- beyond this, tail latency becomes too large for iterative QA debugging

Suggested gate:

- workflow success rate >= 0.50
- GO rate >= 0.40
- total p95 <= 3600000 ms
- failure reasons must be dominated by known business failures, not routing or infra failures

## Why Not Start With 30 Parallel

At current system quality, `30` parallel Discord coding submissions is not the right first gate.

Reason:

- if one coding worker effectively processes one workflow at a time
- and one light workflow costs 5 to 8 minutes
- then the 30th task can wait roughly:
- `(30 - 1) * 5 = 145` minutes
- `(30 - 1) * 8 = 232` minutes

This turns the test into a backlog endurance test, not a useful first-pass QA validation.

So `30` is only justified after:

- route queue suite is stable at 12 to 18 submissions
- real E2E suite is stable at 6 to 10 submissions
- artifact completeness failures are reduced
- either worker count or actual coding throughput improves

## Current Capacity Ladder

Stage 1:

- route queue: 12 runs, concurrency 4
- real E2E: 6 runs, concurrency 2

Stage 2:

- route queue: 18 runs, concurrency 6
- real E2E: 8 runs, concurrency 3

Stage 3:

- route queue: 24 runs, concurrency 6
- real E2E: 10 runs, concurrency 3

Do not advance to the next stage if the prior stage fails gate thresholds.

## Commands

Route queue:

```bash
cd orchestrator
npm run validate:discord_coding_route_queue -- --runs 12 --warmup 3 --concurrency 4 --strict false --min-workflow-success-rate 0.70 --max-dispatch-p95-ms 15000
```

Real E2E:

```bash
cd orchestrator
npm run validate:discord_coding_real_e2e -- --runs 6 --warmup 2 --concurrency 2 --strict false --min-workflow-success-rate 0.50 --min-go-rate 0.40 --max-total-p95-ms 3600000
```

## Report Outputs

Artifacts are written under:

- `orchestrator/artifacts/validation/discord_coding_load_test/<timestamp>/`

Primary files:

- `discord_coding_load_test_report.json`
- `discord_coding_load_test_report.md`

## Interpretation Priority

When reading reports, prioritize failures in this order:

1. dispatch failures
2. unexpected non-workflow modes
3. workflow creation failures
4. workflow terminal failures
5. artifact or GO/No-GO failures

This ordering keeps routing and platform regressions separate from coding-quality regressions.
