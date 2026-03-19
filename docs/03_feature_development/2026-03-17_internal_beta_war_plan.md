# Internal Beta War Plan

## Goal

Push the project to a controlled internal-beta-ready state without expanding scope.

This document is the single execution contract for PM, architecture, engineering, and QA.

## Scope Freeze

Internal beta only covers the following path:

- Discord-style input
- lightweight coding requests only
- `coding_team_v0`
- local preview only
- controlled cloud coding lane only

The following are out of scope for internal beta admission:

- Render deployment or public preview hosting
- heavyweight full-stack builds
- local Ollama dependency on low-resource test machines
- additional product features unrelated to beta admission

## Frozen Runtime Profile

Use the following runtime profile for internal beta execution:

- coder provider: `opencode`
- coder model: `dashscope/qwen-plus-2025-04-28`
- execution lane: `stable_cloud_lane`
- local LLM fallback: disabled on this machine
- preview mode: localhost preview

Configuration sources that must agree:

- `configs/runtime/runtime_defaults.json`
- `infra/.env`
- `orchestrator` runtime defaults
- `worker-coder` provider preflight and adapter auth validation

Current blocker under this frozen profile:

- live Discord smoke still fails at `pm_spec`
- error class: `E_AUTH_FAILED`
- upstream stderr: `Incorrect API key provided`
- evidence:
  - `orchestrator/artifacts/validation/discord_coding_load_test/2026-03-17T10-02-19-137Z/discord_coding_load_test_report.json`
  - `artifacts/runs/388ff6a6-15ae-4eb1-ba1d-b3ac5b580ac6/memory/coding_failure_latest.json`

## P0 Tasks

1. Keep beta scope frozen. Do not add features.
2. Keep cloud coding defaults pinned to the frozen runtime profile.
3. Keep local fallback disabled on non-Ollama test machines.
4. Keep local preview as the only preview target for beta.
5. Require every bug fix to be followed by the affected smoke or load test.

## Mandatory Test Gates

### Gate A: Route Queue

Command:

```bash
cd orchestrator
npm.cmd run validate:discord_coding_route_queue -- --runs 18 --warmup 3 --concurrency 6 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-dispatch-p95-ms 15000
```

Pass conditions:

- workflow success rate = 1.0
- GO rate = 1.0
- dispatch p95 <= 15000 ms
- no unexpected dispatch mode

### Gate B: Real E2E

Command:

```bash
cd orchestrator
npm.cmd run validate:discord_coding_real_e2e -- --runs 8 --warmup 2 --concurrency 3 --strict false --min-workflow-success-rate 1.0 --min-go-rate 1.0 --max-total-p95-ms 3600000
```

Pass conditions:

- workflow success rate = 1.0
- GO rate = 1.0
- no preview-link failures
- no provider-auth false positives
- no artifact-pack false negatives

## Role Assignments

### PM

- freeze scope
- reject new feature requests
- publish supported and unsupported beta capabilities

### Architecture

- keep the frozen runtime profile consistent
- remove environment-specific ambiguity
- keep cloud-only behavior explicit on this machine

### Engineering

- fix provider, workflow, artifact, and preview blockers only
- do not start unrelated refactors
- keep commits small and attributable to a gate failure or a freeze item

### QA

- run Gate A and Gate B
- record command, config snapshot, result, and blocker for every run
- reject manual explanations in place of passing reports

## Stop Rules

Stop and fix before advancing if any of the following appears:

- provider auth mismatch
- workflow terminal failure
- GO or artifact false negative
- preview URL unreachable
- route queue latency breach

## Exit Criteria

Internal beta can be declared ready only if:

- Gate A passes
- Gate B passes
- frozen runtime profile remains unchanged during the final passing runs
- release note for beta scope is written
- no unresolved P0 blocker remains
