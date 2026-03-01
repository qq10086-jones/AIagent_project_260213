# Coding Agent Design (Latest)

## Date
2026-03-01

## Source Baseline
This version is aligned with:
- `docs/90_archive/legacy_workspace/AIagent_project/Project_OpenClaw_Nexus_v1_2_3_MAS.md`
- `docs/90_archive/legacy_workspace/AIagent_project/OpenClaw_Nexus_v1_2_7_Execution_Strategy_Patch_r1.md`

The MAS baseline contributes three principles that are now mandatory for Coder:
- facts are evidence-backed
- all actions are traceable by `run_id`
- execution has explicit governance (approval, timeout, cost/log ledger)

## Positioning
Coder is the project core, but it is not a new Codex CLI.
Nexus/OpenClaw should orchestrate existing coding software, not re-implement it.

The architecture target is:
- OpenClaw/Nexus = control plane (routing, approval, audit, replay)
- Codex/Claude Code/local-compatible coding tool = execution plane (actual code changes)

## Problem Statement
- Existing legacy path (`coding.patch` + `coding.execute`) is self-built patch execution.
- Your required path is delegation-first: `/coder:` should directly call real coding software.
- Prior failure symptom (`SYSTEM Analysis Report`) came from coder delegation failure and fallback/render path mismatch, not from your intent.

## Current Implementation Status (as of this doc date)
- Discord route:
  - `/coder:` enqueues `coding.delegate`.
  - `/approve:` and `/reject:` are available for task gating feedback in Discord.
- Worker:
  - `worker-coder` has `adapters/codex_adapter.js`.
  - `coding.delegate` is registered and wired.
  - Codex CLI is installed in worker image.
- Auth:
  - requires `OPENAI_API_KEY` or `/root/.codex/auth.json`.
- Runtime:
  - codex invocation uses non-interactive `exec`.
  - sandbox is set to writable mode for workspace execution.

## Scope
In scope:
- `/coder:` as standard command prefix for coding tasks.
- `coding.delegate` as primary tool.
- provider adapter abstraction (`codex` first, then `claude_code`, then others).
- approval and evidence trace in Discord + DB + artifacts.

Out of scope:
- rebuilding full Codex/Claude product UX.
- training a new coding foundation model.

## Canonical Flow
1. User sends `/coder: <task>`.
2. Orchestrator creates `run_id`, `idempotency_key`, enqueues `coding.delegate`.
3. Policy engine decides direct run or `waiting_approval`.
4. Approved task is consumed by `worker-coder`.
5. Adapter invokes provider CLI in workspace.
6. Worker collects outputs:
   - changed files / diff stats
   - stdout/stderr logs
   - test result
   - artifacts
7. Orchestrator sends result summary to Discord and stores replay data.

## Tool Contract
`tool_name`: `coding.delegate`

Required payload:
- `task_prompt`

Recommended payload:
- `provider`: `auto|codex|claude_code`
- `model`
- `workspace_subdir`
- `max_runtime_s`
- `test_command` (allowlisted)

Required output:
- `ok`
- `provider_used`
- `summary`
- `files_changed`
- `diff_stats`
- `test_result`
- `artifacts`
- `diagnostics.error_code` (if failed)

## Provider Strategy
Phase default:
- `auto -> codex` (Codex as adapter core)

Next:
- add `claude_code` adapter with same normalized schema
- then add more skills (`/ui:`, `/db:`) using the same delegation interface

Rule:
- all providers must be behind adapter contract
- orchestrator never depends on provider-specific output text

## Model Strategy
- If you want official Codex capability: cloud auth is effectively required.
- If you want local-first: you can still run delegation architecture with local models, but you need a compatible executable/endpoint adapter.
- Therefore cloud is not mandatory for architecture, but may be mandatory for a specific provider capability.

## Governance and Approval
- `coding.delegate` is high-risk.
- In production policy: approval-by-default for write/exec/network sensitive tasks.
- Approval loop can stay fully in Discord (`/approve:` and `/reject:`).
- Task result text must explicitly indicate:
  - actual execution status (`succeeded/failed/waiting_approval`)
  - artifact path or workspace path
  - `run_id` and `task_id`

## Replay and Evidence Contract
Each run should provide replayable evidence package:
- `run_id`, `task_id`, provider/model used
- full timeline events
- diff artifact
- raw stdout/stderr logs
- final decision/result message

This follows MAS verifiability requirements and prevents "completed but cannot locate output" ambiguity.

## Error Taxonomy (minimum)
- `E_PROVIDER_UNAVAILABLE` (binary missing/auth missing)
- `E_TIMEOUT`
- `E_POLICY_DENY`
- `E_APPLY_FAILED`
- `E_TEST_FAILED`
- `E_INTERNAL`

Discord-facing message should map 1:1 from this taxonomy; do not downgrade into generic quant report template.

## Backward Compatibility
- Keep `coding.patch` and `coding.execute` as fallback tools.
- Fallback should only happen when:
  - provider unavailable/policy denied
  - and rollback/clean state is guaranteed

## Acceptance Criteria
- `/coder:` can create/modify files in target workspace through delegated provider.
- user can approve/reject within Discord.
- success reply contains where to find result (workspace path/artifact path + `run_id`).
- run is replayable with evidence and logs.
- quant flow remains unaffected.

## Roadmap
1. Phase A (now): Codex-first adapter stable in real chain.
2. Phase B: unify result renderer to avoid quant-template fallback.
3. Phase C: add second provider adapter (`claude_code`).
4. Phase D: skill expansion (`/ui:`, `/db:`) reusing same control plane.
