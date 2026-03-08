# Milestone 3 Remediation Checklist
## Date
2026-03-07

## Purpose
Align the current codebase with the active constraints in `docs/01_design/system/260307/`.

## Source Constraints
- `OpenClaw_Nexus_Design_Document_v2.md`
- `OpenClaw_Execution_Governance_Scope_Control_v2.md`
- `Architect_Engineer_Role_Contract.md`
- `OpenClaw_Nexus_Engineering_Task_List_M3.md`
- `Orchestrator_Layer_Map_Draft.md`
- `Route_Audit.md`

## Current QA Judgment
- `M2` functional chain: closed with prior live evidence
- `M3` structural hardening: closed
- Remaining work: none inside M3 scope; move to next milestone definition

## Priority 0: Correct The Source Of Truth
- `P0-01` Update progress docs to reflect real status of `WS-11-03`, `WS-11-04`, `WS-15-03`
  - Current status: `done`
  - Acceptance: latest progress snapshot matches actual code

## Priority 1: Restore Layer Compliance
- `P1-01` `WS-11-02` Extract Discord adapter from `src/index.js`
  - Current status: `done`
  - Acceptance:
  - `src/adapters/discord_gateway.js` exists
  - `src/index.js` does not import `discord.js`
  - Discord event registration is not defined inline in `src/index.js`
- `P1-02` `WS-11-03` remove raw SQL from Layer 1 and Layer 2
  - Current status: `done`
  - Acceptance:
  - `src/index.js` contains zero `pool.query`
  - `src/vnext/*.js` contains zero raw SQL
  - repository coverage exists for moved queries
- `P1-03` create shared infra connection boundary
  - Current status: `done`
  - Acceptance:
  - DB/Redis/S3 initialization moved out of `src/index.js`

## Priority 2: Restore Complexity Budget
- `P2-01` `WS-11-04` actually decompose `src/workflow_engine.js`
  - Current status: `done`
  - Acceptance:
  - `src/workflow_engine.js <= 600` lines
  - main execution loop/state/artifact audit delegated to `src/domain/`
- `P2-02` `WS-11-05` reduce `src/index.js` to thin router
  - Current status: `done`
  - Acceptance:
  - `src/index.js <= 800` lines
  - no inline Discord business flow
  - no inline LLM calls

## Priority 3: Close Contract Gaps
- `P3-01` Architect output validator should verify content quality, not file existence only
  - Current status: `done`
  - Acceptance:
  - `plan/arch.md` required headings checked
  - `plan/interfaces.md` contains at least one interface heading
  - ADR presence enforced
- `P3-02` verify memory integration status and align docs
  - Current status: `done`
  - Acceptance:
  - docs state that `arch_design` prompt injection is already wired
- `WS-12-04` Architect canary test with real artifact check
  - Current status: `done`
  - Acceptance:
  - required artifacts checked
  - handoff schema checked
  - non-empty `decisions` checked
  - failure cases include missing and empty `plan/interfaces.md`

## Priority 4: Verification Recovery
- `P4-01` re-run orchestrator tests in an environment that permits Node child process spawn
  - Current status: `done`
  - Acceptance:
  - full `cmd /c npm --prefix orchestrator test` result captured: `32/32 pass`
- `P4-02` re-run live validators after `WS-11` changes
  - Current status: `done`
  - Acceptance:
  - `cmd /c npm --prefix orchestrator run validate:live_vnext_runtime` pass
  - `cmd /c npm --prefix orchestrator run validate:live_workflow_runtime` pass

## Remaining Closeout Work
1. M3 closed on `2026-03-07`
2. Use `progress_20260307_235000_m3_final_closure.md` as the final closeout record
