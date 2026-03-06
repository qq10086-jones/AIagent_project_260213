# OpenClaw Nexus Progress Update (AGENTS.md Style)

## Snapshot
- Date: `2026-03-06 14:39:51`
- Sprint context: `vnext_brain_first + coding_team_ws06_contract_hardening`
- North Star: `Discord/HTTP input -> Brain Router -> TaskEnvelope -> OpenClaw dispatch -> Coding Team contract-driven execution`
- Current status: upstream routing/contracts are in place; `WS-06` is in progress with runtime-enforced PM/Architect validators and typed handoff manifests.

## What Was Completed Today
- P0 upstream boundary closure completed to current-stage acceptance level:
  - `Discord request`, `TaskEnvelope`, `Brain Router output`, `route/dispatch success/error` contracts added.
  - `/vnext/route` and `/vnext/dispatch` now use normalized response contracts and runtime shape checks.
  - chat bypass and coding dispatch entry are active.
- Prompt Script Registry minimum slice delivered:
  - loader + schema added.
  - `pm.design_doc.v1`
  - `architect.system_spec.v1`
  - `qa.test_plan.v1`
  - `coding_team_v0` steps are now bound to prompt script ids.
- Agent Contract Layer minimum slice delivered:
  - agent contract schema added.
  - `pm/architect/ui/backend/frontend/qa` agent specs added.
  - prompt script to agent binding now validates at runtime startup.
- Runtime contract hardening delivered:
  - startup now refuses invalid prompt-script/agent bindings.
  - workflow runtime now hard-fails on missing or mismatched `prompt_script_id`.
- `WS-06` minimum handoff and validator chain delivered:
  - coding-team handoff registry added.
  - PM and Architect output validators added and wired into workflow runtime.
  - validators upgraded from weak text checks to:
    - markdown heading checks
    - JSON artifact schema checks
  - handoff upgraded from plain artifact/content presence to:
    - typed handoff manifests
    - typed handoff schema validation

## Mapping To Design / Task List
- Design document alignment:
  - Brain-first routing: implemented.
  - TaskEnvelope/contract boundary: implemented.
  - Agent as contract, not persona: minimum implementation delivered.
  - Coding Team contract-driven orchestration: in progress, now at PM/Architect validator + handoff layer.
- Task list alignment:
  - P0 Brain Router / TaskEnvelope / direct chat bypass / OpenClaw boundary cleanup: done at current-stage level.
  - `WS-05 Prompt Script Registry`: minimum slice done.
  - `WS-06 Role handoff contracts`: minimum slice done and hardened.
  - `WS-06 PM output validator`: minimum slice done and hardened.
  - `WS-06 Architect output validator`: minimum slice done and hardened.
- Governance alignment:
  - followed upstream-first sequencing.
  - did not expand into UI/backend/frontend runtime prompts, dashboard, quant, or memory layers.

## Runtime Evidence (Today)
- Upstream contract canary:
  - `node scripts/canary_brain_router.js`
  - latest known result: `passed_cases=16`, `failed_cases=0`
- Prompt registry canary:
  - `node scripts/canary_prompt_registry.js`
- Agent contract layer canary:
  - `node scripts/canary_agent_contract_layer.js`
- Runtime hardening canary:
  - `node scripts/canary_runtime_contract_hardening.js`
- Coding team validator canary:
  - `node scripts/canary_coding_team_output_validators.js`
- Coding team handoff canary:
  - `node scripts/canary_coding_team_handoff.js`
- Startup import check:
  - `import('./src/index.js') -> ok`
  - current mode remains API-only because no valid `DISCORD_TOKEN` is present in this environment.

## Current Gaps
- `WS-06` is not fully signed off yet.
- Real workflow-level failure-path automation is still weaker than contract/canary coverage.
- Markdown outputs are still validated via structural heuristics plus schema-backed JSON sidecars, not full typed document schemas.
- Live Discord end-to-end validation is not available in current environment.

## Next Priority (Mainline)
- Continue `WS-06` without widening scope:
  - add more workflow-level failure-path automation around PM/Architect validator and typed handoff enforcement.
  - then reassess whether `WS-06` reaches sign-off level under QA review.

## Changed Files (Today)
- `orchestrator/src/vnext/input_normalizer.js`
- `orchestrator/src/vnext/task_envelope.js`
- `orchestrator/src/vnext/brain_router.js`
- `orchestrator/src/vnext/response_protocol.js`
- `orchestrator/src/vnext/coder_directive.js`
- `orchestrator/src/vnext/route_contract.js`
- `orchestrator/src/vnext/dispatch_contract.js`
- `orchestrator/src/vnext/contract_validator.js`
- `orchestrator/src/prompt_script_registry.js`
- `orchestrator/src/agent_contract_registry.js`
- `orchestrator/src/handoff_contract_registry.js`
- `orchestrator/src/coding_team_validators.js`
- `orchestrator/src/coding_team_handoff_validators.js`
- `orchestrator/src/schema_lite_validator.js`
- `orchestrator/src/workflow_engine.js`
- `orchestrator/src/index.js`
- `orchestrator/contracts/*.json`
- `orchestrator/configs/contracts/coding_team_v0_handoffs.json`
- `configs/agents/*.json`
- `configs/prompt_scripts/registry.json`
- `configs/registry/capability_registry.json`
- `orchestrator/scripts/canary_*.js`
- `docs/01_design/system/260306/*.md`
- `docs/03_feature_development/progress_reports/progress_20260306_143951_ws06_contract_hardening.md`

## Tracking Notes
- This update stays within the design document, engineering task list, governance file, and the currently applicable subset of the MetaGPT patch.
- Existing unrelated git/worktree state was not reverted.
