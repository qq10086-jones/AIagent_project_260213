# OpenClaw Nexus Progress Report
## WS-16 Closure + WS-17-01 Backend Step Alignment

- Date: `2026-03-08`
- Phase: `Milestone 4 — LLM Routing Layer + Coding Team Execution Chain`
- Session: `WS-16 complete, WS-17-01 backend step aligned to M4 contract`

---

## Summary

This session completed the WS-16 workstream and advanced WS-17-01.

Completed in this session:
- `WS-16-02` LLM Dispatcher implementation
- `WS-16-03` Prompt registry migration from `model` to `llm_role`
- `WS-16-04` Execution-path call boundary cleanup
- `WS-16-05` Dispatcher canary
- `WS-16-06` Brain Router P-06 unknown intent confirmation rule
- `WS-17-01` Backend step contract alignment to M4 file-output format

---

## WS-16 Status

### WS-16-02 — DONE

Created:
- `orchestrator/src/vnext/llm_dispatcher.js`

Implemented:
- role -> provider/model resolution from `llm_role_policy.json`
- transport retry with exponential backoff
- local model fallback to `secondary_model`
- typed errors for unknown role / unknown provider / dispatch failure
- `validateProviders()` startup health check

### WS-16-03 — DONE

Updated:
- `orchestrator/configs/prompt_scripts/registry.json`
- `configs/prompt_scripts/registry.json`
- `orchestrator/src/prompt_script_registry.js`
- `orchestrator/contracts/prompt_script_registry.schema.json`

Result:
- deprecated `model` field removed from prompt script entries
- `llm_role` is now required

### WS-16-04 — DONE

Result:
- direct `callQwenChat()` / `callLocalOllamaChat()` usage now remains only in:
  - `orchestrator/src/vnext/llm_dispatcher.js`
  - `orchestrator/src/vnext/chat_entrypoint.js`

### WS-16-05 — DONE

Created:
- `orchestrator/scripts/canary_llm_dispatcher.js`

Artifact:
- `orchestrator/artifacts/canary/llm_dispatcher/llm_dispatcher_canary.json`

### WS-16-06 — DONE

Updated:
- `orchestrator/src/vnext/brain_router_policy.js`
- `orchestrator/src/vnext/brain_router.js`

Result:
- if intent remains `unknown` and execution cues are present, router returns `clarification_required`
- `/coder` override remains higher priority

---

## WS-17-01 Status

### Backend Step — DONE (contract alignment)

Updated:
- `configs/registry/capability_registry.json`
- `orchestrator/configs/prompt_scripts/registry.json`
- `configs/prompt_scripts/registry.json`
- `orchestrator/configs/contracts/coding_team_v0_handoffs.json`
- `orchestrator/src/domain/workflow_state.js`
- `orchestrator/src/coding_execution_adapters.js`
- `orchestrator/src/domain/workflow_step_validator.js`
- `orchestrator/src/coding_team_handoff_validators.js`
- `worker-coder/coding_service.js`

New backend output contract:
- `impl/be_changes/`
- `impl/be_notes.md`
- `handoff/be_to_fe.json`

New workflow binding:
- `impl_be -> backend.impl.v1`

Validation added:
- backend step now fails if:
  - `impl/be_changes/` is missing or empty
  - `impl/be_notes.md` is missing
  - `handoff/be_to_fe.json` is missing

---

## Verification

Passed:
- `node --check orchestrator/src/vnext/llm_dispatcher.js`
- `node --check orchestrator/src/vnext/brain_router_policy.js`
- `node --check orchestrator/src/index.js`
- `node scripts/canary_llm_dispatcher.js`
- `node scripts/canary_prompt_registry.js`
- `node scripts/canary_agent_contract_layer.js`
- `node scripts/canary_backend_execution_adapter.js`
- `node scripts/canary_tool_adapter_interface.js`
- `node --test --experimental-test-isolation=none test/llm_dispatcher.test.js`
- `node --test --experimental-test-isolation=none test/brain_router.integration.test.js`
- `node --test --experimental-test-isolation=none test/vnext.test.js`
- `node --test --experimental-test-isolation=none test/workflow_step_validator_backend.test.js`

Sandbox limitation:
- `cmd /c npm --prefix orchestrator test` is not reliable in the current sandbox because Node test runner subprocess spawning returns `EPERM`
- targeted tests were run with `--experimental-test-isolation=none`

---

## Next Allowed Work

- `WS-17-02` Frontend Implementation Step
- then `WS-17-03` QA Verify Step
- then `WS-17-04` Release Pack Step
- then `WS-17-05` Coding Team E2E Canary

---

## Source Of Truth

- Active design: `docs/01_design/system/260308/`
- Latest snapshot: `docs/03_feature_development/PROGRESS_LATEST.md`
