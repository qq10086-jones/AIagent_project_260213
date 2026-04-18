# OpenClaw Nexus — Pause State

**Paused on**: 2026-04-17
**Last active version**: v3.7.3 (B+C paths implemented, not verified)
**Git tag**: `v3.7.3-paused`
**Reason**: Owner transitioning focus to using Hermes Agent (Nous Research) and operating the existing OpenClaw Nexus stack as-user. Not abandoned.

---

## 1. State in one sentence

Nexus is runnable end-to-end on `stable_gemma4_lane` (gemma4:26b). Webapp-CRM acceptance sits at **~7/10 algorithmic / 3/10 user-terminal**. v3.7.3 introduced B+C prompt paths but the follow-up E2E run was inconclusive because the orchestrator container was not restarted and a downstream step timed out.

## 2. How to resume in one command

```bash
# Check the tag and read the two docs at root:
git checkout v3.7.3-paused
cat NEXUS_PAUSE_STATE.md ROADMAP_ON_HOLD.md

# Smoke the stack:
docker compose restart orchestrator   # B3 injection won't take effect without this
node orchestrator/scripts/run_smoke_test.mjs
```

## 3. Feature matrix (what's GA / experimental / half-done)

| Status | Feature | Notes |
|---|---|---|
| **GA** | 4-layer architecture (Transport / Service / Domain / Infra) | stable, unchanged since v3.2 |
| **GA** | Workflow state machine + 8-step coding_team_v0 pipeline | 306/313 tests pass |
| **GA** | OpenCode adapter + Ollama gemma4:26b lane | `stable_gemma4_lane` default |
| **GA** | Surgical patch / refinement re-entry / context resolver | all enabled in config |
| **GA** | Domain acceptance packs (crm/ecommerce/document_release) | fidelity_gate = blocking |
| **GA** | Lane escalation chain `stable_gemma4_lane → stable_cloud_lane` | enabled, but policy is "don't use cloud" (MiniMax not renewed) |
| **Experimental** | v3.7 static_audit gate (xss / class_injection / delete_semantics / be_contract_checker) | `static_audit_mode: dry_run` — does NOT block pipeline |
| **Experimental** | v3.6 FE decomposition (skeleton + modules, 2-pass) | works, replaced single-pass impl_fe |
| **Half-done** | v3.7.3 B — `webapp_crm` MANDATORY prompt injection | code committed in `8a11cf3`; **not verified in E2E** (container not restarted, impl_be did not see MANDATORY block in last run) |
| **Half-done** | v3.7.3 C — inject `plan/spec.md` into impl steps | verified in last run's prompt, but upstream PM returns scaffold so spec has no keywords for C to amplify |
| **Not started** | static_audit → blocking mode + retry feedback loop | design notes in ROADMAP_ON_HOLD.md |
| **Known broken** | `impl_fe_skeleton` OpenCode timeout (first observed 2026-04-13) | reproducibility unknown |
| **Known limitation** | PM step returns scaffold template, not a real LLM-generated spec | root cause persists across v3.6 → v3.7.3 |

## 4. Known issues (don't re-discover these)

### Blocking (prevents score > 7/10)
1. **PM step not calling LLM** — returns a scaffold template, `spec.md` is ~45 lines, goal truncated mid-word. Downstream arch/impl never see the v3.7.2 keywords (activity_log / detail view / quick actions), so C path can't help. **This is the single highest-impact unsolved root-cause.**
2. **B3 injection requires container restart** — `workflow_state.js:buildStepPrompt` change in `8a11cf3` is loaded only after `docker compose restart orchestrator`. The 2026-04-13 E2E ran against stale code and is therefore not a valid test of B3.
3. **impl_fe_skeleton timeout** — seen once in run `f7f03662`, step 4/10. Root cause unknown; BE succeeded on same model so not a model issue. Suspect OpenCode adapter or prompt length.

### Non-blocking pre-existing (7 test failures, not introduced by B+C)
- registry validator
- workflow integration tests (3)
- arch interface contract regex
- context packet
- smoke_test run_acceptance_test.mjs regex

### Known scanner findings that dry_run won't block
- BE DELETE endpoints missing 404 check ×2
- BE POST missing numeric validation ×1 (e.g. `!amount` accepts `"abc"`)

## 5. Config snapshot (do not change without reading ROADMAP)

```yaml
master_enabled: true
dynamic_routing_enabled: true
router_mode: dynamic_routing_enforced
fidelity_gate_mode: blocking
static_audit_mode: dry_run                 # switch to "blocking" is in ROADMAP
execution_lane_default: stable_gemma4_lane
lane_escalation_enabled: true
lane_escalation_chain: [stable_gemma4_lane, stable_cloud_lane]
project_planner_enabled: false             # kept off per 2026-04-11 decision
surgical_patch_enabled: true
refinement_reentry_enabled: true
context_resolver_enabled: true
running_timeout_sec: 1800
wall_clock_timeout_s_default: 900
stream_batch_size: 1
```

## 6. Files touched in v3.7.3 (B+C)

All committed in `8a11cf3 coder(v3.7.3): B+C paths artifacts + v3.5-v3.7 progress docs`.

- `orchestrator/src/domain/workflow_state.js` — webapp_crm MANDATORY injection in `buildStepPrompt`
- `orchestrator/src/domain/workflow_step_builder.js:1008` — impl_* inject `plan/spec.md` (6K truncation)
- `configs/prompt_scripts/registry.json` + `orchestrator/configs/prompt_scripts/registry.json` — frontend.impl.v1/v2 spec.md authoritative-source instruction
- `worker-coder/artifact_scaffold.js` — CRM BE (activity_log, recordActivity, DELETE, /api/activity, /api/dashboard/stats) + FE (quick actions, activity feed, detail view, delete button)
- `worker-coder/tests/artifact_scaffold.test.js` — DELETE assertion flipped to `match`, /api/activity added

## 7. Two registries must stay in sync

`configs/prompt_scripts/registry.json` and `orchestrator/configs/prompt_scripts/registry.json` — every prompt change must land in both. This was the root of multiple past v3.x issues.

## 8. Sandbox / scaffold gotchas

- Sandbox dirs (`sandbox/app/` etc.) **must be empty** before a run. Non-empty triggers `structured_patch` mode and fails.
- `docker-compose` env vars `CODER_EXECUTION_LANE_DEFAULT` and `CODER_MODEL_DEFAULT` must match the same lane.
- `acceptance.json` criteria must be string-array or object-array.
- `coding_team_v0` has 8 steps (includes `deploy_preview`), which has a `queued` race with external credentials — independent issue, not pipeline-core.

## 9. Related docs (read these on resume)

- Latest detailed progress: `docs/03_feature_development/progress_reports/progress_20260413_v373_b_c_paths.md`
- Prior E2E validations: `progress_20260411_v35_e2e_results.md`, `progress_20260412_v36_e2e_results.md`, `progress_20260412_v37_static_audit_e2e.md`, `progress_20260412_v372_e2e_issues_and_policy_fix.md`
- Canary baseline: memory `project_canary_baseline_2026-03-28.md` (preview_mismatch = 0%)
