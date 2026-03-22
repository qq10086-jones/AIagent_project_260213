# N2 Concurrent Baseline

- Verdict: FAIL
- Execution lane: stable_local_lane
- stream_batch_size: 1
- Concurrency: 2 workflows

## Runs
- `34f47612-e7b0-4231-9456-940dd3c0764a` / `c77bcb2a-1e8a-46a7-9c24-dfd2a7f10755`: failed at `arch_design` with `ARCH_REQUIRED_SECTIONS_MISSING`.
- `91bf258c-ded4-47c9-97a5-1448d4d8d70e` / `b7ace6df-b535-4344-95e1-1bdcd5a8cbe0`: remained running at `arch_design` during the observation window.

## Interpretation
- Current stable_local_lane is not concurrency-ready for two full coding_team_v0 workflows.
- The first pressure point is upstream design generation (`pm_spec` / `arch_design`), not only implementation stages.
- Next action: diagnose `arch_design` under concurrent load before trying `stream_batch_size=2`.

## Rerun After Arch Guidance Fix
- Code changes applied:
- `configs/prompt_scripts/registry.json`
- `orchestrator/configs/prompt_scripts/registry.json`
- `orchestrator/src/domain/workflow_state.js`
- Concurrent rerun evidence: `n2_concurrent_baseline_rerun_after_arch_fix.json`
- Rerun workflow IDs:
- `21787f20-fc6e-499f-9e4b-a25c99ba80f7` / `20ae588f-1e2d-4918-8249-50483862b9ef`
- `e83773dc-ad4b-49d6-824f-15f5eccc49b4` / `f1f41937-967c-48a9-9043-eeb605242ac3`
- Observed improvement:
- Both workflows advanced past `arch_design` and reached `impl_be`.
- Orchestrator logs recorded `step.completed ... completed=arch_design next=impl_be` for both runs.
- Current conclusion:
- The `arch_design` prompt/contract alignment reduced the original concurrent failure mode.
- `stable_local_lane` is still not concurrency-ready for 2 full workflows because both rerun workflows remained non-terminal during the observation window and the active bottleneck shifted to `impl_be`.
