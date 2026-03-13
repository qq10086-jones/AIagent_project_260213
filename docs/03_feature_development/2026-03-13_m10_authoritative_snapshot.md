# M10 Authoritative Snapshot - 2026-03-13

- Date: 2026-03-13
- Purpose: freeze the authoritative baseline for today's M10 QA / Architecture assessment
- Status: ACTIVE

---

## 1. Snapshot Identity

- git branch: `main`
- git commit: `59a70710880a0563c8364ec09a55a341b1ae86c3`
- workspace posture: dirty working tree present during assessment
- assessment mode: documentation + runtime + artifact reconciliation

This snapshot is the only baseline that should be used for today's M10 review updates.

---

## 2. Runtime Governance Posture

From `configs/production_parallel_rollout.json`:

- `master_enabled = true`
- `force_sequential = false`
- `dynamic_routing_enabled = true`
- `router_mode = dynamic_routing_enforced`

Interpretation:

- M10 remains inside the approved limited enforced posture
- today is not a cohort-expansion day
- today's work is evidence closure and resilience proof, not rollout expansion

---

## 3. Execution Lane Posture

From `configs/runtime/runtime_defaults.json`:

- `worker_coder.execution_lane_default = stable_cloud_lane`
- `stable_cloud_lane = opencode + alibaba-coding-plan/qwen3-coder-plus`
- `stable_local_lane = opencode + ollama/glm-4.7-flash:latest`
- `primary_qwen_lane = opencode + alibaba-coding-plan/qwen3-coder-plus`

Assessment interpretation:

- `primary_execution_lane` for current runtime configuration is `stable_cloud_lane`
- `allowed_validation_lane` for prior worker-coding authoritative evidence also includes `stable_local_lane`
- `triage_only_lane` remains `primary_qwen_lane` until provider-path authentication is proven stable

Additional findings from live container inspection:

- `opencode` CLI requires `provider/model` refs and does not accept `dashscope/*` as an OpenCode model
- the prior `stable_cloud_lane = opencode + dashscope/...` combination was therefore an invalid provider/model pairing
- current runtime has been corrected away from that mismatch

This means today's M10 review starts from a mixed historical lane state and must not merge these lanes into one undocumented PASS conclusion.

---

## 4. Evidence Roots In Scope

Primary documentation roots reviewed:

- `docs/03_feature_development/PROGRESS_LATEST.md`
- `docs/03_feature_development/2026-03-12_m10_draft_tasklist.md`
- `docs/governance/2026-03-12_m10_phase_b_limited_enforced_signoff.md`
- `docs/03_feature_development/2026-03-12_coding_team_recovery_execution_plan.md`

Primary validation artifact root reviewed:

- `orchestrator/artifacts/validation/m10_load_test/`

Observed `m10_load_test` execution folders:

1. `2026-03-12T06-17-59-245Z`
2. `2026-03-12T06-49-13-062Z`
3. `2026-03-12T07-01-32-491Z`
4. `2026-03-12T07-13-48-265Z`
5. `2026-03-12T07-17-14-713Z`
6. `2026-03-12T07-33-55-700Z`
7. `2026-03-12T08-02-32-732Z`
8. `2026-03-13T06-46-01-592Z`
9. `2026-03-13T07-14-49-748Z`
10. `2026-03-13T07-20-07-790Z`
11. `2026-03-13T07-21-22-009Z`
12. `2026-03-13T07-22-55-749Z`
13. `2026-03-13T07-51-01-491Z`
14. `2026-03-13T07-57-31-984Z`

---

## 5. Current T-32 Reality At Snapshot Time

At snapshot time, all reviewed persisted `m10_load_test_report.md` files under the current artifact root were still `FAIL`.

Repeated common failure:

- `missing go_no_go_result.json on at least one run`

Observed latest reviewed warm-up report:

- artifact: `2026-03-12T08-02-32-732Z`
- verdict: `FAIL`
- workflow status counts: `{"failed":2,"running":4}`

Observed today's post-fix progression:

- `2026-03-13T06-46-01-592Z`
  - verdict: `FAIL`
  - workflow status counts: `{"running":1}`
  - interpretation: queue settlement blocker still active at that time
- `2026-03-13T07-14-49-748Z`
  - verdict: `FAIL`
  - workflow failed at `impl_be` with `PATCH_BUNDLE_INVALID`
  - interpretation: worker success masking and placeholder artifact contamination were still present
- `2026-03-13T07-20-07-790Z` / `2026-03-13T07-21-22-009Z` / `2026-03-13T07-22-55-749Z`
  - verdict: `FAIL`
  - workflow failed at `pm_spec` with `OpenCode authentication failed`
  - interpretation: provider auth failure became visible after worker result-path fixes
- `2026-03-13T07-51-01-491Z`
  - verdict: `FAIL`
  - workflow failed at `pm_spec` with `OpenCode model resolution failed`
  - interpretation: invalid `opencode + dashscope/...` model pairing had been removed and the next provider issue surfaced
- `2026-03-13T07-57-31-984Z`
  - verdict: `FAIL`
  - workflow failed at `pm_spec` with `Alibaba Coding Plan auth missing: set ALIBABA_CODING_PLAN_API_KEY.`
  - interpretation: current authoritative blocker is now explicit missing provider credential for the selected cloud lane

Assessment interpretation:

- the repository currently contains historical evidence of T-32 failure states
- the repository does not currently contain a persisted authoritative `PASS` load-test artifact under the reviewed `m10_load_test` root
- the current authoritative blocker is no longer ambiguous:
  - it is missing provider credential, not a latent workflow hang
- any claim that T-32 is already authoritatively passed must therefore point to a newer artifact, or be treated as a narrative conclusion that still requires evidence closure

---

## 6. Runtime Availability Check

Observed live stack state during today's review:

- `orchestrator` container was running
- `brain` container was running
- `worker-quant` container was running
- `worker-coder` was initially absent from `docker compose ps`

Root cause observed from `worker-coder` logs:

- `coding_service.js` imported `clampInt` from `static_checks.js`
- `static_checks.js` did not export `clampInt`
- result: startup-time ESM import failure in `worker-coder`

Corrective action performed today:

- added `clampInt` export to `worker-coder/static_checks.js`
- validated:
  - `node worker-coder/tests/static_checks.test.js`
  - `node --check worker-coder/coding_service.js`
  - `node --check worker-coder/static_checks.js`
- recreated `worker-coder` and confirmed it returned to `docker compose ps`

Assessment interpretation:

- before this fix, any attempt to continue authoritative M10 reruns had an execution-environment blocker
- that blocker is now reduced from active startup failure to "fixed and ready for further validation"

---

## 7. Snapshot Conclusion

This snapshot establishes the following as today's authoritative baseline:

1. M10 remains in limited enforced mode.
2. The configured runtime default lane is now `stable_cloud_lane`.
3. Historical worker-coding evidence previously used `stable_local_lane`, so lane mixing is a real governance risk.
4. The persisted `T-32` artifact set currently reviewed is still failure-only.
5. `worker-coder` had a real startup blocker today and required a code fix before further mainline validation could continue.
6. After the startup fix, worker queue settlement and false-success masking were corrected.
7. The previous `opencode + dashscope/...` lane mapping was invalid and has been removed.
8. The current mainline cloud-lane blocker is explicit missing `ALIBABA_CODING_PLAN_API_KEY`.

Therefore:

- M10 may be technically recovering
- but M10 is not yet documentation-clean or evidence-clean at this snapshot boundary
- further rerun / triage / governance updates must reference this snapshot explicitly
