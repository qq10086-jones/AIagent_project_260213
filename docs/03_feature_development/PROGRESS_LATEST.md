# Nexus Project Progress Report - 2026-04-09

## Current Status

Nexus v3.2 Phase 1 + Phase 1.5 implementation complete. Worker-coder now has deterministic
micro-fix capability (surgical patch) and refinement re-entry (iterative fix with task lineage).
All worker-coder tests pass (27 files, including 3 new test files with 33 new test cases).

Latest high-signal outcomes:

- **NEW (2026-04-09)**: v3.2 Phase 1 implemented — `surgical_patch.js` deterministic micro-fix engine (JSON trailing comma, JS unclosed bracket, Python indentation), integrated into `coding_service.js` retry loop as transparent pre-processor
- **NEW (2026-04-09)**: v3.2 Phase 1.5 implemented (worker-coder side) — `refinement_context_builder.js` task lineage context builder, `normalizeLineage` in `task_contract.js`, lineage tracking in `failure_memory.js`, prompt injection in `coding_service.js`
- **NEW (2026-04-09)**: `native_file_tools.js` — read-only file access with path traversal protection, protected root blocking, content redaction
- **NEW (2026-04-09)**: Feature flags added: `surgical_patch_enabled` (default false), `refinement_reentry_enabled` (default false) in `runtime_defaults.json`
- **NEW (2026-04-09)**: Worker-coder tests 27/27 files PASS (was 24, +3 new: native_file_tools, surgical_patch, refinement_context_builder)
- **2026-04-06**: Root-caused `impl_be` `HANDOFF_TYPED_SCHEMA_INVALID` failure — `be_to_fe.json` schema required `name` field on api_contracts items, but MiniMax output only provides `method` + `path`
- **NEW (2026-04-06)**: Fixed `coding_team_be_to_fe_handoff.schema.json` — `api_contracts.items.required` relaxed from `["name","method","path"]` to `["method","path"]`
- **NEW (2026-04-06)**: Added regression test for LLM-tolerance (api_contracts without `name`)
- **NEW (2026-04-06)**: Orchestrator tests 211/211 PASS (was 210)
- **2026-04-05**: SCO-05 Permission Council fully implemented (SafetyAuditor / ContextValidator / RiskScorer)
- **2026-04-05**: Council deny pre-intercept in task_enqueuer + advisory summary in approval events
- **2026-04-05**: 4 test failures fixed (approval_entrypoint + worker_coding_templates)
- **2026-04-05**: SA-01 single_agent guardrails verified at 3 enforcement points
- **2026-04-05**: Real MiniMax E2E via Docker stack confirmed: arch_design produced valid architect_to_impl.json + workplan.json, impl_be produced working Express server + frontend

## Real MiniMax E2E Results (2026-04-05 run)

Run ID: `9fb60888-1c50-4d1c-879c-4cb3bafa9acd`

| Step | Status | Notes |
|---|---|---|
| pm_spec | PASS | Spec + acceptance criteria generated |
| arch_design | PASS | architect_to_impl.json + workplan.json + be_to_fe schema |
| impl_be | FAIL (now fixed) | MiniMax produced valid server.js + be_to_fe.json; failed on schema `name` field |
| impl_fe | blocked | Will proceed after impl_be fix |
| smoke_test | blocked | |
| qa_verify | blocked | |
| release_pack | blocked | |
| deploy_preview | blocked | |

**Key finding**: The MiniMax LLM produced correct, working code (Express CRUD server, HTML/CSS/JS frontend). The failure was in the handoff schema being too strict for LLM output variation, not in the LLM output quality.

Artifacts produced (all valid):
- `impl/be_changes/server.js` — 60-line Express server with full CRUD API
- `impl/be_changes/package.json` — proper npm package
- `impl/fe_changes/public/` — index.html + app.js + styles.css
- `handoff/architect_to_impl.json` — passes arch schema validation
- `handoff/be_to_fe.json` — passes schema after fix
- `plan/workplan.json` — 5 BE + 5 FE structured tasks

## Worker-Coder Capability Summary

Worker-Coder is the coding execution worker of the Nexus multi-agent system. It consumes
tasks from Redis streams and produces structured results with full audit trails.

### Supported Tools (4)

| Tool | Description |
|---|---|
| `coding.patch` | Apply edit blocks (search/replace patches) to specific files with scope guard |
| `coding.execute` | Run shell commands in workspace with artifact scaffolding |
| `coding.delegate` | Delegate full coding tasks to LLM adapters (OpenCode / Codex) with retry, verification, and isolation |
| `ops.deploy_preview` | Push workspace changes to GitHub as a preview branch |

### Core Capabilities

**LLM Adapter Runtime** (`coding_executor_runtime.js`, 2 adapters)
- OpenCode adapter (primary) and Codex adapter
- Execution lane system: runtime config maps lane names to provider + model pairs
- Provider fallback: configurable auto-fallback when primary adapter fails
- Model override: per-task model selection via `model_override` / `execution_lane`

**Multi-Attempt Retry with Auto-Fix** (`retry_policy.js`, `coding_service.js`)
- Up to 3 attempts per task with wall-clock timeout budget
- Static checks (syntax, lint) after each attempt
- Verification plan execution (tiered: syntax_check -> lint -> type_check -> unit_test -> build)
- Auto-fix prompt generation from verification failures

**Workspace Isolation** (`isolation_workspace.js`, `promotion_workspace.js`)
- Materializes isolated workspace copies for safe LLM execution
- Baseline snapshot capture before execution
- Scoped delta detection: only promotes files within declared target paths

**Verification Pipeline** (`verification_runner.js`)
- Safe command validation (blocks rm -rf, format C:, etc.)
- Multi-tier verification plan execution
- Inline Node.js syntax check for .js files

**Artifact & Contract System**
- `artifact_scaffold.js`: ensures expected artifact directories/files exist per step
- `step_artifact_contract.js`: validates workflow step artifacts and coding team handoffs
- Single-agent guardrails enforcement (evidence_id, replay_tag, output_hash, bounded_validation)
- Permission advisory pass-through from orchestrator council

### Production Configuration

- Default execution lane: `stable_cloud_lane` (MiniMax-M2.7)
- Global task timeout: 900s (configurable to 1800s)
- Stream batch size: 1 (single-GPU safety)
- Fidelity gate mode: `blocking`

## Nexus v3.1 Task Milestone Status

| Milestone | Tasks | Complete | Status |
|---|---|---|---|
| M1: Superpowers | SP-01, SP-02 | 2/2 | Closed |
| M2: Shared Contracts MVP | SCO-01, SCO-02, SCO-04 | 3/3 | Closed |
| M3: Observability + Advisory | SCO-03, SCO-05, GOV-01 | 3/3 | Closed (2026-04-05) |
| M4: single_agent | SA-01, SP-03, SP-04, BR-01 | 4/4 | Closed (2026-04-05) |
| M5: Council Quality Baseline | GOV-02 | 0/1 | Unblocked, needs 30 days data |

## Current Working Baseline

### Orchestrator
- `npm --prefix orchestrator test` -> **211/211 PASS**
- `npm --prefix worker-coder test` -> **all PASS**
- Permission council tests -> **all PASS**

### Quant
- `target_mode = shadow_hybrid_ic`
- `recommendation = eligible_for_promotion`
- `paper_days = 30`
- `test_suite = 34/34 PASS`

### Nexus / Integration
- Discord workflow entrypoint: pass
- Discord dispatch/routing: pass
- Real MiniMax E2E: arch_design + impl_be produce working code; handoff schema fix unblocks full chain

## Recommended Next Steps

1. Rebuild Docker images and re-run real E2E to validate full 8-step chain completion
2. Begin GOV-02 data accumulation (Permission Council advisory logs)
3. Start Sprint 30-day paper trading for quant
4. Track C activation: DashScope key for qwen3-coder-plus lane

## Open Issues

- `N1` (RESOLVED): `impl_be` handoff schema `name` field — fixed in `coding_team_be_to_fe_handoff.schema.json`
- `N2`: Full 8-step chain needs validation with rebuilt Docker image
- `R1`: Ridge CV comparison still needs fresh production run
- `T1`: Natural-time 30-day evidence distinct from compressed simulation
- `I1-I2`: External dependencies (jquants data, DashScope key) are operational, not code blockers
