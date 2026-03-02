# OpenClaw Nexus Multi-Agent System Design (Latest)

## Version
- Date: 2026-03-01
- Baseline sources:
  - `docs/OpenClaw_Nexus_Refactor_Patch_v1_0.md`
  - `docs/90_archive/legacy_workspace/AIagent_project/Project_OpenClaw_Nexus_v1_2_3_MAS.md`
  - `docs/01_design/system/task_queue_protocol.md`
  - `docs/01_design/coding/coding_agent_design_latest.md`

---

## 1. Executive Decision

### 1.1 Core Judgment (Strict)
Current Nexus progress is **not invalid**, but architecture investment is imbalanced:
- Valuable and should be retained: task protocol, governance trail, worker isolation, quant domain pipeline.
- Over-invested and should be reduced/replaced: generic agent session bridge, multi-agent orchestration shell, and workflow engine duplication.

This means "70% effort wasted" is directionally understandable at the feeling level, but technically inaccurate:
- Not waste: governance primitives and quant execution chain are reusable core assets.
- Real waste risk: continuing to self-build commodity orchestration that open-source already solves better.

### 1.2 Build-vs-Buy Policy (Hard Rule)
- Adopt/Wrap existing OSS for generic layers:
  - Discord/session bridge: Kimaki-style capability
  - role-pipeline templates: OpenSwarm-style patterns
  - deterministic workflow + resumable approvals: Lobster-style shell
- Keep self-built only for differentiated layers:
  - policy/approval/audit control plane
  - domain execution rules and replay tests (quant first, then other project types)

---

## 2. Objective Current-State Assessment

### 2.1 What is Working and Valuable
1. Queue + lifecycle base is valid:
   - Redis Streams + task status + event logging already form a usable execution backbone.
2. Governance basis exists:
   - approval, risk levels, run_id/task_id traceability, artifacts all exist in current implementation.
3. Domain chain exists:
   - worker-quant has practical logic and data-flow contracts, not just scaffolding.
4. Coder delegation proved feasible:
   - `/coder` -> `coding.delegate` -> adapter -> workspace write is already validated.

### 2.2 Where Current Design is Heavy
1. Orchestrator responsibilities are too broad:
   - one file currently mixes routing, chat, tool orchestration, Discord UX, result rendering, approval APIs.
2. Brain orchestration is partially hardcoded:
   - tool and flow branching in supervisor is still monolithic and difficult to scale.
3. Generic platform features are being rebuilt:
   - session bridge and multi-role orchestration layers are not project-specific differentiators.

### 2.3 Strict Conclusion on Orchestrator
Orchestrator mechanism is **useful but currently oversized**.
- Correct role: control-plane gateway (policy, approval, audit, dispatch, artifact index).
- Incorrect role: becoming the primary place for workflow intelligence and provider/session behavior details.

---

## 3. Retain / Replace / Deprecate Matrix

### 3.1 Retain (Must Keep)
1. Task protocol and reliability contract:
   - `run_id`, `idempotency_key`, task lifecycle, reclaim/DLQ.
2. Facts/evidence/replay principles from MAS:
   - facts must be evidence-backed and traceable.
3. Governance control plane:
   - risk policy, approval, audit log, artifact registry.
4. Worker isolation model:
   - `worker-quant`, `worker-coder`, future workers with independent runtime.
5. Quant domain logic:
   - this is core differentiation and should continue to deepen.

### 3.2 Replace or Wrap (Adopt Preferred)
1. Discord session bridging and conversation-to-local-workspace glue:
   - replace/wrap with mature OSS integration rather than growing custom channel/session code.
2. Generic multi-agent role orchestration:
   - borrow/adopt existing patterns (worker/reviewer/test/doc) instead of custom from scratch.
3. Deterministic workflow shell and resume semantics:
   - adopt a JSON-first workflow layer with approval gates and resumability.

### 3.3 Deprecate or Freeze (Stop Expanding)
1. Expanding `coding.patch` / `coding.execute` as primary coding path.
2. Adding more provider-specific branches directly into orchestrator routing logic.
3. Growing brain-side monolithic hardcoded workflow branches for each new skill.

### 3.4 Module-Level Decision List (Current Repo)
Keep/Harden:
1. `worker-quant/*`: retain as domain execution core.
2. `worker-coder/adapters/codex_adapter.js`: retain as adapter baseline.
3. `worker-coder/coding_service.js`: retain, continue standardizing output contract.
4. `orchestrator/src/index.js` queue/approval/audit related parts: retain.
5. `docs/01_design/system/task_queue_protocol.md`: retain as protocol source of truth.

Refactor/Split:
1. `orchestrator/src/index.js` (large monolith): split into ingress, policy, task-dispatch, result-render modules.
2. `brain/supervisor.py`: move skill-specific branching toward pluggable workflow/registry model.

Freeze/Deprioritize:
1. `orchestrator/src/patch_manager.js`: no further feature expansion; fallback-only.
2. `coding.patch` and `coding.execute` route as primary path: freeze at maintenance level.

---

## 4. Target Architecture (Nexus vNext)

### 4.1 Control Plane (Nexus-Owned)
- Orchestrator (thin):
  - ingress normalization (`/coder`, `/quant`, future `/ui`, `/db`)
  - policy engine (risk scoring + gate decision)
  - approval service (request/approve/reject/resume)
  - audit/event writer
  - artifact index registry
- Data services:
  - Redis Streams (task/result + pending recovery)
  - Postgres (runs/tasks/event_log/facts)
  - object store (artifacts)

### 4.2 Execution Plane (Pluggable)
- Adapters and engines:
  - coding engines (Codex first, additional providers later)
  - workflow shell (deterministic, resumable)
  - optional bridge/orchestration OSS components for session and role-pipeline
- Workers:
  - worker-quant, worker-coder, future worker-ui/worker-db/worker-media

### 4.3 Domain Plane (Nexus Differentiation)
- Rule packs and acceptance suites:
  - quant execution constraints, replay regression, risk constraints
  - future project-type specific rule packs

---

## 5. Contracts (Non-Negotiable)

### 5.1 Task Contract
Every tool task must include:
- `run_id`
- `task_id`
- `idempotency_key`
- `tool_name`
- `payload_json`

### 5.2 Artifact Pack Contract
Every completed run must provide standardized artifacts:
- `plan` (or execution intent)
- `diff` (or explicit "no code diff" result)
- `stdout/stderr`
- `tests` summary
- `risk_report`
- `run_summary`

### 5.3 Approval Contract
- Intent-level entry (`/coder`) is not equal to side-effect approval.
- Approval is required only for high-risk actions by policy hit.
- Approval event must record reasons and actor.

---

## 6. Policy-as-Code Model

### 6.1 Risk Dimensions
- path sensitivity (infra/ci/secrets/deploy)
- command sensitivity (destructive/system/network/install)
- data sensitivity (db schema destructive actions)
- external side effects (network/remote operations)

### 6.2 Gate Levels
- L0: auto-run (doc/light code)
- L1: auto-run + tests
- L2: guarded run (integration and schema checks)
- L3: manual approval mandatory

---

## 7. Capability Registry

A machine-readable registry should define:
- project types (`quant_execution`, `webapp_crm`, `data_pipeline`, etc.)
- skill roles (`product`, `architect`, `backend`, `frontend`, `qa`, `devops`, `security`, domain skills)
- workflow definitions
- policy sets
- acceptance suites

This prevents prompt drift and keeps scaling deterministic.

---

## 8. Migration Plan (Stop-Loss Refactor)

### Phase A (Immediate, 1-2 weeks)
1. Freeze new monolithic orchestration logic in orchestrator and brain.
2. Keep current queue/governance backbone stable.
3. Establish retain/replace list as engineering backlog labels.

### Phase B (Near-term, 2-4 weeks)
1. Introduce deterministic workflow shell for high-impact flows.
2. Move approval gates to workflow checkpoints (not ad-hoc UI logic).
3. Normalize artifact pack output across coder/quant.

### Phase C (Medium-term, 4-8 weeks)
1. Replace/wrap session bridge and role-pipeline commodity layers.
2. Keep Nexus control plane as single governance authority.
3. Validate at least two project types through same OS pathway.

---

## 9. Acceptance Criteria for This Architecture

1. Two different project types can run on the same control-plane contracts.
2. High-risk side effects consistently stop at approval gates and can resume.
3. Replay artifacts can reconstruct key execution decisions for any run.
4. Orchestrator remains thin: no provider/session-specific heavy logic growth.
5. Domain rules become the primary source of value, not orchestration boilerplate.

---

## 10. Immediate Action Checklist

1. Keep and harden current quant chain + coder delegation adapters.
2. Stop adding generic orchestration features directly in monolithic files.
3. Promote policy, approval, audit, artifact contracts as first-class modules.
4. Introduce pluggable workflow and capability registry before adding new skills.
5. Use OSS in bridge/pipeline/workflow commodity layers via wrappers, not forks-by-default.

---

## Related Docs
- `docs/OpenClaw_Nexus_Refactor_Patch_v1_0.md`
- `docs/01_design/system/task_queue_protocol.md`
- `docs/01_design/coding/coding_agent_design_latest.md`
- `docs/01_design/quant/quant_design_latest.md`
- `docs/90_archive/legacy_workspace/AIagent_project/Project_OpenClaw_Nexus_v1_2_3_MAS.md`

---

# vNext Addendum (2026-03-01)


## 2.x OSS Adoption (vNext — explicit dependencies)

> You already run **OpenClaw** (control plane foundation).  
> vNext makes the **“buy vs build”** stance executable by naming concrete OSS repos/components and defining strict integration boundaries.

### 2.x.1 Selected OSS Components (Pin & Wrap)

**A) Session bridge (Discord → workspace → agent session)**
- **Primary**: **Kimaki** (Discord channels bound to local project dirs; triggers OpenCode sessions; file/command/tool access)
- Strategy: **Replace** any home-grown Discord↔workspace session manager with Kimaki **or Wrap it** behind `ingress.discord.*` so Nexus keeps governance.

**B) Role pipeline template (Worker/Reviewer/Test/Docs)**
- **Reference**: **OpenSwarm** (multi-agent pipelines + Discord reporting + ticketing integrations)
- Strategy: **Borrow patterns / optionally adopt skeleton**, but **governance remains in Nexus** (policy/approval/audit/artifacts).

**C) Deterministic workflow shell (typed steps + approval gate + resumable execution)**
- **Primary**: **Lobster (OpenClaw-native)** workflow shell
- Strategy: **Adopt as the only supported execution wrapper** for any multi-step tool/skill chain (this prevents supervisor/orchestrator “if/else sprawl”).

**D) Coding engines (pluggable execution backends)**
- Supported backends (choose per project type / availability):
  - **Codex adapter** (current)
  - **OpenCode / opencode** (optional alternative engine)
  - **SWE-agent / Aider** (optional, task-type dependent)
- Strategy: engines are **replaceable**, but must speak the same **Task Contract** and produce the same **Artifact Pack**.

### 2.x.2 Integration Boundaries (Hard Rules)

1. **Governance single source of truth**
   - No OSS component may bypass: `policy → approval → audit → artifact registry`.
2. **All side-effects must be mediated**
   - “Side-effect” includes: dependency install, network download, infra/deploy changes, DB migrations, secrets/credentials access, host actions.
   - Side-effects must run inside a Lobster step that is risk-scored and (if needed) gated by approval.
3. **Artifacts are mandatory**
   - Every run produces a complete Artifact Pack (or explicitly declares why a field is absent).
4. **Orchestrator must remain thin**
   - Provider/session/coding-engine specifics live in adapters; orchestrator flow is registry + workflow driven.



## 9.x Risk Register (vNext — strict review)

This section is a **tracking system**, not prose. Each risk should map to an owner + mitigation + “done when”.

### 9.x.1 Architecture / Delivery Risks

1) **Orchestrator re-bloat (if/else sprawl)**
- Failure mode: every new skill/provider adds branches to supervisor/orchestrator, recreating a monolith brain.
- Mitigation:
  - Registry-driven routing only; forbid provider-specific logic in orchestrator.
  - CI check: orchestrator code cannot import provider modules directly.
- Done when: adding a new skill only requires updating registry + workflow, not core orchestrator.

2) **Registry becomes “a doc”, not a contract**
- Failure mode: capability registry exists but is not validated; runtime still depends on prompts/hardcoding.
- Mitigation:
  - Define `registry.schema.json` + `validate_registry.py` in CI.
  - Orchestrator refuses to run if registry invalid.
- Done when: registry validation blocks merges and runtime rejects invalid registry.

3) **Resumable approvals lack formal semantics**
- Failure mode: “resume” re-runs from the beginning; results drift; approvals become unsafe.
- Mitigation:
  - Task protocol must define: `checkpoint_id`, `resume_token`, `idempotency_key`, workspace revision/hash binding.
  - Approval resumes from the next step after checkpoint (no full replay unless explicitly requested).
- Done when: resume is deterministic for the same checkpoint + inputs.

4) **Artifact Pack not enforced**
- Failure mode: runs sometimes omit tests/risk report/diff; later debugging/replay becomes impossible.
- Mitigation:
  - Implement `artifact_pack_validator` (existence + JSON schema validity).
  - Orchestrator marks run failed if artifact pack invalid.
- Done when: every run produces validated artifacts (or structured “not produced” reasons).

5) **Policy-as-code is declared but not operational**
- Failure mode: risk levels are subjective; teams argue; approvals inconsistent.
- Mitigation:
  - Publish `policy_rules_v0.yaml` with ~10 common rules (paths, commands, deps, network, db migration, secrets).
  - Default rule: unknown side-effect escalates to L2/L3.
- Done when: risk scoring is reproducible and visible in `risk_report.json`.

6) **OSS integration drift / upstream breaking changes**
- Failure mode: Kimaki/OpenSwarm/Lobster updates break adapters; integration becomes fragile.
- Mitigation:
  - Pin versions; wrap behind stable internal interfaces.
  - Add contract tests for adapters (golden inputs → expected task+artifact outputs).
- Done when: upgrades require only adapter changes; contract tests catch breakage.

### 9.x.2 Security / Isolation Risks

7) **Credentials & sensitive files leakage**
- Failure mode: low-risk tasks read `.env`, CI tokens, cookies; secrets leak into logs/LLM context.
- Mitigation:
  - Runtime isolation + least-privilege credentials; secrets never enter model context.
  - Path denylist + redaction in logging/artifacts.
- Done when: secrets scanning on artifacts passes and access to sensitive paths is blocked by default.

8) **Supply-chain risk from dependency changes**
- Failure mode: “normal” dependency update introduces malicious package or compromised version.
- Mitigation:
  - Treat deps changes as L3 by default (approval + security scan).
  - SBOM + lockfile diff + SCA gate in CI.
- Done when: dependency updates always produce SCA report and require approval.

### 9.x.3 Product / Scope Risks

9) **Project types remain “one-off”**
- Failure mode: system is claimed as general, but only validated on a single project type.
- Mitigation:
  - Force validation on at least two project types in Phase B (e.g., `webapp_crm` + `data_pipeline`).
- Done when: both types pass the same contracts (task/artifact/policy/gates) with minimal glue changes.

