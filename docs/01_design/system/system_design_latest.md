# OpenClaw Nexus Multi-Agent System Design (Latest)

## Version
- Date: 2026-03-01
- Baseline sources:
  - `docs/OpenClaw_Nexus_Refactor_Patch_v1_0.md`
  - `docs/90_archive/legacy_workspace/AIagent_project/Project_OpenClaw_Nexus_v1_2_3_MAS.md`
  - `docs/01_design/system/task_queue_protocol.md`
  - `docs/01_design/coding/coding_agent_design_latest.md`
  - `docs/01_design/system/260305/coding_team_fasttrack_ui_strategy.md`
  - `docs/01_design/system/260305/coding_team_fasttrack_ui_strategy_r1.md`

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
