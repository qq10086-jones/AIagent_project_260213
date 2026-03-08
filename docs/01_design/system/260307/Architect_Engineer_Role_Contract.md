# Architect Engineer Role Contract
## Coding Team — arch_design Step
## Date: 2026-03-07

---

## 1. Purpose

This contract defines the full specification for the Architect Engineer agent role in the Coding Team workflow.

It supersedes the 3-line `arch_design` step instructions in the current `workflow_engine.js`.

The Architect Engineer is not a document generator. It is the technical decision authority for the Coding Team workflow. Its output is binding on all downstream agents (Backend, Frontend, QA).

---

## 2. Position in North Star Pipeline

```
PM Spec (pm_spec)
↓
[Architect Engineer] ← this contract
↓
Backend Impl (impl_be) + Frontend Impl (impl_fe) + QA Verify (qa_verify)
```

The Architect receives PM output and translates it into a technical blueprint. Implementation agents may not proceed without a valid Architect handoff.

---

## 3. Current State vs Target State

### Current State (as of 2026-03-07)
- Step uses `coding.delegate` with 3 generic instructions
- No codebase analysis phase
- No ADR format required
- No interfaces.md produced
- `decisions` array absent from handoff manifest
- Validator only checks artifact file existence, not content quality

### Target State (this contract)
- Prompt script v2 (`architect.system_spec.v2`) with full architectural guidance
- Explicit codebase context input
- ADR format required for every major decision
- `plan/interfaces.md` required
- `decisions` array required and validated in handoff
- Content validators check for required headings, not just file presence

Migration tracked in: `OpenClaw_Nexus_Engineering_Task_List_M3.md`, WS-12.

---

## 4. Input Specification

### 4.1 Required PM Artifacts
| Artifact | Validation |
|----------|-----------|
| `plan/spec.md` | Must contain: scope, user_stories, acceptance_criteria, non_goals, artifact_list |
| `plan/acceptance.json` | Must be schema-valid per `coding_team_pm_acceptance.schema.json` |
| `plan/milestones.md` | Must exist and be non-empty |
| `handoff/pm_to_architect.json` | Must be schema-valid per `coding_team_pm_handoff.schema.json` |

### 4.2 Codebase Context (injected by workflow engine before prompt)
The workflow engine must inject the following context into the Architect's task prompt:
- Project type (e.g., `webapp_crm`)
- List of top-level directories in the repository
- List of active dependencies from `package.json` or equivalent
- If Memory Layer is available: prior ADR summaries for this project

### 4.3 Constraints from PM Handoff
- `scope_summary` from `handoff/pm_to_architect.json` must frame the architectural scope
- `acceptance.criteria` from the same handoff must be verifiable against the architecture plan

---

## 5. Required Output Artifacts

All artifacts are relative to `artifacts/release/{run_id}/`.

### 5.1 plan/arch.md

**Required headings:**
- `## System Overview` — 1–3 paragraphs describing the technical solution
- `## Module Breakdown` — table or list with: module name, layer (FE/BE/shared/infra), owner, description
- `## Technology Decisions` — references to ADR IDs; one row per major decision
- `## Layer Boundaries` — what lives in each layer, what is explicitly forbidden in each layer
- `## Integration Points` — list of all APIs, webhooks, queues, or data contracts this work introduces or modifies
- `## Dependency Graph` — text or ASCII diagram showing module dependencies
- `## Constraints` — hardcoded constraints that all implementation agents must respect

**Validation rule:**
All 7 headings must be detectable by the heading validator.

---

### 5.2 plan/adr/adr_NNN.md (one file per major decision)

**Format (mandatory):**
```markdown
# ADR-NNN: [Decision Title]

## Status
Accepted | Proposed | Superseded

## Context
[Why is this decision needed? What is the problem or requirement that forces a choice?]

## Decision
[What was decided. Be specific: library names, patterns, data formats.]

## Consequences
[What becomes easier as a result? What becomes harder or more constrained?]

## Alternatives Considered
[Other options that were evaluated and why they were rejected.]
```

**When an ADR is required:**
- Choice of persistence technology (DB, file, cache)
- Choice of authentication/authorization pattern
- Choice of inter-module communication pattern (REST, events, direct call)
- Choice of data format for a new schema
- Any decision that would be hard to reverse later

**Minimum:** 1 ADR per `arch_design` run. Zero ADRs is a validation failure.

---

### 5.3 plan/interfaces.md

**Required content:**
- Every API endpoint introduced or modified by this workflow run
- Every internal module interface (function signature, event schema, or data contract)

**Format per interface:**
```markdown
### Interface: [Name]
- Type: REST endpoint | internal function | event | data contract
- Owner: FE | BE | shared
- Method/Signature: GET /api/customers | function getCustomers(filter): Customer[]
- Input: [schema or description]
- Output: [schema or description]
- Error cases: [list]
```

**Validation rule:**
File must exist and contain at least 1 `### Interface:` heading.

---

### 5.4 risk/risk_report.json

**Schema:** `coding_team_arch_risk_report.schema.json`

**Required fields per risk:**
- `risk_id` — string identifier
- `category` — one of: technical, integration, scope, security, performance
- `description` — what could go wrong
- `probability` — low | medium | high
- `impact` — low | medium | high
- `mitigation` — what the implementation agent should do to reduce the risk

**Minimum:** 1 risk entry per run.

---

### 5.5 plan/workplan.md

**Required content:**
- Per-role task breakdown: BE tasks, FE tasks, QA acceptance criteria
- Explicit scope boundary per role (what is IN scope and what is NOT)
- Sequencing notes if order matters

---

### 5.6 handoff/architect_to_impl.json

**Schema:** `coding_team_arch_handoff.schema.json`

**Required fields:**
```json
{
  "from_step": "arch_design",
  "to_steps": ["impl_be", "impl_fe", "qa_verify"],
  "modules": [
    { "name": "CustomerList", "layer": "fe", "description": "..." }
  ],
  "interfaces": [
    { "name": "GET /api/customers", "type": "rest", "owner": "be" }
  ],
  "decisions": [
    { "adr_id": "ADR-001", "title": "Use PostgreSQL for persistence", "status": "Accepted" }
  ],
  "risks": [
    { "risk_id": "R001", "description": "...", "impact": "medium" }
  ],
  "scope_constraints": [
    "No authentication in this iteration",
    "No pagination in Phase 1"
  ]
}
```

**Validation failure conditions:**
- `decisions` array is empty → reject
- `interfaces` array is empty → reject
- `modules` array is empty → reject
- Any required field missing → reject

---

## 6. Forbidden Actions

The Architect Engineer must not:
- Write implementation code (no `.js`, `.ts`, `.py`, `.sql` files except in the artifact directory)
- Create files outside `artifacts/release/{run_id}/`
- Make technology decisions that contradict existing project constraints without a corresponding ADR
- Leave `decisions`, `interfaces`, or `modules` arrays empty in the handoff

---

## 7. Failure Modes and Escalation

| Condition | Action |
|-----------|--------|
| Insufficient codebase context to make a decision | Set `clarification_required: true` in handoff; emit `clarification_required` step status |
| Cannot determine module boundaries from PM spec | Request PM spec revision via `clarification_required` flag |
| PM artifacts fail validation | Block `arch_design` step; emit `HANDOFF_VALIDATION_FAILED` |
| ADR count is 0 | Validation failure; step must be rerun |
| `plan/interfaces.md` is absent | Artifact audit failure; downstream steps blocked |

---

## 8. Quality Acceptance Criteria

The `arch_design` step passes stage review when:
- All required artifacts exist
- `plan/arch.md` contains all 7 required headings
- At least 1 ADR exists in `plan/adr/`
- `plan/interfaces.md` contains at least 1 interface definition
- `risk/risk_report.json` contains at least 1 risk entry
- `handoff/architect_to_impl.json` passes schema validation with non-empty `decisions`, `interfaces`, and `modules`
- Human review of `plan/arch.md` confirms technology decisions are grounded in project context, not generic placeholders

---

## 9. Prompt Script Reference

Script ID: `architect.system_spec.v2`
Registry: `orchestrator/configs/prompt_script_registry.json`

The prompt must:
- Instruct the model to scan and list existing project modules before designing
- Embed the ADR template inline for structured output
- Specify all artifact paths as absolute paths under the artifact root
- Explicitly enumerate the required output sections for `plan/arch.md`
- Instruct the model to produce a non-empty `decisions` array

---

## 10. Related Documents

- `OpenClaw_Nexus_Design_Document_v2.md` — Section 7.3 (full Architect spec), Section 8 (Coding Team workflow)
- `OpenClaw_Nexus_Engineering_Task_List_M3.md` — WS-12 (hardening tasks)
- `Coding_Team_Handoff_Contract.md` (260306) — current handoff structure (being updated by WS-12-03)
- `orchestrator/contracts/coding_team_arch_handoff.schema.json` — typed handoff schema
- `orchestrator/contracts/coding_team_arch_risk_report.schema.json` — risk report schema
