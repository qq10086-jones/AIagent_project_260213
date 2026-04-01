# Coding Team Generic Project Type Patch Design

- Date: 2026-03-24
- Scope: Coding Team routing, project type modeling, workflow/template decoupling
- Status: Draft patch design

---

## 1. Problem Statement

The current Coding Team runtime has a structural modeling defect:

1. `coding_task.default_workflow` points to `coding_team_v0`.
2. `coding_team_v0.project_type` is hard-bound to `webapp_crm`.
3. `/coder` requests in Discord force `workflow_id=coding_team_v0` and `project_type=webapp_crm`.

This means the only production-grade multi-step Coding Team workflow is treated as if it were a CRM-specialized workflow, even when the user request is unrelated to CRM.

Observed failure mode:

- User asks for a simple coding task such as "write an HTML file with hello, world".
- System enters the correct high-level lane: Coding Team orchestration.
- System enters the wrong template: `webapp_crm`.
- PM/Architect/Impl/QA artifacts become CRM-shaped even if the final code output is small and unrelated.

This is not primarily a model-quality issue. It is a contract and architecture issue: workflow skeleton and project template are currently coupled.

---

## 2. Design Goal

Introduce a generic Coding Team project-type layer so that:

- Coding Team orchestration remains available for non-CRM work.
- Workflow skeleton can be reused across multiple task shapes.
- Project-specific artifact contracts are selected explicitly instead of being inherited accidentally from CRM.
- Unknown-but-valid coding requests degrade safely into a generic template, not a wrong specialized template.

---

## 3. Non-Goals

This patch does not attempt to:

- redesign the full PM/Architect/Impl/QA/Release workflow graph
- replace `coding_team_v0`
- remove `webapp_crm`
- introduce a fully dynamic workflow generator
- solve all project taxonomy cases in one slice

This patch is a modeling correction and a safe extensibility patch.

---

## 4. Root Cause

### 4.1 Current Coupling

The current model incorrectly assumes:

- default Coding Team workflow = CRM workflow

In practice there are two different concepts:

1. Workflow skeleton
   - PM
   - Architect
   - Backend/Frontend or implementation
   - QA
   - Release

2. Project template
   - required artifacts
   - prompt script bindings
   - target paths
   - acceptance commands
   - handoff shapes
   - release-pack requirements

These should not be the same object.

### 4.2 Current Runtime Consequences

- Router chooses the correct collaboration mode but wrong project semantics.
- Prompt contracts overfit to CRM-style outputs.
- Acceptance suites and release-pack expectations become mismatched for non-CRM tasks.
- The system cannot honestly answer "I need Coding Team, but this is not CRM."

---

## 5. Design Principles

1. Workflow skeleton must be reusable.
2. Project type must encode delivery semantics, not orchestration mode.
3. Router must never silently coerce unknown coding work into a specialized template.
4. Generic fallback is preferable to specialized misrouting.
5. Clarification is preferable to hallucinated specialization.
6. Existing `webapp_crm` behavior must remain available and backward-compatible.

---

## 6. Proposed Model

### 6.1 Separate Three Layers

The patched model must distinguish:

1. Intent lane
   - `chat`
   - `coding`
   - `quant`
   - `research`
   - `ops`

2. Orchestration mode
   - `direct_reply`
   - `single_agent`
   - `orchestrated_workflow`
   - `clarification_required`

3. Project type
   - `webapp_crm`
   - `generic_app`
   - `single_file_html`
   - `generic_coding_task`
   - future types such as `cli_tool`, `browser_extension`, `data_pipeline`

The key correction is:

- workflow chooses "how many stages and roles"
- project type chooses "what kind of output contract"

### 6.2 New Generic Project Types

This patch introduces the following minimum set:

#### A. `generic_coding_task`

Purpose:

- safe fallback for coding requests that require Coding Team orchestration but do not match a known specialized template

Expected use cases:

- small utilities
- scripts
- one-off repo changes
- task bundles that are valid coding work but do not clearly map to CRM/web app/data pipeline/etc.

Characteristics:

- generic PM/architect/impl/QA contracts
- minimal required artifacts
- no CRM-specific nouns
- no forced API/customer/pipeline assumptions

#### B. `single_file_html`

Purpose:

- tasks whose primary deliverable is a single static HTML file or tiny static web asset set

Expected use cases:

- `index.html`
- simple landing page
- single-file mockups
- "hello world" HTML

Characteristics:

- allows Coding Team workflow if requested explicitly
- emits reduced artifact contract
- implementation targets static asset paths rather than app/server assumptions

#### C. `generic_app`

Purpose:

- non-CRM application tasks that still need multi-step system planning

Expected use cases:

- internal tool
- dashboard
- admin utility
- simple full-stack prototype not centered on CRM domain

Characteristics:

- preserves app-level architecture artifacts
- avoids CRM-specific semantics
- allows backend/frontend split when justified

### 6.3 Keep `webapp_crm` as Specialized Template

`webapp_crm` remains valid, but only when positively selected by routing evidence such as:

- CRM
- customer management
- leads
- contacts
- sales pipeline
- account/opportunity domain concepts

CRM must become one specialized project type among several, not the default meaning of Coding Team.

---

## 7. Workflow Strategy

### 7.1 Short-Term Patch

Keep `coding_team_v0` as the only production workflow skeleton, but remove the assumption that its semantic identity is CRM.

There are two acceptable implementation patterns:

#### Option 1. Keep one workflow skeleton, vary project type contracts

- `coding_team_v0` becomes a reusable skeleton
- `workflow_runs.project_type` may be `webapp_crm`, `generic_app`, `single_file_html`, or `generic_coding_task`
- prompt builders and validators become project-type-aware

#### Option 2. Split skeleton from template explicitly

- introduce `coding_team_generic_v1`
- preserve `coding_team_v0` as CRM-specialized legacy workflow
- route new work to generic workflow by default

Recommended patch direction:

- Option 1 first, because it minimizes migration cost
- Option 2 later if governance/reporting needs stronger separation

### 7.2 Router Rule

Routing must follow:

1. Determine whether Coding Team orchestration is needed.
2. Determine whether the task matches a known specialized project type.
3. If specialized match is weak or absent:
   - choose `generic_coding_task`, or
   - emit `clarification_required` if target outputs are too ambiguous

Router must not do:

- `coding` => `webapp_crm`

That mapping is invalid.

---

## 8. Discord `/coder` Patch Behavior

### 8.1 Current Behavior

`/coder` currently hard-overrides:

- `workflow_id = coding_team_v0`
- `project_type = webapp_crm`

### 8.2 Required New Behavior

`/coder` should mean:

- "force Coding Team orchestration"

It should not mean:

- "force CRM template"

Patched `/coder` flow:

1. Force `decision=orchestrated_workflow`.
2. Run a project-type selector over the actual task text.
3. Choose:
   - `single_file_html` for single static file requests
   - `webapp_crm` for positive CRM evidence
   - `generic_app` for app/product style work without CRM evidence
   - `generic_coding_task` as safe default fallback
4. Only emit `clarification_required` when output shape is too ambiguous to create a safe contract.

---

## 9. Project Type Selection Rules

### 9.1 Deterministic Evidence First

Use explicit cues before LLM inference:

- `html`, `index.html`, `landing page`, `single page`, `hello world` => `single_file_html`
- `crm`, `customer`, `lead`, `pipeline`, `contact`, `sales` => `webapp_crm`
- `dashboard`, `tool`, `portal`, `admin`, `prototype`, `web app`, `internal app` => `generic_app`

### 9.2 Generic Fallback Rule

If request is clearly coding work but does not strongly match a known template:

- choose `generic_coding_task`

### 9.3 Clarification Rule

Emit `clarification_required` only when at least one of these is true:

- output medium is unclear
- target path/domain is unclear
- request mixes incompatible deliverable types
- execution constraints are missing and cannot be inferred safely

Clarification must be a last safe fallback, not the default escape hatch.

---

## 10. Prompt and Artifact Contract Impact

### 10.1 PM and Architect Prompt Scripts

Prompt scripts must become project-type-sensitive.

Examples:

- `webapp_crm`
  - retain CRM-specific structure
- `single_file_html`
  - focus on page goal, content, structure, style, static verification
- `generic_coding_task`
  - focus on scope, target paths, deliverables, assumptions, validation plan

### 10.2 Implementation Contracts

Implementation adapters must stop assuming:

- `workspace/sandbox/crm_site/`
- server/client split
- backend API existence

Target paths must come from project type or inferred execution template.

### 10.3 Acceptance and Release Pack

Acceptance suites must be project-type-specific:

- `single_file_html`
  - syntax/static existence/basic preview checks
- `generic_coding_task`
  - minimal verification + artifact completeness
- `webapp_crm`
  - current richer app-style contract

---

## 11. Backward Compatibility

The patch must preserve:

- existing `webapp_crm` canaries
- current `coding_team_v0` workflow skeleton
- current governance references where CRM is explicitly intended

Backward-compatible path:

1. add new project types
2. make router selection explicit
3. keep CRM template unchanged
4. update only default/fallback behavior

---

## 12. Risks

### 12.1 Risk: Generic template becomes too weak

If `generic_coding_task` is underspecified, outputs may become vague.

Mitigation:

- define minimum required artifacts
- require verification plan
- require explicit target paths when available

### 12.2 Risk: Existing CRM regressions

If CRM cues are not strong enough, real CRM requests may downgrade to generic.

Mitigation:

- deterministic CRM keyword rules
- regression canaries for CRM prompts

### 12.3 Risk: Governance policy assumes CRM-only cohort

M6/M7/M8 governance artifacts currently assume `coding_team_v0 / webapp_crm`.

Mitigation:

- keep rollout policy conservative
- stage new project types behind explicit allowlists

---

## 13. Acceptance Criteria

This patch is complete only when all are true:

1. A `/coder` request for a simple static HTML file no longer routes to `webapp_crm`.
2. A positive CRM request still routes to `webapp_crm`.
3. A non-CRM app request routes to `generic_app` or equivalent non-CRM template.
4. An unknown-but-valid coding request routes to `generic_coding_task`, not CRM.
5. Prompt contracts and artifact validators use `project_type` instead of implicit CRM defaults.
6. Existing CRM canaries remain green.

---

## 14. Recommended Patch Sequence

1. Introduce new project types and minimal schemas.
2. Patch router and `/coder` override to select project type explicitly.
3. Generalize prompt builders and execution packets away from CRM-only defaults.
4. Add project-type-specific acceptance suites.
5. Add regression canaries across CRM, generic app, and single-file HTML.

---

## 15. Executive Summary

The system should not treat "Coding Team" as synonymous with "CRM project."

The correct model is:

- Coding Team = orchestration skeleton
- Project Type = task template

This patch restores that separation and adds a safe generic fallback, allowing Coding Team to handle valid non-CRM work without semantic corruption.
