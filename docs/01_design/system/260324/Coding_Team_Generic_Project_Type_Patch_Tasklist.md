# Coding Team Generic Project Type Patch Tasklist

- Date: 2026-03-24
- Scope: Generic project-type support for Coding Team
- Status: Draft engineering task list

---

## Goal

Remove the structural assumption that `coding_team_v0` implies `webapp_crm`, while preserving the Coding Team orchestration workflow and existing CRM behavior.

---

## Milestone P0: Registry and Contract Decoupling

### WS-01 Project Type Inventory

- [ ] T01: Add `generic_coding_task` to `configs/registry/capability_registry.json`
  - Owner: Architect
  - DoD: project type exists with required artifacts, acceptance suite, and no CRM-specific semantics

- [ ] T02: Add `single_file_html` to `configs/registry/capability_registry.json`
  - Owner: Architect
  - DoD: project type exists with static-file-oriented required artifacts and acceptance suite

- [ ] T03: Add `generic_app` to `configs/registry/capability_registry.json`
  - Owner: Architect
  - DoD: project type exists for non-CRM app work with reusable app-level artifacts

- [ ] T04: Update registry schema and validator to accept the new project types
  - Owner: Architect
  - DoD: registry validation passes and rejects malformed project-type definitions

### WS-02 Workflow Semantics Cleanup

- [ ] T05: Document `coding_team_v0` as a reusable workflow skeleton rather than a CRM identity
  - Owner: PM/Architect
  - DoD: workflow comments and design docs no longer describe `coding_team_v0` as CRM-only by default

- [ ] T06: Remove implicit fallback code paths that derive `webapp_crm` from `coding` alone
  - Owner: Backend
  - DoD: router and runtime defaults no longer silently coerce generic coding work into CRM

---

## Milestone P1: Router and `/coder` Project-Type Selection

### WS-03 Deterministic Project Type Selector

- [ ] T07: Implement deterministic project-type selection helper in vNext router
  - Owner: Backend
  - DoD: helper maps obvious static HTML, CRM, and generic app requests correctly

- [ ] T08: Add safe generic fallback rule for unmatched coding requests
  - Owner: Backend
  - DoD: unmatched coding requests route to `generic_coding_task`

- [ ] T09: Add clarification rule for truly ambiguous coding requests
  - Owner: Backend
  - DoD: router emits `clarification_required` only when output contract cannot be inferred safely

### WS-04 Discord `/coder` Override Fix

- [ ] T10: Replace hardcoded `/coder -> webapp_crm` override with `/coder -> orchestrated_workflow + project_type selection`
  - Owner: Backend
  - DoD: `/coder` forces Coding Team orchestration but not CRM specialization

- [ ] T11: Preserve explicit CRM routing for CRM-shaped `/coder` prompts
  - Owner: Backend
  - DoD: CRM prompts still produce `project_type=webapp_crm`

---

## Milestone P2: Prompt Contracts and Step Payload Generalization

### WS-05 PM/Architect Prompt Generalization

- [ ] T12: Add project-type-aware PM prompt behavior for `generic_coding_task`
  - Owner: PM
  - DoD: PM outputs generic scope and acceptance artifacts without CRM domain assumptions

- [ ] T13: Add project-type-aware PM prompt behavior for `single_file_html`
  - Owner: PM
  - DoD: PM outputs page-focused artifacts with static deliverable expectations

- [ ] T14: Add project-type-aware Architect prompt behavior for `generic_app`
  - Owner: Architect
  - DoD: architecture output is app-generic and not CRM-specific

- [ ] T15: Add project-type-aware Architect prompt behavior for `single_file_html`
  - Owner: Architect
  - DoD: architecture output avoids backend/API boilerplate when not needed

### WS-06 Implementation Contract Generalization

- [ ] T16: Remove `workspace/sandbox/crm_site/` as the universal impl target-path default
  - Owner: Backend
  - DoD: target paths derive from project type or execution template

- [ ] T17: Add `single_file_html` execution template and target-path policy
  - Owner: Backend/Frontend
  - DoD: implementation packet can target a static output root cleanly

- [ ] T18: Add `generic_coding_task` execution template
  - Owner: Backend
  - DoD: implementation packet supports minimal repo changes without app/server assumptions

---

## Milestone P3: Acceptance, Release Pack, and Validation

### WS-07 Acceptance Suites

- [ ] T19: Add acceptance suite for `single_file_html`
  - Owner: QA
  - DoD: checks file existence, syntax, and basic preview/static validation

- [ ] T20: Add acceptance suite for `generic_coding_task`
  - Owner: QA
  - DoD: suite validates required artifacts and declared verification evidence

- [ ] T21: Add acceptance suite for `generic_app`
  - Owner: QA
  - DoD: suite validates app-level artifact completeness without CRM-specific requirements

### WS-08 Artifact Pack Rules

- [ ] T22: Make release-pack validator read required artifacts from actual selected `project_type`
  - Owner: Backend
  - DoD: validator no longer assumes CRM artifact topology for all Coding Team runs

- [ ] T23: Update run manifest to capture `project_type_selection_reason`
  - Owner: Backend
  - DoD: run manifest records why the chosen project type was selected

---

## Milestone P4: Governance and Exposure Controls

### WS-09 Rollout Safety

- [ ] T24: Audit `parallel_exposure_policy.json`, `m7_exposure_cohorts.json`, and rollout configs for CRM-only assumptions
  - Owner: Architect
  - DoD: configs explicitly document which project types are allowed and which remain gated

- [ ] T25: Keep new generic project types disabled from production parallel exposure until separate approval
  - Owner: Architect/PM
  - DoD: governance defaults remain fail-closed

---

## Milestone P5: Regression and Evidence

### WS-10 Regression Suite

- [ ] T26: Add router regression for simple HTML request -> `single_file_html`
  - Owner: QA
  - DoD: test fails if request routes to `webapp_crm`

- [ ] T27: Add router regression for CRM request -> `webapp_crm`
  - Owner: QA
  - DoD: CRM cue coverage stays stable

- [ ] T28: Add router regression for generic app request -> `generic_app`
  - Owner: QA
  - DoD: generic app requests avoid CRM downgrade

- [ ] T29: Add router regression for unmatched coding request -> `generic_coding_task`
  - Owner: QA
  - DoD: valid unknown coding requests no longer misroute to CRM

### WS-11 End-to-End Canaries

- [ ] T30: Add live canary for `/coder create an index.html with hello world`
  - Owner: QA
  - DoD: no CRM artifacts are generated; final output matches static HTML contract

- [ ] T31: Add live canary for CRM workflow request
  - Owner: QA
  - DoD: CRM workflow still produces expected CRM artifacts

- [ ] T32: Add live canary for generic app request
  - Owner: QA
  - DoD: workflow completes with non-CRM app artifacts

---

## Delivery Order

1. P0 Registry and contract decoupling
2. P1 Router and `/coder` fix
3. P2 Prompt/step payload generalization
4. P3 Acceptance and release-pack updates
5. P4 Governance tightening
6. P5 Regression and canary evidence

---

## Definition of Done

This patch is done only when:

- `/coder` no longer hardcodes `webapp_crm`
- generic coding requests have a non-CRM project type
- CRM requests still retain CRM behavior
- prompt contracts and validators honor selected `project_type`
- release packs reflect the correct template semantics
- regression tests and canaries cover CRM, generic app, and single-file HTML cases
