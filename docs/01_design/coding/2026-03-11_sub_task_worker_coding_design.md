# Sub Task Worker-Coding Design

## Date
2026-03-11

## Status
Draft for review

## Why This Document

The project has completed the current worker-coder hardening baseline:

- guardrails landed
- release gate passed
- runtime path consistency was hardened
- worker lifecycle consistency was fixed
- structural decomposition completed

The next problem is no longer architecture survival.

The next problem is productized usefulness:

`can internal beta users reliably use coding through Nexus across multiple bounded task classes`

This document defines that next-stage sub-program.

---

## 1. Problem Statement

Current worker-coding is technically available, but product success still depends on operator knowledge and controlled prompting.

This creates three gaps:

1. task-class coverage is not yet formally defined
2. first-use beta success is not yet treated as a governed capability target
3. validation evidence exists at workflow level, but not yet as a multi-class coding quality cohort

If unaddressed, the system risks two bad outcomes:

- appearing usable in isolated demos but failing real beta onboarding
- overfitting to one scenario such as simple webpage generation

---

## 2. Design Goal

Promote worker-coding from:

`hardened execution component`

to:

`controlled beta-ready coding capability`

The capability must support a bounded but meaningful set of task classes without collapsing into one narrow demo flow.

---

## 3. In Scope

- define controlled coding task classes
- standardize beta entry contracts for each class
- improve verification defaults and evidence quality
- improve task-result readability for internal users
- create repeatable cohort validation for worker-coding
- tighten execution contracts only where needed for productized use

---

## 4. Out of Scope

- full autonomous product-building
- open-ended internet-dependent coding
- broad provider marketplace support
- unbounded repo bootstrap / dependency installation
- replacing orchestrator governance with direct agent freedom

---

## 5. North Star Mapping

This sub-program directly supports:

`Human Input -> Discord Gateway -> OpenClaw Orchestration -> Coding Team Workflow -> Artifacts`

Specifically, it improves the last three segments:

- task enters coding workflow with clearer task-class framing
- worker-coder executes with higher first-pass reliability
- artifacts and summaries become more usable for internal beta review

---

## 6. Controlled Task Classes

Worker-coding should be explicitly designed around these initial task classes:

### TC-01 FE Create

Example:

- create a simple landing page
- create a small dashboard page in an existing app

Purpose:

- validate greenfield UI output

### TC-02 FE Modify

Example:

- update an existing page
- add a component
- refine layout, copy, or interaction

Purpose:

- validate real repo edit workflows

### TC-03 BE Create

Example:

- add a small API route
- add a service function or controller behavior

Purpose:

- prevent frontend-only roadmap narrowing

### TC-04 Bug Fix

Example:

- repair a scoped defect
- correct validation or state handling

Purpose:

- validate precision + verification quality

### TC-05 Artifact Completion

Example:

- fill expected workflow artifact gaps
- repair a broken handoff artifact bundle

Purpose:

- protect orchestrated workflow continuity

---

## 7. Capability Layers

### Layer A: Beta Entry Contract

Each task class should have a stable entry template:

- task framing
- expected artifacts
- allowed target paths
- verification command guidance
- failure summary expectations

This is the product boundary between the user task and the coding engine.

### Layer B: Execution Reliability

Worker-coder must keep:

- scoped write control
- deterministic artifact scaffold
- static checks
- verification command execution
- retry and failure memory
- single-finalization lifecycle behavior

This layer is largely present and should be reused, not redesigned.

### Layer C: Beta Reviewability

Outputs must be understandable to PM/QA/internal users:

- short summary of what changed
- files changed
- verification outcome
- explicit failure cause when unsuccessful
- structured artifacts for inspection

### Layer D: Cohort Validation

The system needs a beta matrix proving capability by task class, not just one successful workflow sample.

---

## 8. Design Principles

### Principle 1: Avoid Scenario Narrowing

No single scenario may define the roadmap.

`build a webpage` is a sample, not the whole product.

### Principle 2: Productize the Contract, Not Just the Engine

A stronger worker is insufficient if users still need tribal knowledge to use it correctly.

### Principle 3: Preserve Governance

Coding capability uplift must not bypass target paths, verification, artifact contracts, or release evidence.

### Principle 4: Optimize for First-Use Success

Internal beta users should be able to submit a bounded coding task and receive a reviewable outcome without manual system debugging.

### Principle 5: Measure by Task Class

Quality must be assessed separately for FE create, FE modify, BE create, bug fix, and artifact completion.

---

## 9. Proposed Interfaces

### 9.1 Task-Class Metadata

Each coding task should optionally include:

```json
{
  "task_class": "fe_create|fe_modify|be_create|bug_fix|artifact_completion",
  "beta_template_id": "string|null"
}
```

Purpose:

- improve routing/reporting clarity
- make validation cohort results comparable
- avoid free-text-only classification downstream

### 9.2 Beta Template Registry

Introduce a small registry for approved beta templates.

Each template should define:

- task class
- expected artifacts
- recommended verification command
- target path hints
- output summary expectations

This registry is a governance artifact, not a prompt toy.

### 9.3 Cohort Validation Summary

Validation should aggregate by task class:

- runs attempted
- success count
- verification pass count
- common failure reasons
- artifact completeness rate

---

## 10. Delivery Strategy

The next-stage sequence should be:

1. define the task-class contract
2. define beta templates
3. create cohort validation for multiple task classes
4. improve result readability and failure reporting where needed
5. expand only after evidence is stable

---

## 11. Success Criteria

This sub-program is successful when:

- worker-coding is governed as a product capability, not just a worker module
- at least four controlled task classes are validated
- beta task entry becomes easier and more repeatable
- success/failure boundaries are observable by class
- no new subsystem sprawl is introduced

---

## 12. Explicit Non-Goals

This document does not authorize:

- general autonomous engineering expansion
- replacing the existing workflow with direct LLM execution
- massive provider/runtime diversification
- broad open-ended code generation without verification
