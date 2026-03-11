# Sub Task Worker-Coding Task List

- Date: 2026-03-11
- Status: DRAFT FOR REVIEW
- Scope: next-stage worker-coding capability uplift after M9 closeout and next-stage hardening completion

> Note: this file is now a simplified companion note. The authoritative worker-coding execution plan is:
> `docs/01_design/coding/260311/OpenClaw_Nexus_Worker_Coding_Task_List_v2.md`

---

## 1. Decision

Current worker-coding uplift is classified as:

`MAINLINE TYPE A`

Interpretation:

- this is not a side branch
- this is not a new subsystem program
- this is the next productization layer for the existing North Star coding workflow

---

## 2. Task List

### P0

#### WC-NEXT-01: Task-Class Contract Definition

**Status**

`TODO`

**Task Name**

`WC-NEXT-01 Task-Class Contract Definition`

**Pipeline Node**

`OpenClaw Orchestration -> Coding Team Workflow`

**Task Type**

`Type A`

**Upstream Dependency**

- M9 guardrails are landed
- release gate is passing
- worker structural governance is complete

**Goal**

Define the controlled coding task classes and make them explicit in the worker-coding contract.

**Deliverables**

- task-class taxonomy
- payload extension proposal for `task_class`
- reporting field proposal for class-aware evidence
- short contract note for orchestrator and worker ownership boundaries

**Non-Scope Declaration**

- no model-policy redesign
- no new provider work
- no user-facing UI redesign

**Acceptance Criteria**

- task classes are explicit and documented
- task-class metadata does not break current workflow compatibility
- PM/QA can distinguish coding evidence by task class

**LLM Role**

`none`

---

#### WC-NEXT-02: Beta Template Registry

**Status**

`TODO`

**Task Name**

`WC-NEXT-02 Beta Template Registry`

**Pipeline Node**

`Human Input -> OpenClaw Orchestration -> Coding Team Workflow`

**Task Type**

`Type A`

**Upstream Dependency**

- task-class definition exists

**Goal**

Create a governed registry of approved internal beta coding templates so first-use success does not depend on tribal knowledge.

**Deliverables**

- template registry format
- initial templates for:
  - FE create
  - FE modify
  - BE create
  - bug fix
- documentation for required fields:
  - expected artifacts
  - target path hints
  - verification guidance
  - summary expectations

**Non-Scope Declaration**

- no broad prompt-library system
- no open-ended freeform agent marketplace

**Acceptance Criteria**

- internal testers can start a bounded coding task from a documented template
- template fields align with current workflow contracts
- each template maps to one task class

**LLM Role**

`none`

---

### P1

#### WC-NEXT-03: Multi-Class Cohort Validation

**Status**

`TODO`

**Task Name**

`WC-NEXT-03 Multi-Class Cohort Validation`

**Pipeline Node**

`Coding Team Workflow -> Artifacts`

**Task Type**

`Type A`

**Upstream Dependency**

- task-class contract exists
- beta templates exist

**Goal**

Replace single-scenario confidence with a controlled validation cohort covering multiple coding task classes.

**Deliverables**

- one cohort validation plan
- one machine-readable summary artifact grouped by task class
- curated sample tasks covering at least:
  - FE create
  - FE modify
  - BE create
  - bug fix

**Non-Scope Declaration**

- no broad benchmark framework
- no public leaderboard

**Acceptance Criteria**

- validation can show success/failure by task class
- common failure reasons are visible
- project does not rely on one demo scenario for readiness claims

**LLM Role**

`none`

---

#### WC-NEXT-04: User-Facing Result Quality Hardening

**Status**

`TODO`

**Task Name**

`WC-NEXT-04 User-Facing Result Quality Hardening`

**Pipeline Node**

`Coding Team Workflow -> Artifacts -> Human Review`

**Task Type**

`Type A`

**Upstream Dependency**

- cohort validation reveals current result-quality gaps

**Goal**

Improve the readability and actionability of coding outcomes for internal beta users and reviewers.

**Deliverables**

- result summary quality rubric
- failure-summary quality improvements
- artifact completeness checks where current outputs are noisy or ambiguous

**Non-Scope Declaration**

- no new chat product surface
- no cosmetic-only formatting work without evidence benefit

**Acceptance Criteria**

- successful runs produce short, reviewable summaries
- failed runs state what blocked progress and what was attempted
- QA can inspect outcomes without raw log archaeology

**LLM Role**

`none`

---

### P2

#### WC-NEXT-05: Controlled Beta Operations Metrics

**Status**

`TODO`

**Task Name**

`WC-NEXT-05 Controlled Beta Operations Metrics`

**Pipeline Node**

`Human Input -> Coding Team Workflow -> Artifacts`

**Task Type**

`Type B`

**Upstream Dependency**

- multi-class cohort validation exists

**Goal**

Track whether worker-coding is becoming more usable for internal beta, not just more technically correct.

**Deliverables**

- metric definitions for:
  - first-use success rate
  - verification pass rate
  - artifact completeness rate
  - dominant failure categories
- one short reporting note for weekly review

**Non-Scope Declaration**

- no dashboard platform project
- no analytics subsystem expansion

**Acceptance Criteria**

- weekly beta quality can be discussed with evidence
- product and engineering can prioritize using the same metrics

**LLM Role**

`none`

---

## 3. Recommended Order

1. `WC-NEXT-01` Task-Class Contract Definition
2. `WC-NEXT-02` Beta Template Registry
3. `WC-NEXT-03` Multi-Class Cohort Validation
4. `WC-NEXT-04` User-Facing Result Quality Hardening
5. `WC-NEXT-05` Controlled Beta Operations Metrics

Reasoning:

- define the product boundary first
- reduce first-use ambiguity second
- validate across classes third
- improve user-facing quality using evidence rather than guesswork
- add lightweight operational metrics after the workflow shape is stable

---

## 4. Exit Standard

This sub-program should be considered ready for closeout review when:

- worker-coding supports a governed multi-class beta cohort
- first-use task entry is documented and repeatable
- evidence is no longer anchored to one narrow scenario
- PM, QA, and Architect can review capability quality using shared artifacts
