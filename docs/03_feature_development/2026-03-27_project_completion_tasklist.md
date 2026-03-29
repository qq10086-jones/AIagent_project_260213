# Project Completion Tasklist

- Date: 2026-03-27
- Status: READY FOR EXECUTION
- Source Design: `docs/03_feature_development/2026-03-27_project_completion_design.md`

---

## 1. Current Progress Snapshot

Completed or materially improved:

- Discord `/coder` intake routing works
- `generic_app` can complete through `coding_team_v0`
- artifact handoffs, QA, release, and preview outputs are generated
- stable runtime lane was recovered for live Discord-to-coder validation

Open quality concerns:

- workflow success can still produce shallow product output
- QA can still emit `GO` without strong journey proof
- preview may still target a legacy/shared sandbox
- orchestrator and worker-coder still have failing automated tests

---

## 2. P0 Blocking Tasks

## P0-01 Tighten GO Gate

Goal:

- prevent `GO` on scaffold-only product results

Tasks:

- add product fidelity classification to QA/release input
- require fidelity threshold in `go_no_go_result`
- fail or downgrade when product output is obviously placeholder-level

Acceptance:

- scaffold-only frontend/backend outputs cannot reach `GO`

## P0-02 Add Preview Mismatch Detection

Goal:

- block misleading preview success

Tasks:

- compare request project type against preview root and preview source
- flag legacy sandbox reuse when mismatched
- surface mismatch in QA and release artifacts

Acceptance:

- preview mismatch is machine-detected and reported

## P0-03 Fix Automated Regressions

Goal:

- restore trustworthy CI/test signals

Tasks:

- fix failing orchestrator `deploy_preview` payload test
- fix worker-coder auth/error classification regression
- rerun orchestrator and worker-coder core suites

Acceptance:

- orchestrator core suite green
- worker-coder core suite green

## P0-04 Stabilize Runtime Config Governance

Goal:

- prevent environment drift from silently degrading quality

Tasks:

- make execution lane defaults explicit
- add startup rejection for conflicting lane/model config
- expose resolved lane/model in run summaries and operator logs

Acceptance:

- runtime defaults are deterministic across restart/redeploy

---

## 3. P1 Product Completion Tasks

## P1-01 Domain Acceptance Packs

Goal:

- replace generic acceptance with product-type-specific acceptance

Tasks:

- define acceptance pack for ecommerce demo
- define acceptance pack for CRM
- define acceptance pack for document/release management

Acceptance:

- PM/QA artifacts reflect domain-specific journeys and terminology

## P1-02 Upgrade PM Artifact Specificity

Goal:

- stop broad app requests from collapsing into generic CRUD language

Tasks:

- revise PM prompt templates for project-type-aware outputs
- require domain nouns and expected key workflows
- require explicit non-goals aligned to the requested app class

Acceptance:

- ecommerce request produces ecommerce-shaped scope and acceptance

## P1-03 Upgrade QA From Scaffold Review To Journey Review

Goal:

- verify real user flow, not just file shape

Tasks:

- add journey checklist schema
- add QA result statuses for shallow output and preview mismatch
- require evidence lines for primary journey completion

Acceptance:

- QA report can distinguish artifact completeness from demo usability

## P1-04 Improve Preview Quality Evidence

Goal:

- make preview a real product verification step

Tasks:

- capture preview root, mode, and route metadata
- add preview smoke checks aligned to project type
- include preview evidence in release notes

Acceptance:

- preview artifact explains exactly what product was served

---

## 4. P2 Strengthening Tasks

## P2-01 Product Fidelity Report

Tasks:

- add `product_fidelity_report.json`
- compute domain alignment and scaffold-risk score
- integrate into release summary

## P2-02 Observability Improvement

Tasks:

- add dashboard/report dimensions for:
  - workflow success rate
  - GO rate
  - fidelity pass rate
  - preview mismatch count
  - shallow-output count

## P2-03 Real-Run Comparison Pack

Tasks:

- run before/after Discord canary scenarios
- compare artifact fidelity, QA truthfulness, and preview correctness
- publish one summary note

---

## 5. Recommended Execution Order

1. P0-03 Fix automated regressions
2. P0-04 Stabilize runtime config governance
3. P0-02 Add preview mismatch detection
4. P0-01 Tighten GO gate
5. P1-01 Domain acceptance packs
6. P1-02 Upgrade PM specificity
7. P1-03 Upgrade QA journey depth
8. P1-04 Improve preview evidence
9. P2-01 Add product fidelity report
10. P2-02 Expand observability
11. P2-03 Run before/after real canaries

---

## 6. Definition of Done

This tasklist is complete when:

- a shallow generated app cannot be marked `GO`
- preview mismatch is no longer silently accepted
- at least three project types use domain-specific acceptance packs
- QA includes journey-based evidence
- live Discord canaries show lower false-positive success
- test suites are green and stable
