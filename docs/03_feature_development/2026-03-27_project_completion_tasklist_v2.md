# Project Completion Tasklist (v2)

- Date: 2026-03-27
- Status: READY FOR EXECUTION
- Source Design: `docs/03_feature_development/2026-03-27_project_completion_design_v2.md`
- Revision Note: v2 adds phased GO gate rollout, canary baseline capture, shared rubric publication, perceptual quality checks, and domain pack selection validation

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
- **evaluative terms (shallow, scaffold-only, domain-aligned) lack shared operational definitions** (NEW)
- **no canary baseline exists against which to measure improvement** (NEW)
- **perceptual quality is not evaluated at any stage** (NEW)

---

## 2. P0 Blocking Tasks

## P0-01 Tighten GO Gate (Phased Rollout)

Goal:

- prevent `GO` on scaffold-only product results

### Phase 1: Warning-Only (runs parallel with P0-03)

Tasks:

- implement lightweight product fidelity classifier
- emit `product_fidelity_warning` in QA/release output when scaffold-only patterns detected
- do NOT block workflow; log warning to operator summary
- begin collecting signal on warning frequency and accuracy

Acceptance:

- fidelity warnings appear in run output for scaffold-only results
- no workflow is blocked; warnings are informational

### Phase 2: Blocking (after P0-03 and P0-04 are complete)

Tasks:

- upgrade fidelity warning to fidelity gate
- require fidelity threshold in `go_no_go_result`
- fail or downgrade when product output is obviously placeholder-level
- require reasoning chain in fidelity report

Acceptance:

- scaffold-only frontend/backend outputs cannot reach `GO`
- fidelity report includes per-criterion evidence

Rationale for phased approach:

- Phase 1 provides early visibility into fidelity signal, enabling team feedback loops before enforcement
- Phase 2 enforces only after CI and runtime are stable, avoiding compounded failures

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

## P0-05 Publish Shared Judgment Rubric (NEW)

Goal:

- ensure all evaluative agents and human reviewers use consistent definitions

Tasks:

- formalize the rubric from Design Section 4.2 into a machine-readable reference document
- define operational meanings of: scaffold-only, shallow, domain-aligned, demo-usable, preview-matched
- require all QA and fidelity agents to reference rubric criteria in their outputs

Acceptance:

- rubric document is published and accessible to all agents
- QA outputs reference rubric terms with specific criteria citations

## P0-06 Capture Canary Baseline (NEW)

Goal:

- establish measurable before-state for improvement tracking

Tasks:

- run 3-5 representative Discord canary scenarios using current system (before any fidelity gating)
- record per-run: GO/No-Go result, actual product quality (manual assessment), preview correctness
- compute baseline false-positive rate (GO issued for non-demo-usable output)
- store as `canary_baseline_report.json`

Acceptance:

- baseline false-positive rate is documented
- baseline report is available for comparison in P2-03

---

## 3. P1 Product Completion Tasks

## P1-01 Domain Acceptance Packs

Goal:

- replace generic acceptance with product-type-specific acceptance

### Selection Validation Step (NEW)

Before building packs, validate assumed priority order against real data:

- pull recent Discord intake requests (last 30 days or available history)
- count request frequency by inferred project type
- confirm or re-prioritize the initial set: ecommerce, CRM, document/release management
- document selection rationale in acceptance pack metadata

### Pack Development

Tasks:

- define acceptance pack for ecommerce demo
- define acceptance pack for CRM
- define acceptance pack for document/release management
- each pack includes: required journeys, expected UI components, domain nouns, backend behavior, preview expectations
- **each pack includes minimum perceptual quality expectations** (NEW)
- **each pack includes example pass/fail descriptions** (NEW)

Acceptance:

- PM/QA artifacts reflect domain-specific journeys and terminology
- pack selection rationale is documented against intake data

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
- **require QA to cite shared rubric criteria for every evaluative statement** (NEW)
- **require per-criterion evidence, not summary-only verdicts** (NEW)

Acceptance:

- QA report can distinguish artifact completeness from demo usability
- QA output references rubric and provides traceable reasoning

## P1-04 Improve Preview Quality Evidence

Goal:

- make preview a real product verification step

Tasks:

- capture preview root, mode, and route metadata
- add preview smoke checks aligned to project type
- include preview evidence in release notes

Acceptance:

- preview artifact explains exactly what product was served

## P1-05 Add Perceptual Quality Scoring (NEW)

Goal:

- ensure visual/interaction quality is evaluated as a dimension of product completeness

Tasks:

- define perceptual quality criteria: layout intentionality, placeholder text detection, interactive affordance visibility
- integrate scoring into product fidelity report
- require `perceptual_quality.score >= mid` for `demo_usable` classification
- add `visually_incomplete` as a fidelity result class

Acceptance:

- fidelity report includes `perceptual_quality` object with per-criterion results
- outputs with default/unstyled UI receive appropriate fidelity downgrade

---

## 4. P2 Strengthening Tasks

## P2-01 Product Fidelity Report

Tasks:

- add `product_fidelity_report.json`
- compute domain alignment and scaffold-risk score
- **include perceptual quality score** (NEW)
- **include reasoning chain for each classification** (NEW)
- integrate into release summary

## P2-02 Observability Improvement

Tasks:

- add dashboard/report dimensions for:
  - workflow success rate
  - GO rate
  - fidelity pass rate
  - preview mismatch count
  - shallow-output count
  - **perceptual quality distribution** (NEW)
  - **fidelity warning-to-block conversion rate** (NEW)

## P2-03 Real-Run Comparison Pack

Tasks:

- **retrieve baseline from P0-06** (NEW)
- run after Discord canary scenarios with fidelity gating active
- compare artifact fidelity, QA truthfulness, and preview correctness against baseline
- **compute delta in false-positive rate** (NEW)
- publish one summary note

---

## 5. Recommended Execution Order

### Phase 1: Foundation & Early Signal (parallel tracks)

| Order | Task | Track | Notes |
|-------|------|-------|-------|
| 1a | P0-03 Fix automated regressions | Infrastructure | Unblocks CI trust |
| 1b | P0-01 Phase 1 (warning-only fidelity) | Quality Signal | Provides visibility without blocking |
| 1c | P0-05 Publish shared rubric | Alignment | Enables consistent evaluation from the start |
| 1d | P0-06 Capture canary baseline | Measurement | Must complete before any gating changes |

### Phase 2: Infrastructure Hardening

| Order | Task | Notes |
|-------|------|-------|
| 2 | P0-04 Stabilize runtime config governance | Prerequisite for reliable gating |
| 3 | P0-02 Add preview mismatch detection | |

### Phase 3: Gate Enforcement

| Order | Task | Notes |
|-------|------|-------|
| 4 | P0-01 Phase 2 (blocking fidelity gate) | Only after P0-03 and P0-04 are green |

### Phase 4: Product Depth

| Order | Task | Notes |
|-------|------|-------|
| 5 | P1-01 Domain acceptance packs | Start with selection validation step |
| 6 | P1-02 Upgrade PM specificity | |
| 7 | P1-03 Upgrade QA journey depth | |
| 8 | P1-04 Improve preview evidence | |
| 9 | P1-05 Perceptual quality scoring | |

### Phase 5: Reporting & Validation

| Order | Task | Notes |
|-------|------|-------|
| 10 | P2-01 Add product fidelity report | |
| 11 | P2-02 Expand observability | |
| 12 | P2-03 Run before/after real canaries | Compare against P0-06 baseline |

---

## 6. Definition of Done

This tasklist is complete when:

- a shallow generated app cannot be marked `GO`
- preview mismatch is no longer silently accepted
- at least three project types use domain-specific acceptance packs
- QA includes journey-based evidence with per-criterion reasoning
- live Discord canaries show measurably lower false-positive success **relative to the documented baseline** (updated)
- test suites are green and stable
- **all evaluative agents reference the shared judgment rubric** (NEW)
- **product fidelity report includes perceptual quality scoring** (NEW)
- **canary baseline is documented and comparison delta is published** (NEW)
- **domain pack selection is validated against actual intake data** (NEW)

---

## 7. Risk Register (NEW)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fidelity gate blocks too aggressively, reducing throughput | High | Phase 1 warning-only rollout; tune thresholds before enforcement |
| Shared rubric is published but not adopted by agents | Medium | Require rubric citations in QA output; validate in code review |
| Domain pack selection is based on wrong assumptions | Medium | Validate against intake data before building packs (P1-01 step 1) |
| Perceptual quality scoring is too subjective | Medium | Define concrete criteria (placeholder text, layout structure); iterate rubric |
| Canary baseline is not captured before changes ship | High | P0-06 is marked blocking and parallel with Phase 1 |
| Team loses momentum during infrastructure-only Phase 2 | Low-Medium | Phase 1b (warning-only fidelity) provides visible progress signal throughout |
