# Worker-Coding Cohort Task Matrix v2

- Date: 2026-03-11
- Scope: post-v1 recovery next-harder validation slice
- Status: draft for governed review

---

## 1. Purpose

This matrix defines the next worker-coding cohort after the v1 baseline four-case cohort was recovered to `4 pass / 0 fail / 0 partial`.

It exists to prevent:

- repeating the already-closed v1 slice
- over-claiming readiness from only simple single-surface tasks
- widening into unconstrained autonomy without governance

---

## 2. v2 Design Rules

Each v2 task must still:

- map to one existing `task_class`
- reference one existing `beta_template_id`
- remain deterministic and reviewable
- keep target scope bounded enough for artifact-based QA review

Compared with v1, v2 must increase difficulty through:

- multi-file FE change shape
- tighter BE regression expectations
- at least one FE/BE contract-linked scenario
- at least one bug-fix case with higher ambiguity than the v1 baseline

---

## 3. Proposed v2 Cohort Matrix

| Cohort ID | Task Class | Template | Scenario | Validation Focus | Expected Verification Tier |
|-----------|------------|----------|----------|------------------|----------------------------|
| C-FE-03 | `fe_modify` | `wc.fe_modify.v1` | modify an existing dashboard page across multiple files and extract one shared UI component | multi-file frontend modification with bounded shared-component refactor | `lint + type_check + build` |
| C-FE-04 | `fe_create` | `wc.fe_create.v1` | create a scoped feature page with one supporting component and route-level wiring in the existing app | bounded frontend creation with light integration surface | `lint + build` |
| C-BE-02 | `be_create` | `wc.be_create.v1` | add one API route with service logic and update or create the nearest unit test boundary | backend creation with explicit regression surface and contract handoff | `lint + unit_test` |
| C-BUG-02 | `bug_fix` | `wc.bug_fix.v1` | repair a frontend-backend contract mismatch with explicit regression boundaries on both data shape and visible UI behavior | cross-layer bug localization without widening into unconstrained autonomy | `lint + unit_test + build` |

---

## 4. Review Intent Per Case

### C-FE-03

- difficulty increase:
  - no longer a one-page localized tweak
  - requires touching existing UI plus one extracted shared component
- review question:
  - can worker-coding make a bounded multi-file FE change without drifting into unrelated page cleanup?

### C-FE-04

- difficulty increase:
  - still a create task, but no longer isolated to a leaf page
  - requires lightweight route or navigation integration
- review question:
  - can worker-coding add a new UI surface without faking completion through disconnected page scaffolding?

### C-BE-02

- difficulty increase:
  - must cover route plus service logic plus closest test boundary
  - handoff quality matters, not only code emission
- review question:
  - can worker-coding land backend behavior with a realistic regression boundary instead of a route-only demo edit?

### C-BUG-02

- difficulty increase:
  - defect spans data contract and visible behavior
  - requires the system to localize the real mismatch rather than patch symptoms
- review question:
  - can worker-coding repair a cross-layer defect while staying inside a reviewable fix boundary?

---

## 5. Governance Constraints

v2 still does not authorize:

- open-ended repository-wide search-and-rewrite
- multi-issue batching
- provider expansion
- unguided FE+BE full-product implementation

The intent is harder validation, not broader autonomy.

---

## 6. Exit Signal For v2 Draft Acceptance

This v2 cohort definition should be accepted when:

1. QA agrees each case is materially harder than v1 but still reviewable.
2. architecture review agrees the scope does not exceed current worker-coding governance.
3. the machine-readable plan is validated and can be executed without changing the cohort schema.
