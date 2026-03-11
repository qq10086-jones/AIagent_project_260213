# Worker-Coding Cohort Task Matrix

- Date: 2026-03-11
- Scope: `WC-NEXT-04` preparation
- Status: draft matrix for controlled beta validation

---

## 1. Purpose

This matrix defines the first controlled validation cohort for worker-coding.

It exists to prevent:

- readiness claims based on a single demo scenario
- ambiguous success criteria across task classes
- inconsistent operator framing during internal beta validation

---

## 2. Cohort Principles

Each cohort task must:

- map to one `task_class`
- reference one `beta_template_id`
- define explicit target scope
- define expected artifacts
- define expected verification tier
- define likely failure attribution categories

The initial cohort is intentionally small and reviewable.

---

## 3. Initial Cohort Matrix

| Cohort ID | Task Class | Template | Scenario | Validation Focus | Expected Verification Tier |
|-----------|------------|----------|----------|------------------|----------------------------|
| C-FE-01 | `fe_create` | `wc.fe_create.v1` | create a small landing page in an existing app | greenfield page creation | `lint + build` |
| C-FE-02 | `fe_modify` | `wc.fe_modify.v1` | modify an existing page and add one component | repo-aware targeted edit | `lint + type_check + build` |
| C-BE-01 | `be_create` | `wc.be_create.v1` | add one API route or handler | backend contract creation | `lint + unit_test` |
| C-BUG-01 | `bug_fix` | `wc.bug_fix.v1` | repair one scoped defect with stated regression boundary | localization and verification quality | `lint + unit_test + build` |

---

## 4. Task Definitions

### C-FE-01

- Task class: `fe_create`
- Template: `wc.fe_create.v1`
- Goal:
  - create one small marketing or utility page
- Recommended scope:
  - one page
  - one to two supporting components
- Target path pattern:
  - `src/pages/`
  - `src/components/`
- Required artifacts:
  - `impl/fe_patch_bundle.json`
  - `impl/fe_notes.md`
- Primary review question:
  - can the system create a bounded page without drifting into unrelated UI work?
- Likely failure categories:
  - `coding_logic_failure`
  - `verification_failure`

### C-FE-02

- Task class: `fe_modify`
- Template: `wc.fe_modify.v1`
- Goal:
  - change one existing user flow or page section
- Recommended scope:
  - one existing page
  - one additive component or behavior change
- Target path pattern:
  - `src/pages/`
  - `src/components/`
  - `src/features/`
- Required artifacts:
  - `impl/fe_patch_bundle.json`
  - `impl/fe_notes.md`
- Primary review question:
  - can the system modify an existing UI surface without breaking surrounding behavior?
- Likely failure categories:
  - `context_failure`
  - `coding_logic_failure`
  - `verification_failure`

### C-BE-01

- Task class: `be_create`
- Template: `wc.be_create.v1`
- Goal:
  - add one small route, handler, or service behavior
- Recommended scope:
  - one endpoint
  - minimal supporting service logic
- Target path pattern:
  - `src/routes/`
  - `src/controllers/`
  - `src/services/`
- Required artifacts:
  - `impl/be_patch_bundle.json`
  - `impl/be_notes.md`
  - `handoff/be_to_fe.json`
- Primary review question:
  - can the system implement backend work without collapsing into frontend-centric assumptions?
- Likely failure categories:
  - `coding_logic_failure`
  - `context_failure`
  - `verification_failure`

### C-BUG-01

- Task class: `bug_fix`
- Template: `wc.bug_fix.v1`
- Goal:
  - repair one stated defect with explicit regression boundary
- Recommended scope:
  - one issue
  - one to three touched files
- Target path pattern:
  - `src/`
  - `app/`
  - `server/`
- Required artifacts:
  - `impl/bugfix_notes.md`
- Primary review question:
  - can the system localize and fix the right problem rather than produce a plausible but misplaced edit?
- Likely failure categories:
  - `context_failure`
  - `coding_logic_failure`
  - `infrastructure_failure`

---

## 5. Validation Recording Standard

For every cohort run, record:

- `cohort_id`
- `task_class`
- `beta_template_id`
- `verification_tier_target`
- `verification_tier_achieved`
- `result`
- `failure_attribution`
- `operator_note`

Allowed `result` values:

- `pass`
- `fail`
- `partial`

Allowed `failure_attribution` values:

- `coding_logic_failure`
- `context_failure`
- `verification_failure`
- `infrastructure_failure`
- `none`

---

## 6. Non-Scope

This matrix does not yet define:

- automation scripts for cohort execution
- score weighting across task classes
- public benchmark reporting
- context retrieval automation

---

## 7. Next Step

After this matrix is accepted:

1. create one cohort validation plan around these four tasks
2. define machine-readable result format grouped by `cohort_id`
3. run the first internal beta cycle and inspect failure attribution quality
