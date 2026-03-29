# Evaluation Standards Reference

- Version: 2026-03-28.v1
- Source Design: `docs/03_feature_development/2026-03-27_project_completion_design_v2.md`
- Machine-readable rubric: `orchestrator/configs/product_fidelity_rubric.json`
- Status: ACTIVE — all evaluative agents and human reviewers must reference this document

---

## Purpose

This document operationalizes the judgment terms used across agents, QA reports, fidelity classifiers, and human reviewers. It prevents drift in interpretation and enables post-hoc review when evaluations conflict.

**Every evaluative statement must cite a criterion from this document.**

---

## 1. Classification Hierarchy

When multiple conditions are true, apply the first matching classification (highest priority first):

| Priority | Classification | Trigger |
|----------|---------------|---------|
| 1 | `preview_mismatch` | Preview root points to a sandbox unrelated to the requested project type |
| 2 | `artifact_complete_but_shallow` | Implementation files have placeholder text, insufficient depth, or QA is scaffold-only |
| 3 | `domain_misaligned` | Implementation uses generic CRUD nouns (entity/item/record) instead of domain vocabulary |
| 4 | `visually_incomplete` | All functional checks pass but implementation contains no UI rendering signals |
| 5 | `demo_usable` | All criteria in Section 2 are satisfied |

---

## 2. `demo_usable` — All Requirements

A product output is `demo_usable` when ALL of the following hold:

| Criterion | Operational Definition |
|-----------|----------------------|
| `frontend_depth` | `impl/fe_changes/app.js` has ≥ 8 non-empty lines and ≥ 180 bytes |
| `backend_depth` | `impl/be_changes/server.js` has ≥ 8 non-empty lines and ≥ 180 bytes |
| `placeholder_free` | No placeholder/scaffold markers in impl or QA artifacts (`stub`, `placeholder`, `scaffold`, `pending human review`, `auto-generated`, `sample item`, `sample customer`, `todo`, `lorem ipsum`) |
| `qa_evidence_not_scaffold_only` | `verify/qa_report.json` exists and is not entirely composed of pending-review warnings |
| `preview_matched` | Preview root is not a mismatched legacy sandbox |
| `domain_not_generic_crud` | Implementation does not use `entity`, `record`, or `item` as primary domain vocabulary |
| `perceptual_quality_minimum` | Frontend contains at least one UI rendering signal (see Section 4) |

---

## 3. Term Definitions

### `scaffold-only`
Output contains correct file structure and route stubs but fewer than 2 implemented user interactions. The implementation satisfies schema and file-shape requirements without satisfying any meaningful user journey.

**Signal in report:** `classification=artifact_complete_but_shallow` with `frontend_depth.pass=false` or `backend_depth.pass=false`

### `shallow`
Primary journey exists but critical steps are hardcoded, mocked, or non-functional. Sample records are returned from handlers without persistence; forms serialize payloads without executing a journey.

**Signal in report:** `classification=artifact_complete_but_shallow` with `qa_evidence_not_scaffold_only.pass=false`

### `domain-aligned`
UI labels, API endpoints, and data models use vocabulary specific to the requested product type. Generic CRUD nouns like `entity`, `item`, `record` are absent from primary journey artifacts.

**Signal in report:** `domain_not_generic_crud.pass=true` and, when a domain acceptance pack is loaded, `domain_noun_present.pass=true`

### `demo-usable`
All criteria in Section 2 are met. See full definition above.

### `preview-matched`
Preview root serves artifacts from the current run's output, not a legacy or shared sandbox. The `preview_source` field in `preview_validation_report.json` is `run_scoped_or_custom`, not `shared_crm_sandbox`.

### `visually_incomplete`
All functional criteria pass but the frontend implementation contains no UI rendering signals: no HTML elements (`<button`, `<form`, `<input`), no event handlers (`onClick`, `submit`), no rendering patterns (`render`, `return (`, `createElement`), no async interactions (`fetch(`), no routing (`router`, `route`). The implementation is pure utility/data code with no visual layer.

---

## 4. Perceptual Quality Scoring

| Score | Conditions |
|-------|-----------|
| `low` | No FE file, OR placeholder markers present, OR fewer than 8 non-empty lines, OR zero UI rendering signals |
| `mid` | Has UI rendering signals and ≥ 8 non-empty lines but fewer than 18 |
| `high` | Has UI rendering signals and ≥ 18 non-empty lines |

**Minimum for `demo_usable`:** `mid`

UI rendering signals are detected via the following patterns (case-insensitive):
`<button`, `<form`, `<input`, `onClick`, `submit`, `render`, `return (`, `createElement`, `fetch(`, `router`, `route`

---

## 5. GO Gate Requirements

`GO` verdict requires ALL of the following:

| Check | Description |
|-------|-------------|
| `artifact_pack_validator` | Artifact pack validation passed |
| `workflow_status` | Workflow manifest status is `succeeded` |
| `step_success` | All workflow steps have `succeeded` status |
| `acceptance_gate` | The `qa_verify` step succeeded |
| `strict_canary_verdict` | Strict canary report verdict is `pass` |
| `strict_canary_missing_artifacts` | Zero missing artifacts in canary report |

**In `blocking` gate mode**, the following additional checks apply:

| Check | Description |
|-------|-------------|
| `product_fidelity_gate` | `product_fidelity_report.classification` must not trigger a warning (`demo_usable` only) |
| `preview_validation_gate` | `preview_validation_report` must not show a mismatch |

**Gate mode** is controlled by `orchestrator.fidelity_gate_mode` in `configs/runtime/runtime_defaults.json`:
- `"warning"` — fidelity issues are surfaced but do not block `GO`
- `"blocking"` — fidelity and preview issues cause `NO_GO`

---

## 6. Domain Acceptance Pack Usage

When a `domain acceptance pack` is available for the project type (see `orchestrator/configs/domain_acceptance_packs/`), two additional criteria are evaluated:

| Criterion | Description |
|-----------|-------------|
| `domain_noun_present` | At least 2 domain-specific nouns from the pack appear in the implementation (substring match, case-insensitive) |
| `domain_forbidden_nouns_absent` | None of the pack's `forbidden_generic_nouns` appear as whole words in the implementation |

Available packs: `ecommerce`, `crm`, `document_release`

---

## 7. How to Reference This Document

All evaluative outputs (QA reports, fidelity reports, go/no-go results) must include traceable reasoning. Each criterion should appear explicitly:

```json
{
  "criterion": "placeholder_free",
  "evidence": "matched patterns: \\bsample item\\b",
  "pass": false
}
```

When disputing an evaluation, cite the specific criterion and evidence from this document.

---

## 8. Recalibration Policy

This document should be reviewed and updated:
- Quarterly
- After any incident where a significant false-positive or false-negative is discovered
- When a new project type domain pack is added

Recalibration history: `docs/03_feature_development/evaluation_standards_changelog.md`
