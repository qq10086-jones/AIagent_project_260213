# Worker-Coding Cohort Result Format

- Date: 2026-03-11
- Scope: `WC-NEXT-04` preparation
- Status: draft

---

## 1. Purpose

This note defines the machine-readable result format for the first worker-coding cohort runs.

It exists so that:

- cohort outcomes are comparable across task classes
- failure attribution is aggregated consistently
- PM / QA / Architect review can use one shared artifact shape

---

## 2. Artifact Path

Recommended artifact root:

`orchestrator/artifacts/validation/worker_coding_cohort/<cohort_run_id>/worker_coding_cohort_result.json`

---

## 3. Required Top-Level Fields

```json
{
  "cohort_run_id": "string",
  "generated_at": "ISO timestamp",
  "summary": {
    "total_runs": "number",
    "pass_count": "number",
    "fail_count": "number",
    "partial_count": "number"
  },
  "results": []
}
```

Schema authority:

- `orchestrator/contracts/worker_coding_cohort_result.schema.json`

---

## 4. Required Per-Result Fields

Each result item must include:

```json
{
  "cohort_id": "string",
  "task_class": "string",
  "beta_template_id": "string",
  "verification_tier_target": "string",
  "verification_tier_achieved": "string",
  "result": "pass|fail|partial",
  "failure_attribution": "coding_logic_failure|context_failure|verification_failure|infrastructure_failure|none"
}
```

Optional but recommended:

- `operator_note`
- `run_id`
- `task_id`
- `files_changed_count`
- `artifact_completeness`

---

## 5. Summary Expectations

At minimum, the summary block should support:

- total cohort size
- pass/fail/partial counts
- grouping by task class in downstream review
- grouping by failure attribution in downstream review

This first phase does not require full automatic aggregation by task class inside the summary block.

That can be added after the first validation cycle.

---

## 6. Non-Scope

This format does not yet define:

- automatic score weighting
- public benchmark compatibility
- ranking across providers
- long-term analytics storage

---

## 7. Next Step

After this format is accepted:

1. add one validation helper for the cohort result schema
2. run first internal cohort and write result artifacts in this format
3. review whether per-class aggregation fields should become mandatory
