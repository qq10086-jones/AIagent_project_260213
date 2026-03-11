# Worker-Coding Task Contract Note

- Date: 2026-03-11
- Scope: `WC-NEXT-01` initial contract landing
- Status: draft with compatible implementation

---

## 1. Purpose

This note defines the first compatible extension of the worker-coding contract for next-stage beta productization.

The goal is to make coding runs classifiable and reviewable without breaking the current `coding.delegate` flow.

---

## 2. New Optional Payload Fields

`coding.delegate` may now include:

```json
{
  "task_class": "fe_create|fe_modify|be_create|bug_fix|artifact_completion",
  "beta_template_id": "string|null",
  "context_envelope": {
    "max_files": "int|null",
    "max_tokens": "int|null",
    "dependency_depth": "int|null",
    "context_source": "manual|template|automated|null"
  }
}
```

All fields are optional in this phase.

Compatibility rule:

- if omitted, current workflows continue unchanged
- if present, the fields are normalized and persisted into diagnostics / failure memory

---

## 3. Current Behavior

Current compatible landing covers:

- payload pass-through from worker entry into `CodingService.delegateTask`
- normalized `task_contract` block in success diagnostics
- normalized `task_contract` block in failure diagnostics
- `failure_attribution` persisted for failure summaries and coding failure memory

Current phase does not yet enforce:

- template registry lookup
- context envelope hard-stop validation
- task-class-aware routing or runtime policy

Those belong to later `WC-NEXT-*` workstreams.

---

## 4. Failure Attribution v1

Current normalized values:

- `coding_logic_failure`
- `context_failure`
- `verification_failure`
- `infrastructure_failure`

Current mapping is phase/error-code-based and intentionally conservative.

This mapping is sufficient for cohort reporting bootstrap, but may be refined after real beta evidence exists.

---

## 5. Output Contract Addition

`coding.delegate` result diagnostics may now include:

```json
{
  "task_contract": {
    "task_class": "string|null",
    "beta_template_id": "string|null",
    "context_envelope": {
      "max_files": "int|null",
      "max_tokens": "int|null",
      "dependency_depth": "int|null",
      "context_source": "string|null"
    }
  },
  "failure_attribution": "string|null"
}
```

Failure memory entries also carry the same normalized `task_contract` block plus `failure_attribution`.

---

## 6. Non-Scope

This landing does not authorize:

- RAG integration
- automatic context retrieval
- execution isolation redesign
- template registry execution
- forced rejection based on context envelope limits

---

## 7. Next Step

After this compatible contract landing:

1. define the authoritative task-class taxonomy and template registry shape
2. decide where context envelope validation should be enforced
3. use cohort validation to test whether failure attribution is operationally useful
