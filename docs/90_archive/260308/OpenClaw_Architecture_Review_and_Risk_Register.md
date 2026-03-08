# OpenClaw Nexus --- Architecture Review & System Risk Register

Date: 2026-03

------------------------------------------------------------------------

# Part I --- Architecture Review Note

## 1. Review Context

Documents reviewed:

-   OpenClaw_Nexus_Design_Document_v3.md
-   OpenClaw_Nexus_Engineering_Task_List_M4.md

Focus areas:

-   Coding Team Workflow
-   LLM Dispatcher
-   Brain Router
-   M4 Engineering Plan

Primary review goals:

1.  Identify architectural risks
2.  Distinguish **M4 must‑fix issues** vs **future improvements**
3.  Provide clear engineering decisions

------------------------------------------------------------------------

## 2. Executive Summary

The architecture is **logically coherent and engineering‑oriented**, and
the M4 milestone correctly focuses on:

-   Deterministic execution
-   Observability
-   Validation
-   Running a complete Coding Team workflow

However, real deployment introduces several risks.

  Risk Category                    Severity   Must Fix in M4
  -------------------------------- ---------- ----------------
  Local LLM resource constraints   High       Yes
  LLM call stability               High       Yes
  Schema only validating format    Medium     Yes
  Router false negatives           Medium     Yes
  Full file output token cost      Medium     No
  Sequential workflow efficiency   Medium     No

Recommendation:

-   **4 issues adopt immediately**
-   **2 issues planned for future versions**

------------------------------------------------------------------------

# 3. Detailed Review

## 3.1 Local 32B Model Resource Risk

Current configuration assigns:

backend → deepseek-r1:32b\
frontend → deepseek-r1:32b\
qa → deepseek-r1:32b\
release → deepseek-r1:32b

Typical local machine:

GPU: RX 7900 XTX\
VRAM: 24GB\
RAM: 32GB

32B models (4‑bit quantization):

VRAM ≈ 19‑21GB

Risks:

-   large handoff contexts
-   multi‑agent calls
-   worker concurrency

Possible failures:

-   OOM
-   extreme latency
-   Ollama crashes

### Decision

Adopt mitigation.

### Action

Introduce model fallback:

primary_model: deepseek-r1:32b\
secondary_model: qwen2.5-coder:7b

Downgrade triggers:

-   context overflow
-   latency threshold
-   OOM detection

------------------------------------------------------------------------

## 3.2 Fail‑Fast Policy Fragility

Current:

fallback_policy: fail_fast

Any LLM error terminates workflow.

Real causes of failures include:

-   Ollama timeout
-   temporary GPU overload
-   API instability

### Decision

Adopt improvement.

### Strategy

Separate policies:

Transport layer → retry\
Provider layer → fail fast

### Action

Add retry:

retry_policy: strategy: exponential_backoff retries: 3 delay: 2s

------------------------------------------------------------------------

## 3.3 Schema Validation Blind Spot

Schema validation only ensures:

-   valid JSON
-   required fields
-   correct types

But cannot ensure:

-   correct API logic
-   requirement alignment

Example:

Backend produces a valid API schema but with incorrect semantics.

Frontend consumes it anyway.

### Decision

Adopt improvement.

### Action

Introduce **semantic validation** at QA stage.

Two layers:

Deterministic checks

-   artifacts exist
-   run instructions present
-   tests present

Semantic checks

-   API contracts vs frontend usage
-   requirement alignment

------------------------------------------------------------------------

## 3.4 Brain Router Unknown Intent Handling

Current behavior:

Unknown intent → chat mode

Risk:

User requests execution but system replies conversationally.

### Decision

Adopt improvement.

### Action

Add confirmation step.

Example prompt:

"This looks like a development task. Should I start the Coding Team
workflow?"

------------------------------------------------------------------------

## 3.5 Full File Output Strategy

Current design requires:

Agents output **complete files** rather than diffs.

Pros:

-   avoids baseline mismatch
-   easier sandbox execution

Cons:

-   high token usage
-   large file truncation risk

### Decision

Partial acceptance.

Maintain for M4 but plan evolution.

### Future Roadmap

M5 introduce:

-   structured diff
-   AST patching

------------------------------------------------------------------------

## 3.6 Sequential Workflow

Current pipeline:

PM → Architect → Backend → Frontend → QA → Release

Parallelism theoretically possible between BE and FE.

### Decision

Defer change.

Reason:

M4 priority is **stable deterministic pipeline**.

Parallel DAG orchestration belongs to a later engine version.

------------------------------------------------------------------------

# 4. Final Decision Table

  Issue                        Decision   Priority
  ---------------------------- ---------- ----------
  32B model resource risk      Adopt      High
  LLM retry mechanism          Adopt      High
  Schema semantic validation   Adopt      Medium
  Router confirmation step     Adopt      Medium
  Full file output             Partial    Low
  Sequential workflow          Defer      Low

------------------------------------------------------------------------

# Part II --- System Risk Register

This register lists **structural risks typical in AI‑agent engineering
systems**.

------------------------------------------------------------------------

## Risk 1 --- Context Explosion

Agents pass large artifacts between steps.

Risks:

-   excessive token usage
-   model truncation
-   latency spikes

Mitigation:

-   strict context budgets
-   artifact referencing instead of embedding
-   chunked context loading

------------------------------------------------------------------------

## Risk 2 --- Agent Drift

Agents may deviate from original task scope across steps.

Example:

Frontend generates UI beyond requirement scope.

Mitigation:

-   scope_constraints artifacts
-   QA semantic validation
-   PM requirement checksum

------------------------------------------------------------------------

## Risk 3 --- Model Non‑Determinism

LLM responses vary across runs.

Risks:

-   inconsistent artifacts
-   unstable workflows

Mitigation:

-   temperature control
-   structured prompts
-   deterministic validators

------------------------------------------------------------------------

## Risk 4 --- Pipeline Garbage Propagation

Incorrect output from one step contaminates downstream steps.

Example:

Incorrect API contract leads to invalid UI generation.

Mitigation:

-   validation gates between steps
-   QA stage enforcement
-   artifact schema + semantic checks

------------------------------------------------------------------------

## Risk 5 --- Local LLM Availability

Local models can fail due to:

-   GPU memory exhaustion
-   Ollama service instability

Mitigation:

-   model fallback
-   dispatcher retries
-   health monitoring

------------------------------------------------------------------------

## Risk 6 --- Router Misclassification

Heuristic routers may misclassify user intent.

Risk:

Execution tasks downgraded to chat.

Mitigation:

-   clarification prompts
-   escalation rules
-   future LLM routing

------------------------------------------------------------------------

## Risk 7 --- Observability Gaps

Without logging, debugging agent systems becomes impossible.

Mitigation:

-   workflow event logs
-   artifact versioning
-   run‑id traceability

------------------------------------------------------------------------

# 6. Architectural Direction

For the M4 milestone, the correct priorities are:

-   Determinism
-   Observability
-   Validation
-   End‑to‑end workflow stability

Not yet:

-   maximum parallelism
-   token efficiency
-   advanced orchestration

Future roadmap:

M5 --- workflow DAG support\
M6 --- AST patch system\
M7 --- adaptive model routing
