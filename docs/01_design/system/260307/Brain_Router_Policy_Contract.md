# Brain Router Policy Contract
## WS-13-01
## Date: 2026-03-07

---

## 1. Purpose

This contract defines the deterministic policy override layer that sits on top of LLM-based intent classification in the Brain Router.

Governed by: Design Document v2 §5.4, Governance v2 Principle 6 (determinism requirement).

The policy layer ensures that:
- Ambiguous or erroneous LLM output does not silently route to the wrong pipeline
- High-risk inputs are always escalated regardless of LLM classification
- Known command prefixes always resolve to the correct intent
- The system degrades gracefully when the LLM is unavailable or returns invalid JSON

---

## 2. Two-Phase Routing Model

```
Phase A — LLM Classification
  Input: normalized raw_input
  Output: { intent, confidence, sub_intent } or null on failure

Phase B — Policy Override Layer   ← this contract
  Input: Phase A result + raw_input
  Output: final { decision, intent, route }
```

Phase B always runs, even if Phase A fails.

---

## 3. Policy Rules

Each rule has:
- `rule_id` — unique identifier
- `trigger` — condition evaluated against raw_input and/or Phase A output
- `override_action` — what the policy forces
- `log_level` — info / warn

### Rule P-01: Coder Directive Force

```json
{
  "rule_id": "P-01",
  "trigger": "raw_input starts with '/coder' (case-insensitive)",
  "override_action": {
    "decision": "orchestrated_workflow",
    "intent": "coding",
    "workflow_id": "coding_team_v0"
  },
  "log_level": "info",
  "note": "Explicit user command — no LLM input needed"
}
```

### Rule P-02: Trivial Input Force

```json
{
  "rule_id": "P-02",
  "trigger": "raw_input token count < 3 after normalization",
  "override_action": {
    "decision": "direct_reply",
    "intent": "chat"
  },
  "log_level": "info",
  "note": "Too short to be an execution task; avoids wasting orchestration resources"
}
```

### Rule P-03: Financial Keyword Escalation

```json
{
  "rule_id": "P-03",
  "trigger": "raw_input contains any of: ['buy', 'sell', 'trade', 'execute order', 'market order', 'broker', 'portfolio rebalance'] AND LLM intent == 'quant'",
  "override_action": {
    "decision": "human_review_required",
    "intent": "quant"
  },
  "log_level": "warn",
  "note": "Financial execution requires explicit human approval regardless of confidence"
}
```

### Rule P-04: LLM Unknown Intent Downgrade

```json
{
  "rule_id": "P-04",
  "trigger": "Phase A returns intent == 'unknown' OR confidence < 0.4",
  "override_action": {
    "decision": "direct_reply",
    "intent": "chat",
    "clarification_prompt": "I'm not sure what you'd like me to do. Could you rephrase your request?"
  },
  "log_level": "warn",
  "note": "Unknown intent is treated as chat with clarification, not a silent routing failure"
}
```

### Rule P-05: LLM Failure Fallback

```json
{
  "rule_id": "P-05",
  "trigger": "Phase A throws an exception OR returns null OR returns non-JSON",
  "override_action": {
    "decision": "direct_reply",
    "intent": "chat",
    "error_note": "LLM_CLASSIFICATION_FAILED"
  },
  "log_level": "warn",
  "note": "System must not crash or route blindly when LLM is unavailable"
}
```

---

## 4. Rule Evaluation Order

Rules are evaluated in this order. First match wins:

```
P-01 (coder directive) → explicit command, highest priority
P-02 (trivial input)   → prevent waste before LLM call
P-05 (LLM failure)     → evaluated after LLM call attempt
P-03 (financial guard) → evaluated with LLM result
P-04 (unknown intent)  → evaluated with LLM result
(no match)             → use LLM classification result as-is
```

---

## 5. Policy Output Contract

Schema: `orchestrator/contracts/brain_router_policy.schema.json` (to be created in WS-13-02)

```json
{
  "applied_rule": "P-01 | P-02 | P-03 | P-04 | P-05 | null",
  "decision": "direct_reply | single_agent | orchestrated_workflow | human_review_required",
  "intent": "chat | coding | quant | docs | research | ops | unknown",
  "override": true,
  "log_level": "info | warn | null"
}
```

If `applied_rule` is `null`, the LLM output was used without override.

---

## 6. Logging Requirements

Every policy evaluation must log:
- `rule_id` applied (or "none")
- `raw_input` (first 100 chars, truncated)
- `llm_intent` from Phase A (or "null" if LLM failed)
- `final_decision` from Phase B
- `override: true/false`

Log format:
```
[brain_router_policy] rule=P-01 input="..." llm_intent=null final=orchestrated_workflow override=true
```

---

## 7. Definition of Done (WS-13)

The policy layer is complete when:
- `src/vnext/brain_router_policy.js` module exists and exports `applyRoutingPolicy(rawInput, llmResult)`
- `orchestrator/contracts/brain_router_policy.schema.json` validates policy output
- `brain_router.js` calls `applyRoutingPolicy` after every `parseIntent` call
- Unit tests cover all 5 rules
- Integration test covers: P-01 coder directive, P-02 trivial input, P-05 LLM failure

---

## 8. Non-Scope

- No machine learning on routing decisions
- No per-user policy customization
- No A/B testing framework
- No confidence threshold tuning UI
