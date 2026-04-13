# v3.5.1 Dynamic Model Escalation — Design Document

**Date**: 2026-04-11
**Status**: Implemented
**Triggered by**: pm_spec step failing 3x GOAL_FIDELITY_VIOLATION on gemma4:26b with v3.5 Design Quality prompts

---

## 1. Problem

v3.5 Design Quality Intelligence injects ~2KB of UX rules into PM prompt. gemma4:26b (26B local model) cannot reliably follow all instructions, causing GOAL_FIDELITY_VIOLATION on retry. The system retries with the same model 2x, wastes 6+ minutes, and fails.

**Root cause**: No model escalation on validation failure. The retry logic re-dispatches with identical model + adjusted prompt, but the model's instruction-following capacity is the bottleneck, not the prompt content.

---

## 2. Solution: Lane Escalation Chain

### Concept
When a step fails validation, the retry uses a **stronger model** from a pre-configured escalation chain. This is analogous to:
- **MetaGPT**: uses GPT-4 for complex planning, GPT-3.5 for simple tasks
- **Aider**: escalates from "weak model" to "strong model" on edit failures
- **AlphaCodium**: multi-pass with increasing model capability per pass

### Escalation Chain
```
stable_gemma4_lane (26B local) 
  -> stable_cloud_lane (MiniMax-M2.7 cloud)
```

### Config
```json
// runtime_defaults.json
{
  "worker_coder": {
    "lane_escalation_chain": ["stable_gemma4_lane", "stable_cloud_lane"],
    "lane_escalation_enabled": true,
    "step_validation_retry_max": 2
  }
}
```

### Flow
```
Step executes on gemma4:26b
  -> Validation fails (GOAL_FIDELITY_VIOLATION)
  -> workflow_engine.js: compute next lane from escalation_chain
  -> Write escalated_lane to meta/validation_feedback_<step>.txt
  -> Re-dispatch step
  -> workflow_step_builder.js: read escalated_lane from feedback
  -> Override payload.execution_lane, provider, model
  -> Step executes on MiniMax-M2.7
  -> Validation passes (stronger model follows instructions)
```

### Retry Behavior
| Attempt | Model | Lane | Trigger |
|---------|-------|------|---------|
| 1 | gemma4:26b | stable_gemma4_lane | Initial dispatch |
| 2 (retry 1) | MiniMax-M2.7 | stable_cloud_lane | GOAL_FIDELITY_VIOLATION |
| 3 (retry 2) | MiniMax-M2.7 | stable_cloud_lane | Same (no further escalation) |

---

## 3. Files Modified

| File | Change |
|------|--------|
| `configs/runtime/runtime_defaults.json` | Added `lane_escalation_chain`, `lane_escalation_enabled` |
| `orchestrator/src/workflow_engine.js` | Compute escalated lane on validation retry, write to feedback file |
| `orchestrator/src/domain/workflow_step_builder.js` | Read `escalated_lane` from feedback, override payload lane/provider/model |

---

## 4. Design Decisions

1. **Config-driven, not code-driven**: Escalation chain is in runtime_defaults.json, not hardcoded. Can add more tiers (e.g. glm-4.7-flash between gemma4 and M2.7) without code changes.

2. **Per-retry, not per-step**: Escalation happens on retry, not pre-emptively. This means simple tasks still use the cheap local model. Only tasks that actually fail get escalated.

3. **Feedback file as transport**: Reuses the existing `meta/validation_feedback_<step>.txt` mechanism. No new IPC or database schema needed.

4. **No rollback**: Once escalated, the step stays on the stronger model for remaining retries. This is intentional -- if the stronger model also fails, the issue is likely in the prompt/validator, not the model.

---

## 5. Future Extensions

- **Complexity-based pre-selection**: Use brain_router complexity score to skip gemma4 for complex tasks entirely
- **Token estimation**: Estimate prompt tokens before dispatch, pre-emptively use cloud for large prompts
- **Cost tracking**: Log model usage per run for cost analysis (local vs cloud)
- **Per-step escalation policy**: Different steps could have different chains (e.g. PM escalates faster than QA)
