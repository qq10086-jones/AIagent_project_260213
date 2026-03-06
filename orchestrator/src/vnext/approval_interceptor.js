import { classifyRisk } from "./risk_classifier.js";
import { makeTaskQueuedResponse } from "./response_protocol.js";
import { assertDispatchSuccessResponse } from "./contract_validator.js";

/**
 * Intercepts task dispatch attempts to enforce approval requirements.
 * If the risk is high, yields an approval request instead of executing.
 */
export function interceptAndDispatch({ taskEnvelope, executionPlan, runId, taskId }) {
  if (!taskEnvelope || !executionPlan || !runId || !taskId) {
    throw new Error("Missing required parameters for interceptAndDispatch");
  }

  const riskResult = classifyRisk({
    intent: taskEnvelope.intent,
    tool_name: executionPlan.tool_name,
    raw_input: taskEnvelope.raw_input,
  });

  // If high risk and not yet approved
  if (riskResult.requires_approval && !taskEnvelope.constraints?.approved) {
    return assertDispatchSuccessResponse(makeTaskQueuedResponse({
      run_id: runId,
      task_envelope: taskEnvelope,
      task_id: taskId,
      tool_name: executionPlan.tool_name,
      waiting_approval: true,
    }));
  }

  // Otherwise proceed normally (mode: progress_update)
  return assertDispatchSuccessResponse(makeTaskQueuedResponse({
    run_id: runId,
    task_envelope: taskEnvelope,
    task_id: taskId,
    tool_name: executionPlan.tool_name,
    waiting_approval: false,
  }));
}
