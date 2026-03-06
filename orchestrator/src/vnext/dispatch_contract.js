import { normalizeInputRequest } from "./input_normalizer.js";
import { routeTaskRequest } from "./brain_router.js";
import {
  makeDirectReplyResponse,
  makeTaskQueuedResponse,
  makeWorkflowQueuedResponse,
  makeErrorResponse,
} from "./response_protocol.js";
import {
  assertDispatchErrorResponse,
  assertDispatchSuccessResponse,
} from "./contract_validator.js";

function buildPreviewIds(seed = "preview") {
  return {
    run_id: `${seed}-run`,
    task_id: `${seed}-task`,
    workflow_run_id: `${seed}-workflow-run`,
  };
}

export function buildDispatchContractPreview({
  body = {},
  analyzerResult = null,
  registry,
  routeOverride = null,
  previewSeed = "preview",
}) {
  const normalized = normalizeInputRequest(body || {});
  if (!normalized.raw_input) {
    return assertDispatchErrorResponse(makeErrorResponse({
      run_id: `${previewSeed}-run`,
      error: "raw_input/message is required",
      error_code: "BAD_REQUEST",
      task_envelope: null,
    }));
  }

  const routed = routeOverride || routeTaskRequest({
    ...normalized,
    analyzerResult,
    registry,
  });
  const ids = buildPreviewIds(previewSeed);
  const plan = routed.task_envelope.execution_plan || {};

  if (routed.decision === "direct_reply") {
    return assertDispatchSuccessResponse(makeDirectReplyResponse({
      run_id: ids.run_id,
      task_envelope: routed.task_envelope,
      reply: "__preview_reply__",
    }));
  }

  if (routed.decision === "single_agent") {
    return assertDispatchSuccessResponse(makeTaskQueuedResponse({
      run_id: ids.run_id,
      task_envelope: routed.task_envelope,
      task_id: ids.task_id,
      tool_name: String(plan.tool_name || "coding.delegate"),
      waiting_approval: false,
    }));
  }

  if (routed.decision === "orchestrated_workflow") {
    return assertDispatchSuccessResponse(makeWorkflowQueuedResponse({
      run_id: ids.run_id,
      task_envelope: routed.task_envelope,
      workflow_run_id: ids.workflow_run_id,
      workflow_id: String(plan.workflow_id || "coding_team_v0"),
      first_step: null,
    }));
  }

  return assertDispatchErrorResponse(makeErrorResponse({
    run_id: ids.run_id,
    error: "human review required",
    error_code: "UNKNOWN_ERROR",
    task_envelope: routed.task_envelope,
  }));
}
