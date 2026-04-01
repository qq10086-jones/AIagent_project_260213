export const RESPONSE_MODES = [
  "direct_reply",
  "progress_update",
  "approval_request",
  "final_completion_reply",
];

export function makeDirectReplyResponse({ run_id, task_envelope, reply }) {
  return {
    ok: true,
    response_mode: "direct_reply",
    run_id,
    task_envelope,
    reply,
  };
}

export function makeErrorResponse({ run_id = "", error = "", error_code = "UNKNOWN_ERROR", task_envelope = null }) {
  return {
    ok: false,
    response_mode: "final_completion_reply",
    run_id: String(run_id || ""),
    error: String(error || "unknown error"),
    error_code: String(error_code || "UNKNOWN_ERROR"),
    task_envelope: task_envelope || null,
  };
}

export function makeTaskQueuedResponse({ run_id, task_envelope, task_id, tool_name, waiting_approval = false, advisory = null }) {
  return {
    ok: true,
    response_mode: waiting_approval ? "approval_request" : "progress_update",
    run_id,
    task_envelope,
    execution: {
      task_id,
      tool_name,
      waiting_approval,
      advisory: advisory || null,
    },
  };
}

export function makeWorkflowQueuedResponse({ run_id, task_envelope, workflow_run_id, workflow_id, first_step }) {
  return {
    ok: true,
    response_mode: "progress_update",
    run_id,
    task_envelope,
    execution: {
      workflow_run_id,
      workflow_id,
      first_step: first_step || null,
    },
  };
}
