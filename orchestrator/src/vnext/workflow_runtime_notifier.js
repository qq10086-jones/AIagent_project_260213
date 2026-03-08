import { formatFailureReport, formatTransitionNotification } from "./observability_reporter.js";

export function buildWorkflowRuntimeNotification({
  workflowState = null,
  workflowTerminal = null,
  status = "",
  normalizedErrorCode = "",
  streamError = "",
  output = {},
}) {
  const steps = Array.isArray(workflowState?.steps) ? workflowState.steps : [];
  const previousStep = Number.isInteger(workflowTerminal?.step_index)
    ? steps.find((item) => Number(item.step_index) === Number(workflowTerminal.step_index)) || null
    : null;
  const nextStep = steps.find((item) => {
    const s = String(item?.status || "");
    return s === "queued" || s === "waiting_approval" || s === "running";
  }) || null;

  if (String(status || "") === "succeeded") {
    return {
      kind: "transition",
      message: formatTransitionNotification({
        previousStep: previousStep ? { id: previousStep.step_id, role: previousStep.role_name } : null,
        nextStep: nextStep ? { id: nextStep.step_id, role: nextStep.role_name } : null,
      }),
    };
  }

  return {
    kind: "failure",
    message: formatFailureReport({
      step_id: previousStep?.step_id || String(workflowTerminal?.step_id || "unknown_step"),
      role: previousStep?.role_name || "unknown_role",
      error_code: normalizedErrorCode,
      error_message: streamError || normalizedErrorCode || "workflow step failed",
      raw_logs: output?.stderr || output?.stdout || output?.raw || "",
    }),
  };
}
