import { normalizeStepStatus } from "./workflow_state.js";

const BE_STEP_ID = "impl_be";
const FE_STEP_ID = "impl_fe";
const QA_STEP_ID = "qa_verify";

function getStepStatus(stepRows, stepId) {
  const row = stepRows.find((r) => String(r.step_id || "") === stepId);
  return row ? normalizeStepStatus(row.status) : "pending";
}

/**
 * WS-24.5-03
 * Evaluates whether QA admission is valid after parallel BE+FE execution.
 *
 * Returns { allowed: boolean, denial_reason?: string }
 *
 * Allowed only when:
 *  - Both impl_be and impl_fe are in "succeeded" state
 *  - Neither branch is in a terminal failure state
 */
export function evaluateQaAdmission({ stepRows = [], gateDecision = {} }) {
  if (gateDecision.effective_exposure_decision !== "gated_parallel_allowed") {
    return { allowed: true };
  }

  const beStatus = getStepStatus(stepRows, BE_STEP_ID);
  const feStatus = getStepStatus(stepRows, FE_STEP_ID);

  if (beStatus !== "succeeded") {
    return { allowed: false, denial_reason: `be_branch_not_succeeded:${beStatus}` };
  }
  if (feStatus !== "succeeded") {
    return { allowed: false, denial_reason: `fe_branch_not_succeeded:${feStatus}` };
  }
  return { allowed: true };
}

/**
 * WS-24.5-03
 * Evaluates whether release pack generation is permitted.
 *
 * Returns { allowed: boolean, denial_reason?: string }
 *
 * Blocked when the workflow run is in partial_failure or any impl branch failed.
 */
export function evaluateReleaseGating({ workflowRun = {}, stepRows = [], gateDecision = {} }) {
  const runStatus = String(workflowRun.status || "");
  if (runStatus === "partial_failure") {
    return { allowed: false, denial_reason: "workflow_in_partial_failure" };
  }

  if (gateDecision.effective_exposure_decision === "gated_parallel_allowed") {
    const beStatus = getStepStatus(stepRows, BE_STEP_ID);
    const feStatus = getStepStatus(stepRows, FE_STEP_ID);
    const qaStatus = getStepStatus(stepRows, QA_STEP_ID);

    if (beStatus !== "succeeded" || feStatus !== "succeeded") {
      return { allowed: false, denial_reason: "parallel_branches_not_both_succeeded" };
    }
    if (qaStatus !== "succeeded") {
      return { allowed: false, denial_reason: "qa_not_succeeded" };
    }
  }

  return { allowed: true };
}
