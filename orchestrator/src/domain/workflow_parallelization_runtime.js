import { buildDagPlan } from "./dag_scheduler.js";
import { createWorkflowParallelizationPolicyService } from "./workflow_parallelization_policy.js";
import { updateRunStatus } from "../data/run_repository.js";
import { updateWorkflowRunPartialFailure } from "../data/workflow_repository.js";

export function createWorkflowParallelizationRuntime({ registry, workspaceRoot, recordEvent, pool }) {
  const parallelizationPolicyService = createWorkflowParallelizationPolicyService({ registry, workspaceRoot });
  const lastParallelizationDecision = new Map();

  async function getResolvedWorkflow(run) {
    const resolved = parallelizationPolicyService.resolveWorkflowForRun(run);
    const decision = resolved?.gateDecision || { allowed: false, mode: "sequential", reason_code: "UNKNOWN" };
    const signature = JSON.stringify(decision);
    if (lastParallelizationDecision.get(run.workflow_run_id) !== signature) {
      lastParallelizationDecision.set(run.workflow_run_id, signature);
      await recordEvent(run.workflow_run_id, "workflow.parallelization.gate_decided", {
        workflow_run_id: run.workflow_run_id,
        workflow_id: run.workflow_id,
        project_type: run.project_type,
        ...decision,
      });
    }
    return resolved;
  }

  async function getDagPlan(run) {
    const resolved = await getResolvedWorkflow(run);
    const wf = resolved?.workflow || null;
    if (!wf || !Array.isArray(wf.steps)) return null;
    return buildDagPlan(wf.steps);
  }

  async function markWorkflowPartialFailure(run, failedSteps = []) {
    const errorCode = "PARTIAL_FAILURE";
    const errorMessage = failedSteps.length > 0
      ? `workflow partial failure: ${failedSteps.map((step) => String(step.step_id || step.step_index)).join(", ")}`
      : "workflow partial failure";
    await updateWorkflowRunPartialFailure(pool, run.workflow_run_id, errorCode, errorMessage);
    if (run.run_id) {
      await updateRunStatus(pool, run.run_id, "failed").catch(() => {});
    }
    await recordEvent(run.workflow_run_id, "workflow.partial_failure", {
      workflow_run_id: run.workflow_run_id,
      failed_steps: failedSteps.map((step) => ({
        step_index: Number(step.step_index),
        step_id: String(step.step_id || ""),
        error_code: String(step.error_code || ""),
      })),
    });
  }

  return { getResolvedWorkflow, getDagPlan, markWorkflowPartialFailure };
}
