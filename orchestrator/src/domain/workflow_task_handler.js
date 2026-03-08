/**
 * workflow_task_handler.js
 *
 * Lightweight task lifecycle event handlers (claimed / approved / rejected).
 * The heavier handleTaskTerminal remains in workflow_engine.js because it
 * depends on dispatchStepByIndex and createCheckpoint which are local to the engine.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import { parseJsonSafe } from "./workflow_runner.js";
import {
  markWorkflowStepQueued,
  markWorkflowStepRunning,
  rejectWorkflowStep,
} from "../data/workflow_repository.js";
import { getTaskPayloadRecord } from "../data/task_repository.js";

/**
 * @param {{ pool, recordEvent, getRun, failWorkflowRun }} deps
 */
export function createTaskHandlerService({ pool, recordEvent, getRun, failWorkflowRun }) {
  async function handleTaskClaimed(task_id) {
    const task = await getTaskPayloadRecord(pool, task_id);
    if (!task) return { handled: false };
    const payload = parseJsonSafe(task.payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };
    await markWorkflowStepRunning(pool, workflow_run_id, step_index);
    return { handled: true, workflow_run_id, step_index };
  }

  async function handleTaskApproved(task_id) {
    const task = await getTaskPayloadRecord(pool, task_id);
    if (!task) return { handled: false };
    const payload = parseJsonSafe(task.payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };
    await markWorkflowStepQueued(pool, workflow_run_id, step_index);
    await recordEvent(task_id, "workflow.step.approval.approved", { workflow_run_id, step_index });
    return { handled: true, workflow_run_id, step_index };
  }

  async function handleTaskRejected(task_id, reason = "") {
    const task = await getTaskPayloadRecord(pool, task_id);
    if (!task) return { handled: false };
    const payload = parseJsonSafe(task.payload_json, {});
    const workflow_run_id = payload.workflow_run_id;
    const step_index = Number(payload.step_index);
    const step_id = String(payload.step_id || "");
    if (!workflow_run_id || !Number.isInteger(step_index)) return { handled: false };
    await rejectWorkflowStep(pool, workflow_run_id, step_index, { rejected: true, reason: String(reason || "") });
    const run = await getRun(workflow_run_id);
    if (run) {
      await failWorkflowRun({
        run,
        stepDef: { id: step_id },
        stepIndex: step_index,
        error_code: "APPROVAL_REJECTED",
        error_message: String(reason || "approval rejected"),
      });
    }
    await recordEvent(task_id, "workflow.step.approval.rejected", {
      workflow_run_id, step_index, reason: String(reason || ""),
    });
    return { handled: true, workflow_run_id, step_index };
  }

  return { handleTaskClaimed, handleTaskApproved, handleTaskRejected };
}
