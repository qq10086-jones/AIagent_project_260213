/**
 * workflow_resume.js
 *
 * Resume-token issuance and resume-from-token flow.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import { signResumePayload, verifyResumePayload } from "./workflow_runner.js";
import { normalizeStepStatus } from "./workflow_state.js";
import {
  clearWorkflowRunErrorAndSetRunning,
  getWorkflowCheckpointById,
  getWorkflowCheckpointForRun,
  resetWorkflowStepForResume,
  updateWorkflowRunResumeToken,
} from "../data/workflow_repository.js";

/**
 * @param {{ pool, resumeTokenSecret, resumeTokenTtlSec,
 *           getRun, getSteps, dispatchStepByIndex, recordEvent }} deps
 */
export function createResumeService({
  pool,
  resumeTokenSecret,
  resumeTokenTtlSec,
  getRun,
  getSteps,
  dispatchStepByIndex,
  recordEvent,
}) {
  async function issueResumeToken(workflow_run_id) {
    const run = await getRun(workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    if (!run.last_checkpoint_id) {
      const err = new Error("RESUME_INVALID: no checkpoint available");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const cp = await getWorkflowCheckpointById(pool, run.last_checkpoint_id);
    if (!cp) {
      const err = new Error("RESUME_INVALID: checkpoint not found");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const nowSec = Math.floor(Date.now() / 1000);
    const tokenPayload = {
      workflow_run_id,
      checkpoint_id: cp.checkpoint_id,
      step_index: Number(cp.step_index),
      workspace_hash: cp.workspace_hash,
      iat: nowSec,
      exp: nowSec + Math.max(300, Number(resumeTokenTtlSec || 86400)),
    };
    const token = signResumePayload(tokenPayload, resumeTokenSecret);
    await updateWorkflowRunResumeToken(pool, workflow_run_id, token);
    return {
      resume_token: token,
      expires_at: tokenPayload.exp,
      checkpoint_id: cp.checkpoint_id,
      step_index: tokenPayload.step_index,
    };
  }

  async function resumeFromToken(workflow_run_id, resume_token, context = null) {
    const checked = verifyResumePayload(resume_token, resumeTokenSecret);
    if (!checked.ok) {
      const err = new Error(checked.error);
      err.code = "RESUME_INVALID";
      throw err;
    }
    const tokenPayload = checked.payload;
    if (tokenPayload.workflow_run_id !== workflow_run_id) {
      const err = new Error("RESUME_INVALID: workflow_run mismatch");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const run = await getRun(workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    const cp = await getWorkflowCheckpointForRun(pool, tokenPayload.checkpoint_id, workflow_run_id);
    if (!cp || cp.workspace_hash !== tokenPayload.workspace_hash) {
      const err = new Error("RESUME_INVALID: checkpoint mismatch");
      err.code = "RESUME_INVALID";
      throw err;
    }
    const steps = await getSteps(workflow_run_id);
    if (normalizeStepStatus(run.status) === "partial_failure") {
      const failedStep = steps.find((s) => normalizeStepStatus(s.status) === "failed");
      if (!failedStep) {
        const err = new Error("RESUME_INVALID: partial_failure run has no failed step");
        err.code = "RESUME_INVALID";
        throw err;
      }
      await resetWorkflowStepForResume(pool, workflow_run_id, Number(failedStep.step_index));
      await clearWorkflowRunErrorAndSetRunning(pool, workflow_run_id);
      const dispatch = await dispatchStepByIndex(workflow_run_id, Number(failedStep.step_index), context);
      await recordEvent(workflow_run_id, "workflow.resumed", {
        workflow_run_id,
        step_index: Number(failedStep.step_index),
        checkpoint_id: cp.checkpoint_id,
      });
      return { ok: true, workflow_run_id, resumed_step_index: Number(failedStep.step_index), dispatch };
    }
    const nextStep = steps.find(
      (s) => Number(s.step_index) > Number(cp.step_index) && normalizeStepStatus(s.status) !== "succeeded"
    );
    if (!nextStep) {
      const err = new Error("RESUME_INVALID: no resumable step");
      err.code = "RESUME_INVALID";
      throw err;
    }
    await resetWorkflowStepForResume(pool, workflow_run_id, Number(nextStep.step_index));
    await clearWorkflowRunErrorAndSetRunning(pool, workflow_run_id);
    const dispatch = await dispatchStepByIndex(workflow_run_id, Number(nextStep.step_index), context);
    await recordEvent(workflow_run_id, "workflow.resumed", {
      workflow_run_id,
      step_index: Number(nextStep.step_index),
      checkpoint_id: cp.checkpoint_id,
    });
    return { ok: true, workflow_run_id, resumed_step_index: Number(nextStep.step_index), dispatch };
  }

  return { issueResumeToken, resumeFromToken };
}
