/**
 * workflow_checkpoint.js
 *
 * Extracted from workflow_engine.js (WS-33-02).
 *
 * Provides createWorkflowCheckpointService({ pool }) which encapsulates
 * checkpoint creation: hash computation, DB insert, and step/run linkage.
 */

import { v4 as uuidv4 } from "uuid";
import { buildWorkspaceHash } from "./workflow_artifact_audit.js";
import {
  insertWorkflowCheckpoint,
  setWorkflowStepCheckpoint,
  updateWorkflowRunLastCheckpoint,
} from "../data/workflow_repository.js";

export function createWorkflowCheckpointService({ pool }) {
  async function createCheckpoint({ workflow_run_id, stepIndex, step_id, task_id, status, output }) {
    const artifacts = Array.isArray(output?.artifacts)
      ? output.artifacts.map((a) => ({
          bucket: a?.bucket || null, object_key: a?.object_key || null,
          name: a?.name || null, sha256: a?.sha256 || null, mime: a?.mime || null,
        }))
      : [];
    const workspace_hash = buildWorkspaceHash({ workflow_run_id, step_index: stepIndex, task_id, status, artifacts });
    const checkpoint_id = uuidv4();
    await insertWorkflowCheckpoint(pool, {
      checkpoint_id,
      workflow_run_id,
      step_index: stepIndex,
      step_id,
      task_id,
      workspace_hash,
      artifact_refs: artifacts,
      checkpoint_json: { workflow_run_id, step_index: stepIndex, step_id, task_id, status, artifacts },
    });
    await setWorkflowStepCheckpoint(pool, workflow_run_id, stepIndex, checkpoint_id);
    await updateWorkflowRunLastCheckpoint(pool, workflow_run_id, checkpoint_id);
    return { checkpoint_id, workspace_hash, artifacts };
  }

  return { createCheckpoint };
}
