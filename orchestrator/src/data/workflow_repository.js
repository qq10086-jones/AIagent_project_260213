/**
 * Data-access layer for workflow run/step/checkpoint queries used by timeline APIs.
 */

export async function getWorkflowStepByIndex(pool, workflowRunId, stepIndex) {
  const row = await pool.query(
    "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2",
    [workflowRunId, stepIndex]
  );
  return row.rows[0] || null;
}

export async function getWorkflowStepGateByIndex(pool, workflowRunId, stepIndex) {
  const row = await pool.query(
    "SELECT gate_name FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2 LIMIT 1",
    [workflowRunId, stepIndex]
  );
  return row.rows[0] || null;
}

export async function insertWorkflowRun(pool, workflowRun) {
  await pool.query(
    `INSERT INTO workflow_runs(workflow_run_id, run_id, workflow_id, project_type, status, current_step_index, input_json)
     VALUES ($1,$2,$3,$4,'running',0,$5)`,
    [
      workflowRun.workflow_run_id,
      workflowRun.run_id,
      workflowRun.workflow_id,
      workflowRun.project_type,
      JSON.stringify(workflowRun.input || {}),
    ]
  );
}

export async function insertWorkflowStep(pool, workflowStep) {
  await pool.query(
    `INSERT INTO workflow_steps(workflow_run_id, step_index, step_id, role_name, tool_name, gate_name, status)
     VALUES ($1,$2,$3,$4,$5,$6,'pending')`,
    [
      workflowStep.workflow_run_id,
      workflowStep.step_index,
      workflowStep.step_id,
      workflowStep.role_name,
      workflowStep.tool_name,
      workflowStep.gate_name,
    ]
  );
}

export async function updateWorkflowRunFailed(pool, workflowRunId, errorCode, errorMessage) {
  await pool.query(
    `UPDATE workflow_runs
     SET status='failed', error_code=$2, error_message=$3, updated_at=NOW()
     WHERE workflow_run_id=$1`,
    [workflowRunId, String(errorCode || "WORKFLOW_FAILED"), String(errorMessage || "workflow failed")]
  );
}

export async function updateWorkflowRunPartialFailure(pool, workflowRunId, errorCode, errorMessage) {
  await pool.query(
    `UPDATE workflow_runs
     SET status='partial_failure', error_code=$2, error_message=$3, updated_at=NOW()
     WHERE workflow_run_id=$1`,
    [workflowRunId, String(errorCode || "PARTIAL_FAILURE"), String(errorMessage || "workflow partial failure")]
  );
}

export async function updateWorkflowRunSucceeded(pool, workflowRunId) {
  await pool.query(
    "UPDATE workflow_runs SET status='succeeded', updated_at=NOW() WHERE workflow_run_id=$1",
    [workflowRunId]
  );
}

export async function updateWorkflowRunCurrentStep(pool, workflowRunId, stepIndex) {
  await pool.query(
    "UPDATE workflow_runs SET current_step_index=$2, status='running', updated_at=NOW() WHERE workflow_run_id=$1",
    [workflowRunId, stepIndex]
  );
}

export async function clearWorkflowRunErrorAndSetRunning(pool, workflowRunId) {
  await pool.query(
    "UPDATE workflow_runs SET status='running', error_code=NULL, error_message=NULL, updated_at=NOW() WHERE workflow_run_id=$1",
    [workflowRunId]
  );
}

export async function updateWorkflowRunResumeToken(pool, workflowRunId, resumeToken) {
  await pool.query(
    "UPDATE workflow_runs SET resume_token=$2, updated_at=NOW() WHERE workflow_run_id=$1",
    [workflowRunId, resumeToken]
  );
}

export async function updateWorkflowRunLastCheckpoint(pool, workflowRunId, checkpointId) {
  await pool.query(
    "UPDATE workflow_runs SET last_checkpoint_id=$2, updated_at=NOW() WHERE workflow_run_id=$1",
    [workflowRunId, checkpointId]
  );
}

export async function updateWorkflowStepDispatchState(pool, workflowRunId, stepIndex, dispatchState) {
  await pool.query(
    `UPDATE workflow_steps
     SET status=$3, task_id=$4, risk_level=$5, approval_required=$6,
         approval_reasons_json=$7, started_at=COALESCE(started_at, NOW()), updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [
      workflowRunId,
      stepIndex,
      dispatchState.status,
      dispatchState.task_id,
      dispatchState.risk_level || "low",
      Boolean(dispatchState.approval_required),
      JSON.stringify(dispatchState.approval_reasons || []),
    ]
  );
}

export async function claimStepForDispatch(pool, workflowRunId, stepIndex, expectedStatus) {
  const result = await pool.query(
    `UPDATE workflow_steps
     SET status='claiming', updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2 AND status=$3`,
    [workflowRunId, stepIndex, expectedStatus]
  );
  return (result.rowCount || 0) > 0;
}

export async function markWorkflowStepRunning(pool, workflowRunId, stepIndex) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='running', started_at=COALESCE(started_at, NOW()), updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex]
  );
}

export async function markWorkflowStepQueued(pool, workflowRunId, stepIndex) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='queued', updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex]
  );
}

export async function resetWorkflowStepForResume(pool, workflowRunId, stepIndex) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='pending', task_id=NULL, updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex]
  );
}

export async function updateWorkflowStepFailed(pool, workflowRunId, stepIndex, resultJson, errorCode) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='failed', result_json=$3, error_code=$4, ended_at=NOW(), updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex, JSON.stringify(resultJson), errorCode]
  );
}

export async function failWorkflowStepIfNotSucceeded(pool, workflowRunId, stepIndex, errorCode) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='failed', error_code=$3, ended_at=NOW(), updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2 AND status <> 'succeeded'`,
    [workflowRunId, stepIndex, String(errorCode || "WORKFLOW_FAILED")]
  );
}

export async function rejectWorkflowStep(pool, workflowRunId, stepIndex, resultJson) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='failed', error_code='APPROVAL_REJECTED',
         result_json=$3, ended_at=NOW(), updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex, JSON.stringify(resultJson)]
  );
}

export async function updateWorkflowStepSucceeded(pool, workflowRunId, stepIndex, resultJson, checkpointId) {
  await pool.query(
    `UPDATE workflow_steps
     SET status='succeeded', result_json=$3, error_code=NULL,
         ended_at=NOW(), checkpoint_id=$4, updated_at=NOW()
     WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex, JSON.stringify(resultJson), checkpointId]
  );
}

export async function setWorkflowStepCheckpoint(pool, workflowRunId, stepIndex, checkpointId) {
  await pool.query(
    `UPDATE workflow_steps SET checkpoint_id=$3, updated_at=NOW() WHERE workflow_run_id=$1 AND step_index=$2`,
    [workflowRunId, stepIndex, checkpointId]
  );
}

export async function insertWorkflowCheckpoint(pool, checkpoint) {
  await pool.query(
    `INSERT INTO workflow_checkpoints(checkpoint_id, workflow_run_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, checkpoint_json)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
    [
      checkpoint.checkpoint_id,
      checkpoint.workflow_run_id,
      checkpoint.step_index,
      checkpoint.step_id,
      checkpoint.task_id || "",
      checkpoint.workspace_hash,
      JSON.stringify(checkpoint.artifact_refs || []),
      JSON.stringify(checkpoint.checkpoint_json || {}),
    ]
  );
}

export async function getWorkflowCheckpointById(pool, checkpointId) {
  const row = await pool.query(
    "SELECT checkpoint_id, step_index, workspace_hash FROM workflow_checkpoints WHERE checkpoint_id=$1",
    [checkpointId]
  );
  return row.rows[0] || null;
}

export async function getWorkflowCheckpointForRun(pool, checkpointId, workflowRunId) {
  const row = await pool.query(
    "SELECT * FROM workflow_checkpoints WHERE checkpoint_id=$1 AND workflow_run_id=$2",
    [checkpointId, workflowRunId]
  );
  return row.rows[0] || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} workflowRunId
 */
export async function getWorkflowRunById(pool, workflowRunId) {
  const runs = await pool.query("SELECT * FROM workflow_runs WHERE workflow_run_id=$1", [workflowRunId]);
  return runs.rows[0] || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} workflowRunId
 */
export async function listWorkflowSteps(pool, workflowRunId) {
  const stepsRes = await pool.query(
    "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC",
    [workflowRunId]
  );
  return stepsRes.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} workflowRunId
 */
export async function listWorkflowCheckpoints(pool, workflowRunId) {
  const checkpointsRes = await pool.query(
    "SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash FROM workflow_checkpoints WHERE workflow_run_id=$1 ORDER BY step_index ASC",
    [workflowRunId]
  );
  return checkpointsRes.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function countFailedWorkflowRunsForRun(pool, run_id) {
  const row = await pool.query(
    "SELECT COUNT(1)::int AS c FROM workflow_runs WHERE run_id=$1 AND status='failed'",
    [run_id]
  );
  return Number(row.rows[0]?.c || 0);
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} workflow_id
 * @param {string} name
 * @param {object} definition
 */
export async function insertWorkflowDefinition(pool, workflow_id, name, definition) {
  await pool.query(
    `INSERT INTO workflows(workflow_id, name, definition_json)
     VALUES ($1,$2,$3)`,
    [workflow_id, String(name || "chat-workflow"), JSON.stringify(definition || {})]
  );
}
