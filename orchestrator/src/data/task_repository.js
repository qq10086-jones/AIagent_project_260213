/**
 * task_repository.js
 *
 * Data-access layer for the `tasks` and `event_log` tables.
 * All exported functions accept `pool` (pg.Pool) as their first argument
 * so they can be unit-tested without relying on module-level singletons.
 */

/**
 * Insert or update a task row.
 * @param {import('pg').Pool} pool
 * @param {{ task_id: string, tool_name: string, status: string, risk_level?: string,
 *           payload: object, run_id: string, idempotency_key?: string,
 *           workflow_id?: string, step_index?: number }} task
 */
export async function upsertTask(pool, task) {
  await pool.query(
    `INSERT INTO tasks(task_id, tool_name, status, risk_level, payload_json, run_id, idempotency_key, workflow_id, step_index)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
     ON CONFLICT (task_id) DO UPDATE SET status=EXCLUDED.status, updated_at=NOW()`,
    [
      task.task_id,
      task.tool_name,
      task.status,
      task.risk_level || "low",
      JSON.stringify(task.payload),
      task.run_id,
      task.idempotency_key || null,
      task.workflow_id || null,
      Number.isFinite(task.step_index) ? task.step_index : null,
    ]
  );
}

/**
 * Return the Redis stream name for a given tool_name.
 * Accepts `pool` as first param for API consistency (not used).
 * @param {import('pg').Pool} pool
 * @param {string} tool_name
 * @param {{ streamTask: string, streamTaskCoding: string }} streams
 */
export function getTaskStream(pool, tool_name, { streamTask, streamTaskCoding }) {
  if (typeof tool_name === "string" && tool_name.startsWith("coding.")) {
    return streamTaskCoding;
  }
  return streamTask;
}

/**
 * Append a row to the event_log table.
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 * @param {string} event_type
 * @param {object} [payload]
 */
export async function recordTaskEvent(pool, task_id, event_type, payload = {}) {
  try {
    await pool.query(
      "INSERT INTO event_log(task_id, event_type, payload_json) VALUES ($1,$2,$3)",
      [task_id, event_type, JSON.stringify(payload || {})]
    );
  } catch (err) {
    console.warn(`[orchestrator] event_log insert failed (${event_type}):`, err.message);
  }
}

/**
 * Fetch task row required by approve flow.
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 */
export async function getTaskForApproval(pool, task_id) {
  const row = await pool.query(
    "SELECT task_id, tool_name, payload_json, run_id, status, workflow_id, step_index FROM tasks WHERE task_id=$1",
    [task_id]
  );
  return row.rows[0] || null;
}

export async function getTaskPayloadRecord(pool, task_id) {
  const row = await pool.query(
    "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1",
    [task_id]
  );
  return row.rows[0] || null;
}

/**
 * Fetch task row required by reject flow.
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 */
export async function getTaskForRejection(pool, task_id) {
  const row = await pool.query(
    "SELECT task_id, tool_name, run_id, status FROM tasks WHERE task_id=$1",
    [task_id]
  );
  return row.rows[0] || null;
}

/**
 * Mark a task as queued.
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 */
export async function markTaskQueued(pool, task_id) {
  await pool.query("UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2", ["queued", task_id]);
}

/**
 * Mark a task as failed due to approval rejection.
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 * @param {object} resultPayload
 */
export async function markTaskApprovalRejected(pool, task_id, resultPayload) {
  await pool.query(
    "UPDATE tasks SET status=$1, error_code=$3, result_json=$4, updated_at=NOW() WHERE task_id=$2",
    ["failed", task_id, "APPROVAL_REJECTED", JSON.stringify(resultPayload)]
  );
}

/**
 * Count pending tasks for a run.
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function countPendingTasksForRun(pool, run_id) {
  const row = await pool.query(
    "SELECT COUNT(1)::int AS c FROM tasks WHERE run_id=$1 AND status IN ('queued','running','waiting_approval')",
    [run_id]
  );
  return Number(row.rows[0]?.c || 0);
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 */
export async function markTaskRunning(pool, task_id) {
  await pool.query(
    "UPDATE tasks SET status=$1, updated_at=NOW() WHERE task_id=$2 AND COALESCE(status,'') IN ('queued','waiting_approval')",
    ["running", task_id]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 * @param {object} resultPayload
 * @param {string} errorCode
 * @param {string} status
 */
export async function updateTaskTerminalResult(pool, task_id, resultPayload, errorCode, status) {
  await pool.query(
    "UPDATE tasks SET status=$1, result_json=$2, error_code=$3, updated_at=NOW() WHERE task_id=$4",
    [status, JSON.stringify(resultPayload), errorCode, task_id]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 * @param {object} resultPayload
 * @param {string} errorCode
 */
export async function forceTaskFailed(pool, task_id, resultPayload, errorCode) {
  await pool.query(
    "UPDATE tasks SET status='failed', result_json=$2, error_code=$3, updated_at=NOW() WHERE task_id=$1",
    [task_id, JSON.stringify(resultPayload), errorCode]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 */
export async function findRunIdByTaskId(pool, task_id) {
  const row = await pool.query("SELECT run_id FROM tasks WHERE task_id=$1", [task_id]);
  return row.rows[0]?.run_id || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function countFailedTasksForRun(pool, run_id) {
  const row = await pool.query(
    "SELECT COUNT(1)::int AS c FROM tasks WHERE run_id=$1 AND status='failed'",
    [run_id]
  );
  return Number(row.rows[0]?.c || 0);
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function listTasksForRunStatus(pool, run_id) {
  const row = await pool.query(
    `SELECT task_id, tool_name, status, error_code, updated_at
     FROM tasks
     WHERE run_id=$1
     ORDER BY created_at ASC`,
    [run_id]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function listTasksForRunTimeline(pool, run_id) {
  const row = await pool.query(
    "SELECT task_id, tool_name, status, error_code, created_at, updated_at FROM tasks WHERE run_id=$1 ORDER BY created_at ASC",
    [run_id]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} idempotencyKey
 */
export async function findTaskIdByIdempotencyKey(pool, idempotencyKey) {
  const row = await pool.query("SELECT task_id FROM tasks WHERE idempotency_key=$1 LIMIT 1", [idempotencyKey]);
  return row.rows[0]?.task_id || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {number} limit
 */
export async function listPendingApprovalTasks(pool, limit = 50) {
  const row = await pool.query(
    `SELECT task_id, run_id, tool_name, risk_level, error_code, payload_json, created_at, updated_at
     FROM tasks
     WHERE status='waiting_approval'
     ORDER BY created_at ASC
     LIMIT $1`,
    [limit]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {number} timeoutSec
 */
export async function listStaleRunningTasks(pool, timeoutSec) {
  const row = await pool.query(
    `SELECT task_id, run_id, tool_name, payload_json, workflow_id, step_index
     FROM tasks
     WHERE status='running'
       AND updated_at < NOW() - ($1::int * INTERVAL '1 second')
     ORDER BY updated_at ASC
     LIMIT 50`,
    [timeoutSec]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {number} timeoutSec
 */
export async function listStaleQueuedTasks(pool, timeoutSec) {
  const row = await pool.query(
    `SELECT task_id, run_id, tool_name, payload_json, workflow_id, step_index
     FROM tasks
     WHERE status='queued'
       AND updated_at < NOW() - ($1::int * INTERVAL '1 second')
     ORDER BY updated_at ASC
     LIMIT 50`,
    [timeoutSec]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 * @param {string} errorCode
 * @param {object} resultPayload
 */
export async function failRunningTaskIfStillRunning(pool, task_id, errorCode, resultPayload) {
  await pool.query(
    "UPDATE tasks SET status='failed', error_code=$2, result_json=$3, updated_at=NOW() WHERE task_id=$1 AND status='running'",
    [task_id, errorCode, JSON.stringify(resultPayload)]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} task_id
 * @param {string} errorCode
 * @param {object} resultPayload
 */
export async function failQueuedTaskIfStillQueued(pool, task_id, errorCode, resultPayload) {
  await pool.query(
    "UPDATE tasks SET status='failed', error_code=$2, result_json=$3, updated_at=NOW() WHERE task_id=$1 AND status='queued'",
    [task_id, errorCode, JSON.stringify(resultPayload)]
  );
}
