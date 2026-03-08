/**
 * Data-access layer for traces.
 */

/**
 * @param {import('pg').Pool} pool
 * @param {{ trace_id:string, project_id?:string, task_type?:string, context_digest?:string, action_json?:object|string, metrics_json?:object|string }} trace
 */
export async function insertTrace(pool, trace) {
  const suffix = trace.ignore_conflict ? " ON CONFLICT DO NOTHING" : "";
  await pool.query(
    `INSERT INTO traces(trace_id, project_id, task_type, context_digest, action_json, metrics_json, created_at)
     VALUES ($1, $2, $3, $4, $5, $6, NOW())${suffix}`,
    [
      trace.trace_id,
      trace.project_id || "general",
      trace.task_type || "unknown",
      trace.context_digest || "",
      typeof trace.action_json === "string" ? trace.action_json : JSON.stringify(trace.action_json || {}),
      typeof trace.metrics_json === "string" ? trace.metrics_json : JSON.stringify(trace.metrics_json || {}),
    ]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} trace_id
 * @param {object|string} feedback_json
 */
export async function updateTraceFeedback(pool, trace_id, feedback_json) {
  await pool.query(
    "UPDATE traces SET feedback_json=$1 WHERE trace_id=$2",
    [typeof feedback_json === "string" ? feedback_json : JSON.stringify(feedback_json || {}), trace_id]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} trace_id
 */
export async function getTraceProjectId(pool, trace_id) {
  const row = await pool.query("SELECT project_id FROM traces WHERE trace_id=$1", [trace_id]);
  return row.rows[0]?.project_id || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} trace_id
 */
export async function getTraceProjectAndAction(pool, trace_id) {
  const row = await pool.query("SELECT project_id, action_json FROM traces WHERE trace_id=$1", [trace_id]);
  return row.rows[0] || null;
}
