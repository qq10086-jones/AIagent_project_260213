/**
 * waterfall_trace_repository.js  (Layer 4 — Infra / Data)
 *
 * WS-30-02: Waterfall Trace & Latency Attribution
 * CRUD helpers for the `waterfall_stage_log` table.
 *
 * Tracked stages (Design Doc v4.0 § 7):
 *   intake | routing | policy_evaluation | execution_dispatch
 *   branch_completion_be | branch_completion_fe | qa_admission | release_pack_readiness
 *
 * Stages intake–execution_dispatch are recorded explicitly here.
 * Stages branch_completion_* through release_pack_readiness are derived from
 * workflow_steps at query time (started_at / ended_at already captured there).
 */

/**
 * Insert a single stage record.
 *
 * @param {import('pg').Pool} pool
 * @param {{ run_id:string, workflow_run_id?:string, stage:string,
 *           started_at:Date, ended_at?:Date, duration_ms?:number,
 *           metadata?:object }} record
 */
export async function insertWaterfallStage(pool, record) {
  await pool.query(
    `INSERT INTO waterfall_stage_log
       (run_id, workflow_run_id, stage, started_at, ended_at, duration_ms, metadata_json)
     VALUES ($1, $2, $3, $4, $5, $6, $7)`,
    [
      record.run_id,
      record.workflow_run_id ?? null,
      record.stage,
      record.started_at,
      record.ended_at ?? null,
      record.duration_ms ?? null,
      JSON.stringify(record.metadata || {}),
    ]
  );
}

/**
 * Retrieve all explicit stage records for a run, ordered chronologically.
 *
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @returns {Promise<object[]>}
 */
export async function listStagesByRunId(pool, run_id) {
  const res = await pool.query(
    `SELECT * FROM waterfall_stage_log WHERE run_id=$1 ORDER BY started_at ASC`,
    [run_id]
  );
  return res.rows;
}

/**
 * Retrieve workflow_steps rows for branch/QA/release stage derivation.
 * Used by buildWaterfallReport to extend the waterfall with derived stages.
 *
 * @param {import('pg').Pool} pool
 * @param {string} workflow_run_id
 * @returns {Promise<object[]>}
 */
export async function listWorkflowStepsForWaterfall(pool, workflow_run_id) {
  const res = await pool.query(
    `SELECT step_id, started_at, ended_at, status
       FROM workflow_steps
      WHERE workflow_run_id=$1
        AND step_id IN ('impl_be','impl_fe','qa_verify','release_pack')
      ORDER BY started_at ASC NULLS LAST`,
    [workflow_run_id]
  );
  return res.rows;
}

/**
 * Retrieve the most recent N duration_ms values for a given stage.
 * Used for P50/P95 computation across runs.
 *
 * @param {import('pg').Pool} pool
 * @param {string} stage
 * @param {number} limit
 * @returns {Promise<number[]>}  sorted ascending
 */
export async function listStageDurationsForPercentiles(pool, stage, limit = 200) {
  const res = await pool.query(
    `SELECT duration_ms FROM waterfall_stage_log
      WHERE stage=$1 AND duration_ms IS NOT NULL
      ORDER BY started_at DESC LIMIT $2`,
    [stage, limit]
  );
  return res.rows.map((r) => Number(r.duration_ms)).sort((a, b) => a - b);
}

/**
 * Look up workflow_run_id for a given run_id (to correlate explicit stages with
 * workflow steps).
 *
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @returns {Promise<string|null>}
 */
export async function getWorkflowRunIdForRun(pool, run_id) {
  const res = await pool.query(
    `SELECT workflow_run_id FROM workflow_runs WHERE run_id=$1 ORDER BY created_at DESC LIMIT 1`,
    [run_id]
  );
  return res.rows[0]?.workflow_run_id ?? null;
}
