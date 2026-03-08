/**
 * run_repository.js
 *
 * Data-access layer for the `runs` table.
 * All exported functions accept `pool` (pg.Pool) as their first argument
 * so they can be unit-tested without relying on module-level singletons.
 */

/**
 * Insert or update a run row.
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @param {{ client_msg_id: string, user_id: string, status: string, input_text: string }} opts
 */
export async function ensureRun(pool, run_id, { client_msg_id, user_id, status, input_text }) {
  await pool.query(
    `INSERT INTO runs(run_id, client_msg_id, user_id, status, input_text)
     VALUES ($1, $2, $3, $4, $5)
     ON CONFLICT (run_id) DO UPDATE
     SET status = EXCLUDED.status,
         input_text = COALESCE(NULLIF(EXCLUDED.input_text, ''), runs.input_text)`,
    [run_id, client_msg_id, user_id, status, input_text]
  );
}

/**
 * Update a run status row.
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @param {string} status
 */
export async function updateRunStatus(pool, run_id, status) {
  await pool.query("UPDATE runs SET status=$1 WHERE run_id=$2", [status, run_id]);
}

/**
 * Update a run status only if current status is not failed.
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @param {string} status
 */
export async function updateRunStatusIfNotFailed(pool, run_id, status) {
  await pool.query(
    "UPDATE runs SET status=$1 WHERE run_id=$2 AND COALESCE(status,'') <> 'failed'",
    [status, run_id]
  );
}

/**
 * Update completed run state and persist cost ledger.
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @param {object} costLedger
 */
export async function completeRunWithCostLedger(pool, run_id, costLedger = {}) {
  await pool.query(
    "UPDATE runs SET status=$1, cost_ledger_json=$2 WHERE run_id=$3",
    ["completed", JSON.stringify(costLedger || {}), run_id]
  );
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} client_msg_id
 */
export async function findRunIdByClientMsgId(pool, client_msg_id) {
  const row = await pool.query("SELECT run_id FROM runs WHERE client_msg_id = $1", [client_msg_id]);
  return row.rows[0]?.run_id || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function getRunById(pool, run_id) {
  const row = await pool.query("SELECT * FROM runs WHERE run_id=$1", [run_id]);
  return row.rows[0] || null;
}

/**
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 */
export async function getRunInputText(pool, run_id) {
  const row = await pool.query("SELECT input_text FROM runs WHERE run_id=$1", [run_id]);
  return row.rows[0]?.input_text || "";
}
