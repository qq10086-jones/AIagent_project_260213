/**
 * Data-access layer for event_log queries used by timeline/status APIs.
 */

/**
 * @param {import('pg').Pool} pool
 * @param {string[]} taskIds
 */
export async function listEventsForTaskIds(pool, taskIds = []) {
  const normalized = Array.isArray(taskIds) ? taskIds.filter(Boolean) : [];
  if (normalized.length === 0) return [];
  const evRes = await pool.query(
    "SELECT task_id, event_type, payload_json, ts FROM event_log WHERE task_id = ANY($1::text[]) ORDER BY ts ASC",
    [normalized]
  );
  return evRes.rows || [];
}
