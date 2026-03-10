/**
 * Query helpers for latest agent facts.
 */

export async function findLatestFactForRun(pool, { run_id, agent_name, tool_name = "" }) {
  const runId = String(run_id || "").trim();
  const agentName = String(agent_name || "").trim();
  const toolName = String(tool_name || "").trim();
  if (!runId || !agentName) return null;

  let sql = `
    SELECT fact_id, run_id, agent_name, kind, payload_json, created_at
    FROM fact_items
    WHERE run_id = $1 AND agent_name = $2
  `;
  const params = [runId, agentName];
  if (toolName) {
    sql += ` AND payload_json::text LIKE $3`;
    params.push(`%"${toolName}"%`);
  }
  sql += ` ORDER BY created_at DESC LIMIT 1`;

  const result = await pool.query(sql, params);
  return result.rows?.[0] || null;
}
