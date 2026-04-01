/**
 * Query helpers for execution audit evidence chains.
 */

function normalizeEvidenceId(value) {
  const safe = String(value || "").trim();
  return safe || "";
}

export async function listExecutionAuditEventsByEvidenceId(pool, evidence_id) {
  const evidenceId = normalizeEvidenceId(evidence_id);
  if (!evidenceId) return [];

  const result = await pool.query(
    `SELECT
       id,
       run_id,
       task_id,
       worker_name,
       event_type,
       payload_json,
       created_at
     FROM execution_audit_log
     WHERE COALESCE(
       payload_json->>'evidence_id',
       payload_json->'worker_result'->'metadata'->>'evidence_id',
       payload_json->'task_envelope'->>'evidence_id',
       payload_json->'input'->>'evidence_id'
     ) = $1
     ORDER BY created_at ASC, id ASC`,
    [evidenceId],
  );
  return result.rows || [];
}
