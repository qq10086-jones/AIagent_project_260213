/**
 * routing_audit_repository.js  (Layer 4 — Infra / Data)
 *
 * WS-30-01: Routing Decision Audit Log
 * CRUD helpers for the `routing_decision_log` table.
 *
 * All exported functions accept `pool` (pg.Pool) as their first argument
 * so they can be unit-tested with a stub.
 *
 * Design authority: OpenClaw_Nexus_Design_Document_v4.md § 5.3
 */

/**
 * @param {import('pg').Pool} pool
 * @param {object} record - Full routing decision record (see domain service for field mapping)
 * @returns {Promise<string>} The log_id persisted
 */
export async function insertRoutingDecision(pool, record) {
  await pool.query(
    `INSERT INTO routing_decision_log (
       log_id, run_id, workflow_run_id, workflow_id,
       router_mode, dynamic_routing_enabled,
       classifier_version, classifier_confidence, classifier_confidence_band,
       classifier_work_shape, classifier_domain_lead, classifier_parallel_candidate,
       classifier_model_tier, classifier_deny_or_degrade_reason,
       feature_snapshot_ref,
       routing_decision_source, final_execution_decision, safety_override_result,
       decision_json, created_at
     ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,NOW())
     ON CONFLICT (log_id) DO NOTHING`,
    [
      record.log_id,
      record.run_id,
      record.workflow_run_id ?? null,
      record.workflow_id ?? null,
      record.router_mode ?? null,
      record.dynamic_routing_enabled ?? null,
      record.classifier_version ?? null,
      record.classifier_confidence ?? null,
      record.classifier_confidence_band ?? null,
      record.classifier_work_shape ?? null,
      record.classifier_domain_lead ?? null,
      record.classifier_parallel_candidate ?? null,
      record.classifier_model_tier ?? null,
      record.classifier_deny_or_degrade_reason ?? null,
      record.feature_snapshot_ref ?? null,
      record.routing_decision_source,
      record.final_execution_decision,
      record.safety_override_result ?? null,
      JSON.stringify(record),
    ]
  );
  return record.log_id;
}

/**
 * Retrieve all routing decisions for a given run_id, ordered chronologically.
 * Satisfies the WS-30-01 "query API can retrieve decision history per run" criterion.
 *
 * @param {import('pg').Pool} pool
 * @param {string} run_id
 * @returns {Promise<object[]>}
 */
export async function listRoutingDecisionsByRunId(pool, run_id) {
  const res = await pool.query(
    `SELECT * FROM routing_decision_log WHERE run_id=$1 ORDER BY created_at ASC`,
    [run_id]
  );
  return res.rows;
}

/**
 * Retrieve the most recent routing decision for a workflow_run_id.
 * Used for incident review and closure evidence queries.
 *
 * @param {import('pg').Pool} pool
 * @param {string} workflow_run_id
 * @returns {Promise<object|null>}
 */
export async function getLatestRoutingDecision(pool, workflow_run_id) {
  const res = await pool.query(
    `SELECT * FROM routing_decision_log WHERE workflow_run_id=$1 ORDER BY created_at DESC LIMIT 1`,
    [workflow_run_id]
  );
  return res.rows[0] ?? null;
}

/**
 * Aggregate routing decision counts grouped by key dimensions.
 * Used by WS-30-03 to compute routing precision, misroute rates, and fallback ratios.
 *
 * @param {import('pg').Pool} pool
 * @param {{ from?: Date|null, to?: Date|null }} opts
 * @returns {Promise<Array<{ routing_decision_source, final_execution_decision, classifier_confidence_band, classifier_work_shape, classifier_model_tier, count: number }>>}
 */
export async function queryDecisionStats(pool, { from = null, to = null } = {}) {
  const params = [];
  const conditions = [];
  if (from) { params.push(from); conditions.push(`created_at >= $${params.length}`); }
  if (to)   { params.push(to);   conditions.push(`created_at <= $${params.length}`); }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

  const res = await pool.query(
    `SELECT
       routing_decision_source,
       final_execution_decision,
       classifier_confidence_band,
       classifier_work_shape,
       classifier_model_tier,
       COUNT(*)::int AS count
     FROM routing_decision_log
     ${where}
     GROUP BY
       routing_decision_source,
       final_execution_decision,
       classifier_confidence_band,
       classifier_work_shape,
       classifier_model_tier`,
    params
  );
  return res.rows;
}
