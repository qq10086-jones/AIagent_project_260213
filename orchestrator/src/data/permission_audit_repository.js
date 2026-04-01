const CREATE_PERMISSION_AUDIT_LOG_SQL = `
CREATE TABLE IF NOT EXISTS permission_audit_log (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  task_id TEXT,
  tool_name TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  council_advice TEXT NOT NULL,
  safety_verdict TEXT,
  context_verdict TEXT,
  risk_score DOUBLE PRECISION,
  advisory_summary TEXT,
  final_human_decision TEXT,
  duration_ms INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_permission_audit_log_run_id ON permission_audit_log(run_id);
CREATE INDEX IF NOT EXISTS idx_permission_audit_log_risk_level ON permission_audit_log(risk_level);
CREATE INDEX IF NOT EXISTS idx_permission_audit_log_human_decision ON permission_audit_log(final_human_decision);
`;

let schemaReady = false;
let schemaPromise = null;

async function ensurePermissionAuditSchema(pool) {
  if (!pool || schemaReady) return;
  if (!schemaPromise) {
    schemaPromise = pool.query(CREATE_PERMISSION_AUDIT_LOG_SQL).then(() => {
      schemaReady = true;
    });
  }
  await schemaPromise;
}

export async function insertPermissionAuditRecord(pool, record = {}) {
  await ensurePermissionAuditSchema(pool);
  await pool.query(
    `INSERT INTO permission_audit_log (
       run_id, task_id, tool_name, risk_level, council_advice,
       safety_verdict, context_verdict, risk_score, advisory_summary,
       final_human_decision, duration_ms
     ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
    [
      String(record.run_id || ""),
      record.task_id ? String(record.task_id) : null,
      String(record.tool_name || ""),
      String(record.risk_level || "low"),
      String(record.council_advice || "review"),
      record.safety_verdict ? String(record.safety_verdict) : null,
      record.context_verdict ? String(record.context_verdict) : null,
      Number(record.risk_score || 0),
      record.advisory_summary ? String(record.advisory_summary) : null,
      record.final_human_decision ? String(record.final_human_decision) : null,
      Math.max(0, Number(record.duration_ms || 0)),
    ],
  );
}

export async function listPermissionAuditRecords(pool, { run_id = "", risk_level = "", final_human_decision = "", limit = 50 } = {}) {
  await ensurePermissionAuditSchema(pool);
  const params = [];
  const clauses = [];
  if (String(run_id || "").trim()) {
    params.push(String(run_id).trim());
    clauses.push(`run_id = $${params.length}`);
  }
  if (String(risk_level || "").trim()) {
    params.push(String(risk_level).trim());
    clauses.push(`risk_level = $${params.length}`);
  }
  if (String(final_human_decision || "").trim()) {
    params.push(String(final_human_decision).trim());
    clauses.push(`final_human_decision = $${params.length}`);
  }
  params.push(Math.max(1, Math.min(Number(limit || 50), 200)));
  const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
  const result = await pool.query(
    `SELECT *
     FROM permission_audit_log
     ${where}
     ORDER BY created_at DESC, id DESC
     LIMIT $${params.length}`,
    params,
  );
  return result.rows || [];
}

export async function updatePermissionAuditHumanDecision(pool, { task_id = "", final_human_decision = "" } = {}) {
  await ensurePermissionAuditSchema(pool);
  const taskId = String(task_id || "").trim();
  const humanDecision = String(final_human_decision || "").trim();
  if (!taskId || !humanDecision) return 0;
  const result = await pool.query(
    `UPDATE permission_audit_log
     SET final_human_decision = $2
     WHERE id = (
       SELECT id
       FROM permission_audit_log
       WHERE task_id = $1
       ORDER BY created_at DESC, id DESC
       LIMIT 1
     )`,
    [taskId, humanDecision],
  );
  return Number(result.rowCount || 0);
}

function clampSummaryDays(days) {
  const safe = Number(days || 30);
  if (!Number.isFinite(safe)) return 30;
  return Math.max(1, Math.min(Math.trunc(safe), 365));
}

function toRatio(count, total) {
  const safeTotal = Number(total || 0);
  if (safeTotal <= 0) return 0;
  return Number((Number(count || 0) / safeTotal).toFixed(4));
}

export async function getPermissionAuditSummary(pool, { days = 30, risk_level = "" } = {}) {
  await ensurePermissionAuditSchema(pool);
  const safeDays = clampSummaryDays(days);
  const params = [safeDays];
  const clauses = [`created_at >= NOW() - ($1::text || ' days')::interval`];
  if (String(risk_level || "").trim()) {
    params.push(String(risk_level).trim());
    clauses.push(`risk_level = $${params.length}`);
  }

  const where = `WHERE ${clauses.join(" AND ")}`;
  const result = await pool.query(
    `SELECT
       COUNT(*)::int AS total_records,
       COUNT(*) FILTER (WHERE final_human_decision IS NOT NULL)::int AS reviewed_records,
       COUNT(*) FILTER (WHERE council_advice = 'allow')::int AS advice_allow_count,
       COUNT(*) FILTER (WHERE council_advice = 'review')::int AS advice_review_count,
       COUNT(*) FILTER (WHERE council_advice = 'deny')::int AS advice_deny_count,
       COUNT(*) FILTER (WHERE final_human_decision = 'approved')::int AS human_approved_count,
       COUNT(*) FILTER (WHERE final_human_decision = 'rejected')::int AS human_rejected_count,
       COUNT(*) FILTER (
         WHERE final_human_decision IS NOT NULL
           AND (
             (council_advice = 'allow' AND final_human_decision = 'approved')
             OR (council_advice = 'deny' AND final_human_decision = 'rejected')
           )
       )::int AS aligned_decision_count,
       COUNT(*) FILTER (
         WHERE final_human_decision IS NOT NULL
           AND council_advice IN ('allow', 'deny')
       )::int AS comparable_decision_count,
       COUNT(*) FILTER (
         WHERE final_human_decision IS NOT NULL
           AND (
             (council_advice = 'allow' AND final_human_decision = 'rejected')
             OR (council_advice = 'deny' AND final_human_decision = 'approved')
           )
       )::int AS override_count,
       COUNT(*) FILTER (
         WHERE council_advice = 'allow'
           AND final_human_decision = 'rejected'
       )::int AS false_negative_count,
       COUNT(*) FILTER (
         WHERE council_advice = 'review'
           AND final_human_decision IS NOT NULL
       )::int AS review_escalation_count
     FROM permission_audit_log
     ${where}`,
    params,
  );

  const row = result.rows?.[0] || {};
  const totalRecords = Number(row.total_records || 0);
  const reviewedRecords = Number(row.reviewed_records || 0);
  const comparableDecisionCount = Number(row.comparable_decision_count || 0);
  const alignedDecisionCount = Number(row.aligned_decision_count || 0);
  const overrideCount = Number(row.override_count || 0);

  return {
    window_days: safeDays,
    risk_level: String(risk_level || "").trim() || null,
    total_records: totalRecords,
    reviewed_records: reviewedRecords,
    comparable_decision_count: comparableDecisionCount,
    aligned_decision_count: alignedDecisionCount,
    override_count: overrideCount,
    false_negative_count: Number(row.false_negative_count || 0),
    review_escalation_count: Number(row.review_escalation_count || 0),
    advice_breakdown: {
      allow: Number(row.advice_allow_count || 0),
      review: Number(row.advice_review_count || 0),
      deny: Number(row.advice_deny_count || 0),
    },
    human_decision_breakdown: {
      approved: Number(row.human_approved_count || 0),
      rejected: Number(row.human_rejected_count || 0),
      pending: Math.max(0, totalRecords - reviewedRecords),
    },
    rates: {
      advice_allow_rate: toRatio(row.advice_allow_count, totalRecords),
      advice_review_rate: toRatio(row.advice_review_count, totalRecords),
      advice_deny_rate: toRatio(row.advice_deny_count, totalRecords),
      reviewed_rate: toRatio(reviewedRecords, totalRecords),
      alignment_rate: toRatio(alignedDecisionCount, comparableDecisionCount),
      override_rate: toRatio(overrideCount, comparableDecisionCount),
      false_negative_rate: toRatio(row.false_negative_count, totalRecords),
    },
  };
}
