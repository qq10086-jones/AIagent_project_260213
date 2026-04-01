const CREATE_EXECUTION_AUDIT_LOG_SQL = `
CREATE TABLE IF NOT EXISTS execution_audit_log (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  task_id TEXT,
  worker_name TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_execution_audit_log_run_id ON execution_audit_log(run_id);
CREATE INDEX IF NOT EXISTS idx_execution_audit_log_task_id ON execution_audit_log(task_id);
CREATE INDEX IF NOT EXISTS idx_execution_audit_log_event_type ON execution_audit_log(event_type);
`;

function asJson(value) {
  if (value && typeof value === "object") return value;
  return {};
}

export function createAuditHooks({ pool, logger = console, ensureSchema = true } = {}) {
  let schemaReady = !ensureSchema;
  let schemaPromise = null;

  async function ensureAuditSchema() {
    if (!pool || schemaReady) return;
    if (!schemaPromise) {
      schemaPromise = pool.query(CREATE_EXECUTION_AUDIT_LOG_SQL)
        .then(() => {
          schemaReady = true;
        })
        .catch((err) => {
          logger.warn?.("[audit_hooks] schema init failed:", err?.message || err);
        });
    }
    await schemaPromise;
  }

  async function insertAuditEvent({ runId, taskId = null, workerName, eventType, payload = {} }) {
    if (!pool || !runId || !workerName || !eventType) return false;
    try {
      await ensureAuditSchema();
      await pool.query(
        `INSERT INTO execution_audit_log (run_id, task_id, worker_name, event_type, payload_json)
         VALUES ($1, $2, $3, $4, $5::jsonb)`,
        [String(runId), taskId ? String(taskId) : null, String(workerName), String(eventType), JSON.stringify(asJson(payload))],
      );
      return true;
    } catch (err) {
      logger.warn?.("[audit_hooks] insert failed:", err?.message || err);
      return false;
    }
  }

  return {
    ensureAuditSchema,
    onTaskStart(runId, taskEnvelope = {}, context = {}) {
      return insertAuditEvent({
        runId,
        taskId: context.taskId || taskEnvelope.task_id || null,
        workerName: context.workerName || "unknown",
        eventType: "task_start",
        payload: {
          task_envelope: taskEnvelope,
          tool_name: context.toolName || taskEnvelope.tool_name || null,
        },
      });
    },
    onToolCall(runId, toolName, input = {}, context = {}) {
      return insertAuditEvent({
        runId,
        taskId: context.taskId || null,
        workerName: context.workerName || "unknown",
        eventType: "tool_call",
        payload: {
          tool_name: toolName || null,
          input,
          permission_advice: context.permissionAdvice || null,
        },
      });
    },
    onToolResult(runId, toolName, resultSummary = {}, context = {}) {
      return insertAuditEvent({
        runId,
        taskId: context.taskId || null,
        workerName: context.workerName || "unknown",
        eventType: "tool_result",
        payload: {
          tool_name: toolName || null,
          result_summary: resultSummary,
        },
      });
    },
    onTaskComplete(runId, workerResult = {}, context = {}) {
      return insertAuditEvent({
        runId,
        taskId: context.taskId || null,
        workerName: context.workerName || "unknown",
        eventType: "task_complete",
        payload: {
          worker_result: workerResult,
        },
      });
    },
    onTaskError(runId, error = {}, context = {}) {
      return insertAuditEvent({
        runId,
        taskId: context.taskId || null,
        workerName: context.workerName || "unknown",
        eventType: "task_error",
        payload: {
          error,
        },
      });
    },
  };
}
