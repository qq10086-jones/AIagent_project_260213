export function createTaskLifecycle({ taskId, msgId, toolName, runId, emitResult, writeFact, ackMessage, auditHooks = null, workerName = "worker-coder", taskEnvelope = null }) {
  let finalized = false;
  let aborted = false;

  async function runAudit(callback) {
    try {
      await callback?.();
    } catch {
      /* audit hooks are best-effort and must not block task finalization */
    }
  }

  async function finalizeWithFallback({ factPayload, resultStatus, resultOutput, error }) {
    if (finalized) return false;
    finalized = true;
    aborted = true;

    try {
      await writeFact(runId, "coder", factPayload);
    } catch {
      /* writeFact is best effort */
    }

    let emitError = null;
    try {
      await emitResult(taskId, resultStatus, resultOutput, error || null);
    } catch (err) {
      emitError = String(err?.message || err || "emit result failed");
      try {
        await emitResult(
          taskId,
          "failed",
          {
            error: emitError,
            plan: "failed_during_result_emit",
            original_status: resultStatus,
          },
          emitError,
        );
      } catch {
        /* no-op: the task must still be acked to avoid permanent pending backlog */
      }
    }

    try {
      await ackMessage(msgId);
    } catch {
      /* no-op: worker loop recovery will reclaim stale messages */
    }
    return true;
  }

  return {
    isAborted() {
      return aborted;
    },

    async emitClaimed() {
      if (aborted) return false;
      await runAudit(() => auditHooks?.onTaskStart?.(runId, taskEnvelope || { task_id: taskId, tool_name: toolName }, { taskId, workerName, toolName }));
      await emitResult(taskId, "claimed");
      return true;
    },

    async finalizeResult({ ok, output, error }) {
      const eventPayload = output?.worker_result || output || { ok, error };
      await runAudit(() => auditHooks?.onTaskComplete?.(runId, eventPayload, { taskId, workerName, toolName }));
      return finalizeWithFallback({
        factPayload: { tool_name: toolName, output, success: ok },
        resultStatus: ok ? "succeeded" : "failed",
        resultOutput: output,
        error,
      });
    },

    async finalizeExecutionFailure(error) {
      const errorMessage = String(error?.message || error || "unknown worker failure");
      await runAudit(() => auditHooks?.onTaskError?.(runId, { message: errorMessage }, { taskId, workerName, toolName }));
      return finalizeWithFallback({
        factPayload: { tool_name: toolName, error: errorMessage, success: false },
        resultStatus: "failed",
        resultOutput: { error: errorMessage, plan: "failed_during_execution" },
        error: errorMessage,
      });
    },

    async finalizeTimeout(error) {
      const errorMessage = String(error?.message || error || "GLOBAL_TASK_TIMEOUT");
      await runAudit(() => auditHooks?.onTaskError?.(runId, { message: errorMessage, timed_out: true }, { taskId, workerName, toolName }));
      return finalizeWithFallback({
        factPayload: { tool_name: toolName, error: errorMessage, success: false, timed_out: true },
        resultStatus: "failed",
        resultOutput: { error: errorMessage, plan: "failed_during_execution" },
        error: errorMessage,
      });
    },

    async acknowledgeOnly() {
      if (finalized) return false;
      finalized = true;
      aborted = true;
      await ackMessage(msgId);
      return true;
    },
  };
}
