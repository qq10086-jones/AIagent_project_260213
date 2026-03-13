export function createTaskLifecycle({ taskId, msgId, toolName, runId, emitResult, writeFact, ackMessage }) {
  let finalized = false;
  let aborted = false;

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
      await emitResult(taskId, "claimed");
      return true;
    },

    async finalizeResult({ ok, output, error }) {
      return finalizeWithFallback({
        factPayload: { tool_name: toolName, output, success: ok },
        resultStatus: ok ? "succeeded" : "failed",
        resultOutput: output,
        error,
      });
    },

    async finalizeExecutionFailure(error) {
      const errorMessage = String(error?.message || error || "unknown worker failure");
      return finalizeWithFallback({
        factPayload: { tool_name: toolName, error: errorMessage, success: false },
        resultStatus: "failed",
        resultOutput: { error: errorMessage, plan: "failed_during_execution" },
        error: errorMessage,
      });
    },

    async finalizeTimeout(error) {
      const errorMessage = String(error?.message || error || "GLOBAL_TASK_TIMEOUT");
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
