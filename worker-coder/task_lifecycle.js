export function createTaskLifecycle({ taskId, msgId, toolName, runId, emitResult, writeFact, ackMessage }) {
  let finalized = false;
  let aborted = false;

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
      if (finalized) return false;
      finalized = true;
      aborted = true;
      await writeFact(runId, "coder", { tool_name: toolName, output, success: ok });
      await emitResult(taskId, ok ? "succeeded" : "failed", output, error || null);
      await ackMessage(msgId);
      return true;
    },

    async finalizeExecutionFailure(error) {
      if (finalized) return false;
      finalized = true;
      aborted = true;
      const errorMessage = String(error?.message || error || "unknown worker failure");
      await writeFact(runId, "coder", { tool_name: toolName, error: errorMessage, success: false });
      await emitResult(taskId, "failed", { error: errorMessage, plan: "failed_during_execution" }, errorMessage);
      await ackMessage(msgId);
      return true;
    },

    async finalizeTimeout(error) {
      if (finalized) return false;
      finalized = true;
      aborted = true;
      const errorMessage = String(error?.message || error || "GLOBAL_TASK_TIMEOUT");
      await writeFact(runId, "coder", { tool_name: toolName, error: errorMessage, success: false, timed_out: true });
      await emitResult(taskId, "failed", { error: errorMessage, plan: "failed_during_execution" }, errorMessage);
      await ackMessage(msgId);
      return true;
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
