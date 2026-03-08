import { buildWorkflowRuntimeNotification } from "./workflow_runtime_notifier.js";

export async function deliverWorkflowRuntimeNotification({
  workflowTerminal = null,
  notifyCtx = null,
  discord = null,
  workflowEngine = null,
  status = "",
  normalizedErrorCode = "",
  streamError = "",
  output = {},
  safeTranslate = async (text) => text,
  replyChunked = async (channel, text) => channel?.send?.(text),
}) {
  if (!workflowTerminal?.handled || !notifyCtx?.channelId || !discord?.channels?.fetch || !workflowEngine) {
    return { delivered: false, reason: "missing_runtime_dependencies" };
  }

  const notifyChannel = await discord.channels.fetch(notifyCtx.channelId).catch(() => null);
  if (!notifyChannel || typeof notifyChannel.send !== "function") {
    return { delivered: false, reason: "missing_channel" };
  }

  const workflowState = workflowTerminal.workflow_run_id
    ? await workflowEngine.getWorkflowRunStatus(workflowTerminal.workflow_run_id).catch(() => null)
    : null;
  const runtimeNotification = buildWorkflowRuntimeNotification({
    workflowState,
    workflowTerminal,
    status,
    normalizedErrorCode,
    streamError,
    output: output || {},
  });
  const runtimeMsg = await safeTranslate(runtimeNotification.message, notifyCtx.lang || "zh");
  if (runtimeNotification.kind === "failure") {
    await replyChunked(notifyChannel, `[NEXUS] ${runtimeMsg}`);
  } else {
    await notifyChannel.send(`[NEXUS] ${runtimeMsg}`);
  }

  return {
    delivered: true,
    kind: runtimeNotification.kind,
    workflow_run_id: workflowTerminal.workflow_run_id || "",
  };
}
