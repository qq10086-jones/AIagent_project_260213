function truncate(text, maxLen = 800) {
  const safe = String(text || "").trim();
  if (!safe) return "";
  return safe.length <= maxLen ? safe : `${safe.slice(0, maxLen)}...`;
}

function summarizeQuantOutput(output = {}) {
  return truncate(
    output.briefing
    || output.analysis
    || output.summary
    || output.message
    || output.error
    || "",
  );
}

function summarizeGenericOutput(output = {}) {
  return truncate(
    output.summary
    || output.message
    || output.result_summary
    || output.error
    || "",
    240,
  );
}

function summarizePiPayload(event = {}) {
  return truncate(
    event?.text
    || event?.title
    || event?.status
    || event?.message
    || "",
    240,
  );
}

export function buildRuntimeUpdateMessage({ taskId, status, toolName, output = {}, streamError = "" }) {
  const safeTool = String(toolName || "unknown");
  if (status === "claimed") {
    return {
      kind: "progress",
      text: `[NEXUS] ${safeTool} started (task_id=${String(taskId || "")})`,
      shouldSkipDefault: false,
    };
  }

  if (status === "tool_call") {
    const inputSummary = truncate(output?.input_summary || output?.summary || "", 180);
    return {
      kind: "progress",
      text: inputSummary
        ? `[NEXUS] ${safeTool} running\n${inputSummary}`
        : `[NEXUS] ${safeTool} running`,
      shouldSkipDefault: false,
    };
  }

  if (status === "tool_result") {
    const summary = safeTool.startsWith("quant.")
      ? summarizeQuantOutput(output)
      : summarizeGenericOutput(output);
    return {
      kind: "progress",
      text: summary
        ? `[NEXUS] ${safeTool} progress\n${summary}`
        : `[NEXUS] ${safeTool} progress update`,
      shouldSkipDefault: false,
    };
  }

  if (safeTool.startsWith("quant.")) {
    if (status === "succeeded") {
      const summary = summarizeQuantOutput(output) || "quant task completed";
      return {
        kind: "quant_terminal",
        text: `[NEXUS] ${safeTool} completed\n${summary}`,
        shouldSkipDefault: true,
      };
    }
    const failure = truncate(streamError || output?.error || "quant task failed");
    return {
      kind: "quant_terminal",
      text: `[NEXUS] ${safeTool} failed\n${failure}`,
      shouldSkipDefault: true,
    };
  }

  return {
    kind: "noop",
    text: "",
    shouldSkipDefault: false,
  };
}

export function buildPiRuntimeUpdateMessage({ taskId, event = {} }) {
  const eventType = String(event?.type || "").trim();
  const tag = String(event?.tag || "").trim();
  const toolTitle = truncate(event?.title || event?.text || "tool activity", 180);

  if (eventType === "tool_call" || tag === "tool_call") {
    return {
      kind: "progress",
      text: `[NEXUS] tool running (task_id=${String(taskId || "")})\n${toolTitle}`,
      shouldSkipDefault: false,
    };
  }

  if (eventType === "status" && tag === "tool_call_update") {
    const statusText = summarizePiPayload(event) || "tool progress update";
    return {
      kind: "progress",
      text: `[NEXUS] tool progress\n${statusText}`,
      shouldSkipDefault: false,
    };
  }

  if (eventType === "text_delta" && tag === "agent_message_chunk") {
    const chunk = truncate(event?.text || "", 240);
    return {
      kind: "progress",
      text: chunk ? `[NEXUS] agent update\n${chunk}` : "",
      shouldSkipDefault: false,
    };
  }

  if (eventType === "text_delta" && tag === "agent_thought_chunk") {
    const chunk = truncate(event?.text || "", 180);
    return {
      kind: "progress",
      text: chunk ? `[NEXUS] reasoning\n${chunk}` : "",
      shouldSkipDefault: false,
    };
  }

  if (eventType === "status" && tag === "usage_update") {
    const usageText = summarizePiPayload(event) || "usage updated";
    return {
      kind: "progress",
      text: `[NEXUS] usage\n${usageText}`,
      shouldSkipDefault: false,
    };
  }

  return {
    kind: "noop",
    text: "",
    shouldSkipDefault: false,
  };
}

export function createStreamAdapter({
  discord = null,
  safeTranslate = async (text) => text,
  replyChunked = async (channel, text) => channel?.send?.(text),
  logger = console,
  throttleMs = 1000,
} = {}) {
  const throttleState = new Map();

  function shouldThrottle(key) {
    const now = Date.now();
    const last = Number(throttleState.get(key) || 0);
    if (now - last < throttleMs) {
      return true;
    }
    throttleState.set(key, now);
    return false;
  }

  async function sendTaskUpdate({ ctx = null, taskId = "", status = "", toolName = "", output = {}, streamError = "" }) {
    if (!ctx?.channelId || !discord?.channels?.fetch) {
      return { delivered: false, reason: "missing_context", shouldSkipDefault: false, sentMessages: [] };
    }
    const update = buildRuntimeUpdateMessage({ taskId, status, toolName, output, streamError });
    if (!update.text) {
      return { delivered: false, reason: "no_message", shouldSkipDefault: false, sentMessages: [] };
    }
    const throttleKey = `${ctx.channelId}:${taskId}:${update.kind}:${status}`;
    if (shouldThrottle(throttleKey)) {
      return { delivered: false, reason: "throttled", shouldSkipDefault: false, sentMessages: [] };
    }
    const channel = await discord.channels.fetch(ctx.channelId).catch(() => null);
    if (!channel || typeof channel.send !== "function") {
      return { delivered: false, reason: "missing_channel", shouldSkipDefault: false, sentMessages: [] };
    }
    try {
      const translated = await safeTranslate(update.text, ctx.lang || "zh");
      const sentMessages = update.kind === "quant_terminal"
        ? await replyChunked(channel, translated)
        : [await channel.send(translated)];
      return { delivered: true, reason: "sent", shouldSkipDefault: update.shouldSkipDefault, sentMessages };
    } catch (err) {
      logger.warn?.("[stream_adapter] send failed:", err?.message || err);
      return { delivered: false, reason: "send_failed", shouldSkipDefault: false, sentMessages: [] };
    }
  }

  async function sendPiSessionUpdate({ ctx = null, taskId = "", event = {} }) {
    if (!ctx?.channelId || !discord?.channels?.fetch) {
      return { delivered: false, reason: "missing_context", shouldSkipDefault: false, sentMessages: [] };
    }
    const update = buildPiRuntimeUpdateMessage({ taskId, event });
    if (!update.text) {
      return { delivered: false, reason: "no_message", shouldSkipDefault: false, sentMessages: [] };
    }
    const throttleKey = `${ctx.channelId}:${taskId}:${update.kind}:${String(event?.type || "")}:${String(event?.tag || "")}`;
    if (shouldThrottle(throttleKey)) {
      return { delivered: false, reason: "throttled", shouldSkipDefault: false, sentMessages: [] };
    }
    const channel = await discord.channels.fetch(ctx.channelId).catch(() => null);
    if (!channel || typeof channel.send !== "function") {
      return { delivered: false, reason: "missing_channel", shouldSkipDefault: false, sentMessages: [] };
    }
    try {
      const translated = await safeTranslate(update.text, ctx.lang || "zh");
      const sentMessages = [await channel.send(translated)];
      return { delivered: true, reason: "sent", shouldSkipDefault: update.shouldSkipDefault, sentMessages };
    } catch (err) {
      logger.warn?.("[stream_adapter] pi session send failed:", err?.message || err);
      return { delivered: false, reason: "send_failed", shouldSkipDefault: false, sentMessages: [] };
    }
  }

  return {
    sendTaskUpdate,
    sendPiSessionUpdate,
  };
}
