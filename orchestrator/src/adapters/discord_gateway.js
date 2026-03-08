import { AttachmentBuilder, Client, EmbedBuilder, GatewayIntentBits } from "discord.js";

const DISCORD_MAX_CONTENT = 1900;

function splitForDiscord(text, maxLen = DISCORD_MAX_CONTENT) {
  const normalized = String(text || "").replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];
  if (normalized.length <= maxLen) return [normalized];

  const out = [];
  let rest = normalized;
  while (rest.length > maxLen) {
    let cut = rest.lastIndexOf("\n\n", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = rest.lastIndexOf("\n", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = rest.lastIndexOf(" ", maxLen);
    if (cut < Math.floor(maxLen * 0.5)) cut = maxLen;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trimStart();
  }
  if (rest) out.push(rest);
  return out;
}

export function createDiscordGateway({ translate }) {
  const client = new Client({
    intents: [
      GatewayIntentBits.Guilds,
      GatewayIntentBits.GuildMessages,
      GatewayIntentBits.MessageContent,
      GatewayIntentBits.GuildMessageReactions,
    ],
  });

  const taskToContext = new Map();
  const runToContext = new Map();
  const workflowRunToContext = new Map();

  client.on("error", err => console.error("[discord] Client error:", err.message));
  client.on("clientReady", () => console.log(`[discord] Logged in as ${client.user.tag}`));

  async function replyChunked(msg, text, header = "") {
    const merged = header ? `${header}\n${text || ""}` : String(text || "");
    const chunks = splitForDiscord(merged);
    if (chunks.length === 0) return [];
    const sentMsgs = [];
    for (const chunk of chunks) {
      if (msg && typeof msg.reply === "function") {
        sentMsgs.push(await msg.reply(chunk));
      } else if (msg && typeof msg.send === "function") {
        sentMsgs.push(await msg.send(chunk));
      } else {
        throw new Error("replyChunked target has neither reply() nor send()");
      }
    }
    return sentMsgs;
  }

  async function safeTranslate(text, lang = "zh") {
    const raw = String(text ?? "");
    try {
      return await translate(raw, lang || "zh");
    } catch (err) {
      console.warn("[translate] fallback to raw text:", err?.message || err);
      return raw;
    }
  }

  function bindTaskToContext(task_id, context, tool_name) {
    if (!context) return;
    if (!context.pendingTaskIds) context.pendingTaskIds = new Set();
    if (!Number.isFinite(context.totalTaskCount) || context.totalTaskCount <= 0) context.totalTaskCount = 1;
    context.pendingTaskIds.add(task_id);
    runToContext.set(context.run_id, context);
    taskToContext.set(task_id, {
      channelId: context.channelId,
      startTime: context.startTime || Date.now(),
      lang: context.lang || "zh",
      run_id: context.run_id,
      closeRunOnTaskResult: Boolean(context.closeRunOnTaskResult),
      pendingTaskIds: context.pendingTaskIds,
      totalTaskCount: context.totalTaskCount || 1,
      tool_name: tool_name || context.tool_name || "unknown",
    });
  }

  async function sendStepTransitionNotification({ event, workflow_run_id, ...rest }) {
    console.log(
      `[step_transition] event=${event} workflow_run_id=${workflow_run_id}` +
      `${rest.completed_step_id ? ` completed=${rest.completed_step_id}` : ""}` +
      `${rest.next_step_id ? ` next=${rest.next_step_id}` : ""}` +
      `${rest.first_step_id ? ` first=${rest.first_step_id}` : ""}` +
      `${rest.step_id ? ` step=${rest.step_id}` : ""}`
    );
    const notifyCtx = workflowRunToContext.get(workflow_run_id);
    if (!notifyCtx?.channelId) return;
    const channel = await client.channels.fetch(notifyCtx.channelId).catch(() => null);
    if (!channel || typeof channel.send !== "function") return;

    let notifMsg = "";
    if (event === "workflow.started") {
      notifMsg = `🚀 工作流已启动（共 ${rest.step_count} 步），正在执行第一步：**${rest.first_step_id}**...`;
    } else if (event === "step.completed") {
      notifMsg = `✅ 步骤 **${rest.completed_step_id}** 完成。⏳ 进入下一步：**${rest.next_step_id}**...`;
    } else if (event === "step.approval_required") {
      notifMsg = `⏸️ 步骤 **${rest.step_id}** 需要人工审批后才能继续执行。`;
    }
    if (!notifMsg) return;
    const translated = await safeTranslate(notifMsg, notifyCtx.lang || "zh").catch(() => notifMsg);
    await channel.send(`[NEXUS] ${translated}`).catch(err => {
      console.warn(`[step_notify] send failed: ${err?.message}`);
    });
  }

  function createResultEmbed({ status, title, toolName, duration, summary }) {
    const embed = new EmbedBuilder()
      .setTitle(title)
      .setColor(status === "succeeded" ? 0x00ff00 : 0xff0000)
      .setDescription(`**Tool:** ${toolName}\n**Duration:** ${duration}s`)
      .setTimestamp();
    if (summary) {
      embed.addFields({ name: "Result", value: String(summary).slice(0, 1024) });
    }
    return embed;
  }

  function createBinaryAttachment(buffer, name = "artifact") {
    return new AttachmentBuilder(buffer, { name });
  }

  function registerHandlers({ onMessage, onReaction }) {
    if (typeof onMessage === "function") {
      client.on("messageCreate", async msg => {
        await onMessage(msg, { client, replyChunked, safeTranslate, taskToContext, runToContext, workflowRunToContext });
      });
    }
    if (typeof onReaction === "function") {
      client.on("messageReactionAdd", async (reaction, user) => {
        await onReaction(reaction, user, { client, replyChunked, safeTranslate, taskToContext, runToContext, workflowRunToContext });
      });
    }
  }

  async function login(token) {
    if (token && token !== "" && token !== "your_discord_token_here") {
      console.log("[discord] Attempting to login...");
      client.login(token).catch(err => {
        console.error(`[discord] Login failed: ${err.message}`);
      });
      return;
    }
    console.warn("[discord] No valid DISCORD_TOKEN found. Running in API-only mode.");
  }

  return {
    client,
    taskToContext,
    runToContext,
    workflowRunToContext,
    replyChunked,
    safeTranslate,
    bindTaskToContext,
    sendStepTransitionNotification,
    createResultEmbed,
    createBinaryAttachment,
    registerHandlers,
    login,
  };
}
