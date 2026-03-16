/**
 * discord_message_handler.js
 *
 * Discord message and reaction handlers.
 * Factory injects all runtime dependencies so this module is testable in isolation.
 * Extracted from index.js as part of WS-11-05 (Discord adapter finalisation / WS-11-02).
 */

import { v4 as uuidv4 } from "uuid";
import { parseCoderDirectiveOptions } from "../vnext/coder_directive.js";
import { detectLanguageQuick, buildVNextDispatchInput } from "../vnext/composite_planner.js";

/**
 * @param {{ redis, discord, approvalToken, coderProviderDefault, coderModelDefault,
 *           ensureRun, updateRunStatus, findRunIdByClientMsgId, enqueueTask,
 *           makeIdempotencyKey, getToolSpec, executeVNextDispatch,
 *           appState, currentQwenModel, setQwenModel,
 *           translate, safeTranslate, replyChunked,
 *           runToContext, workflowRunToContext }} deps
 */
export function createDiscordMessageHandler({
  redis,
  discord,
  approvalToken,
  coderProviderDefault,
  coderModelDefault,
  ensureRun,
  updateRunStatus,
  findRunIdByClientMsgId,
  enqueueTask,
  makeIdempotencyKey,
  getToolSpec,
  executeVNextDispatch,
  appState,
  currentQwenModel,
  setQwenModel,
  translate,
  safeTranslate,
  replyChunked,
  runToContext,
  workflowRunToContext,
}) {
  const extractCommandArg = (input, command) => {
    const m = String(input || "").match(new RegExp(`^${command}(?::|\\uff1a|\\s+)(.+)$`, "i"));
    return m?.[1]?.trim() || "";
  };

  const parseRunDirective = (input) => {
    const arg = extractCommandArg(input, "\\/run");
    if (!arg) return { tool_name: "", payload: null, error: "missing_arg" };
    const firstSpace = arg.indexOf(" ");
    const tool_name = (firstSpace >= 0 ? arg.slice(0, firstSpace) : arg).trim();
    const payloadPart = (firstSpace >= 0 ? arg.slice(firstSpace + 1) : "").trim();
    if (!tool_name) return { tool_name: "", payload: null, error: "missing_tool" };
    if (!payloadPart) return { tool_name, payload: {}, error: "" };
    try {
      return { tool_name, payload: JSON.parse(payloadPart), error: "" };
    } catch {
      return { tool_name, payload: null, error: "invalid_payload_json" };
    }
  };

  async function handleDiscordMessage(msg) {
    if (msg.author.bot) return;
    try {
      const lockOk = await redis.set(`discord:msg:${msg.id}:handled`, "1", "EX", 180, "NX");
      if (!lockOk) return;
    } catch (e) {
      console.warn("[discord] dedupe lock failed, continue without lock:", e?.message || e);
    }

    const rawInput = msg.content || "";
    const trimmedInput = rawInput.trim();

    if (trimmedInput === "/model-local" || trimmedInput.startsWith("/model-local:") || trimmedInput.startsWith("/model-local ")) {
      const requestedModel = extractCommandArg(trimmedInput, "\\/model-local");
      if (requestedModel) appState.currentLocalModel = requestedModel.replace(/^ollama\//i, "").trim();
      appState.forceLocalLlm = true;
      await msg.reply(`[NEXUS] \u5df2\u5207\u6362\u5230\u672c\u5730\u6a21\u578b\u6a21\u5f0f\u3002\u5f53\u524d\u672c\u5730\u6a21\u578b: **${appState.currentLocalModel}**`);
      return;
    }

    if (trimmedInput === "/model-cloud") {
      appState.forceLocalLlm = false;
      await msg.reply(`[NEXUS] \u5df2\u5207\u56de\u4e91\u7aef\u6a21\u578b\u6a21\u5f0f\u3002\u5f53\u524d\u4e91\u7aef\u6a21\u578b: **${currentQwenModel()}**`);
      return;
    }

    if (trimmedInput === "/model" || trimmedInput.startsWith("/model:") || trimmedInput.startsWith("/model ")) {
      const newModel = extractCommandArg(trimmedInput, "\\/model");
      if (newModel) {
        setQwenModel(newModel);
        await msg.reply(`[NEXUS] \u6a21\u578b\u5df2\u6210\u529f\u5207\u6362\u4e3a: **${newModel}**`);
      } else {
        await msg.reply(`[NEXUS] \u5f53\u524d\u6a21\u578b\u662f: **${currentQwenModel()}**`);
      }
      return;
    }

    if (trimmedInput === "/approve" || trimmedInput.startsWith("/approve:") || trimmedInput.startsWith("/approve\uff1a") || trimmedInput.startsWith("/approve ")) {
      const taskId = extractCommandArg(trimmedInput, "\\/approve");
      if (!taskId) { await msg.reply("[NEXUS] \u7528\u6cd5\uff1a`/approve: <task_id>`"); return; }
      try {
        const resp = await fetch(`http://localhost:3000/tasks/${encodeURIComponent(taskId)}/approve`, {
          method: "POST", headers: { "X-Approval-Token": approvalToken },
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data.ok) await msg.reply(`[NEXUS] \u5ba1\u6279\u5931\u8d25\uff1a${data.error || `HTTP ${resp.status}`}`);
        else await msg.reply(`[NEXUS] \u5df2\u6279\u51c6\u4efb\u52a1\uff1a${taskId}`);
      } catch (e) { await msg.reply(`[NEXUS] \u5ba1\u6279\u5f02\u5e38\uff1a${e.message}`); }
      return;
    }

    if (trimmedInput === "/reject" || trimmedInput.startsWith("/reject:") || trimmedInput.startsWith("/reject\uff1a") || trimmedInput.startsWith("/reject ")) {
      const taskId = extractCommandArg(trimmedInput, "\\/reject");
      if (!taskId) { await msg.reply("[NEXUS] \u7528\u6cd5\uff1a`/reject: <task_id>`"); return; }
      try {
        const resp = await fetch(`http://localhost:3000/tasks/${encodeURIComponent(taskId)}/reject`, {
          method: "POST", headers: { "X-Approval-Token": approvalToken },
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok || !data.ok) await msg.reply(`[NEXUS] \u62d2\u7edd\u5931\u8d25\uff1a${data.error || `HTTP ${resp.status}`}`);
        else await msg.reply(`[NEXUS] \u5df2\u62d2\u7edd\u4efb\u52a1\uff1a${taskId}`);
      } catch (e) { await msg.reply(`[NEXUS] \u62d2\u7edd\u5f02\u5e38\uff1a${e.message}`); }
      return;
    }

    if (trimmedInput === "/run" || trimmedInput.startsWith("/run:") || trimmedInput.startsWith("/run\uff1a") || trimmedInput.startsWith("/run ")) {
      const parsedRun = parseRunDirective(trimmedInput);
      if (parsedRun.error === "missing_arg" || parsedRun.error === "missing_tool") {
        await msg.reply("[NEXUS] \u7528\u6cd5\uff1a`/run <tool_name> [json_payload]`\uff0c\u4f8b\u5982\uff1a`/run news.tdnet_close_flash {\"date\":\"2026-03-02\"}`");
        return;
      }
      if (parsedRun.error === "invalid_payload_json") {
        await msg.reply("[NEXUS] payload \u5fc5\u987b\u662f\u5408\u6cd5 JSON\u3002\u793a\u4f8b\uff1a`/run news.daily_report {\"lookback_hours\":24}`");
        return;
      }
      if (!getToolSpec(parsedRun.tool_name)?.kind) {
        await msg.reply(`[NEXUS] \u672a\u77e5\u5de5\u5177\uff1a\`${parsedRun.tool_name}\`\u3002\u8bf7\u68c0\u67e5 configs/tools.json\u3002`);
        return;
      }
      const run_id = uuidv4();
      const runLang = detectLanguageQuick(trimmedInput);
      const context = { channelId: msg.channel.id, startTime: Date.now(), lang: runLang || "zh", run_id, closeRunOnTaskResult: true };
      runToContext.set(run_id, context);
      try {
        await ensureRun(run_id, { client_msg_id: msg.id, user_id: msg.author.id, status: "running", input_text: trimmedInput.slice(0, 2000) });
        const progressText = await translate(`\u5df2\u6536\u5230 /run\uff0c\u5f00\u59cb\u6267\u884c\u5de5\u5177 ${parsedRun.tool_name} ...`, context.lang || "zh");
        await msg.reply(`[NEXUS] ${progressText}`);
        const queued = await enqueueTask({
          tool_name: parsedRun.tool_name, payload: parsedRun.payload || {}, run_id,
          idempotency_key: makeIdempotencyKey(run_id, parsedRun.tool_name, parsedRun.payload || {}), context,
        });
        if (queued?.waiting_approval) {
          await msg.reply(`[NEXUS] \u4efb\u52a1\u7b49\u5f85\u5ba1\u6279\uff1atask_id=${queued.task_id}\n\u6279\u51c6\uff1a\`/approve: ${queued.task_id}\`\n\u62d2\u7edd\uff1a\`/reject: ${queued.task_id}\``);
        }
      } catch (err) {
        runToContext.delete(run_id);
        await updateRunStatus(run_id, "failed").catch(() => {});
        await msg.reply(`[NEXUS] /run \u6267\u884c\u5931\u8d25\uff1a${err.message}`);
      }
      return;
    }

    const isCoderDirective = trimmedInput === "/coder" || trimmedInput.startsWith("/coder:") || trimmedInput.startsWith("/coder\uff1a") || trimmedInput.startsWith("/coder ");
    const coderTask = isCoderDirective ? extractCommandArg(trimmedInput, "\\/coder") : "";
    if (isCoderDirective && !coderTask) {
      await msg.reply("[NEXUS] \u7528\u6cd5\uff1a`/coder: <\u5f00\u53d1\u4efb\u52a1>`\uff0c\u53ef\u9009\uff1a`@model=qwen-coder-next` / `@provider=opencode`");
      return;
    }

    const coderOptions = isCoderDirective ? parseCoderDirectiveOptions(coderTask, { provider: coderProviderDefault, model: coderModelDefault }) : null;
    const effectiveInput = isCoderDirective ? coderOptions.taskPrompt : rawInput;
    const userInput = effectiveInput.trim();
    if (!userInput) return;

    const client_msg_id = msg.id;
    const run_id = uuidv4();
    const lang = detectLanguageQuick(userInput) || "zh";
    const context = { channelId: msg.channel.id, startTime: Date.now(), lang, run_id, closeRunOnTaskResult: true };

    try {
      const existingRunId = await findRunIdByClientMsgId(client_msg_id);
      if (existingRunId) return;

      await msg.channel.sendTyping();
      await ensureRun(run_id, { client_msg_id, user_id: msg.author.id, status: "starting", input_text: userInput });

      const routeOverride = isCoderDirective ? {
        decision: "orchestrated_workflow",
        route: {
          intent: "coding", complexity: "complex", target_team: "coding_team",
          requires_orchestration: true,
          expected_outputs: ["design_doc", "task_breakdown", "repo_changes", "tests", "qa_summary"],
          execution_plan: { mode: "orchestrated_workflow", workflow_id: "coding_team_v0", project_type: "webapp_crm" },
        },
        task_envelope: {
          task_id: uuidv4(), source: "discord", raw_input: userInput,
          normalized_input: { text: userInput, attachments: [], metadata: { user_id: String(msg.author?.id || ""), username: String(msg.author?.username || ""), channel_id: String(msg.channel?.id || ""), thread_id: String(msg.id || "") } },
          intent: "coding", sub_intent: "coder_directive", requires_orchestration: true, target_team: "coding_team",
          expected_outputs: ["design_doc", "task_breakdown", "repo_changes", "tests", "qa_summary"],
          constraints: { local_only: true, approval_mode: "manual", risk_level: "medium", complexity: "complex" },
          context: { channel_id: String(msg.channel?.id || ""), thread_id: String(msg.id || ""), user_id: String(msg.author?.id || ""), attachments: [], raw_event: null },
          decision: "orchestrated_workflow",
          execution_plan: { mode: "orchestrated_workflow", workflow_id: "coding_team_v0", project_type: "webapp_crm" },
        },
      } : null;

      if (isCoderDirective) {
        const progressText = await safeTranslate(
          `\u5df2\u8fdb\u5165 Coding Team \u6a21\u5f0f\uff0c\u6b63\u5728\u542f\u52a8 PM/\u67b6\u6784/\u524d\u540e\u7aef/QA \u6d41\u7a0b\u3002provider=${coderOptions?.provider || coderProviderDefault}\uff0cmodel=${coderOptions?.model || "qwen-coder-next"}`, lang
        );
        await msg.reply(`[NEXUS] ${progressText}`);
      }

      const result = await executeVNextDispatch({
        requestBody: buildVNextDispatchInput({
          rawInput: userInput, msg,
          payload: {
            provider: coderOptions?.provider || coderProviderDefault,
            model: coderOptions?.model || undefined,
            fast_mode: isCoderDirective,
            model_preference: appState.forceLocalLlm ? "local" : "auto",
          },
        }),
        run_id, client_msg_id, skipEnsureRun: true, routeOverride, context,
      });

      if (!result.ok) { await replyChunked(msg, `[NEXUS] ${result.error || "\u4efb\u52a1\u9700\u8981\u4eba\u5de5\u786e\u8ba4\u3002"}`); return; }
      if (result.response_mode === "direct_reply") { await replyChunked(msg, result.reply || "I didn't understand that."); return; }

      if (result.execution?.workflow_run_id) {
        const initialMsg = await msg.reply(`[NEXUS] 🛠️ **项目制作中**\n⏳ [⏳] (排队中...) \`Initializing\`\n💬 Waiting for workflow to start...`);
        workflowRunToContext.set(result.execution.workflow_run_id, { 
          channelId: context.channelId, 
          lang, 
          progressMessageId: initialMsg.id 
        });
        if (result.execution?.first_step?.waiting_approval && result.execution?.first_step?.task_id) {
          await msg.reply(`[NEXUS] \u4efb\u52a1\u7b49\u5f85\u5ba1\u6279\uff1atask_id=${result.execution.first_step.task_id}\n\u6279\u51c6\uff1a\`/approve: ${result.execution.first_step.task_id}\`\n\u62d2\u7edd\uff1a\`/reject: ${result.execution.first_step.task_id}\``);
        }
        return;
      }

      if (result.execution?.task_id) {
        const queuedText = await safeTranslate(`\u4efb\u52a1\u5df2\u8fdb\u5165\u6267\u884c\u961f\u5217\u3002run_id=${run_id}\uff0ctask_id=${result.execution.task_id}\uff0ctool=${result.execution.tool_name || "unknown"}\u3002`, lang);
        await msg.reply(`[NEXUS] ${queuedText}`);
        if (result.execution.waiting_approval) {
          await msg.reply(`[NEXUS] \u4efb\u52a1\u7b49\u5f85\u5ba1\u6279\uff1atask_id=${result.execution.task_id}\n\u6279\u51c6\uff1a\`/approve: ${result.execution.task_id}\`\n\u62d2\u7edd\uff1a\`/reject: ${result.execution.task_id}\``);
        }
        return;
      }
    } catch (e) {
      console.error("[orchestrator] Error:", e);
      await updateRunStatus(run_id, "failed").catch(() => {});
      runToContext.delete(run_id);
      await replyChunked(msg, `Error: ${e.message}`);
    }
  }

  async function handleDiscordReaction(reaction, user) {
    if (user.bot) return;
    if (reaction.message.author && reaction.message.author.id !== discord.user.id) return;
    if (reaction.partial) {
      try { await reaction.fetch(); } catch (err) { console.warn("[discord] Failed to fetch partial reaction:", err); return; }
    }

    const emoji = reaction.emoji.name;
    let feedback = null, rating = null;
    if (emoji === "\ud83d\udcaf") { feedback = "\u2705"; rating = 5; }
    else if (emoji === "\ud83d\udc4d") { feedback = "\u2705"; rating = 4; }
    else if (emoji === "\ud83d\udc4e") { feedback = "\u274c"; rating = 1; }

    if (feedback) {
      const trace_id = reaction.message.id;
      try {
        console.log(`[learning] User ${user.tag} reacted ${emoji} to msg ${trace_id}, applying feedback: ${feedback}`);
        await fetch(`http://localhost:3000/traces/${trace_id}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feedback, rating, reason: feedback === "\u274c" ? "User clicked thumbsdown on Discord." : "User upvoted on Discord." }),
        });
        await reaction.message.react("\ud83e\udd16");
      } catch (err) { console.warn(`[learning] Error applying feedback from reaction: ${err.message}`); }
    }
  }

  return { handleDiscordMessage, handleDiscordReaction };
}
