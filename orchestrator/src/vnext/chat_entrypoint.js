import { makeErrorResponse } from "./response_protocol.js";
import { assertDispatchErrorResponse } from "./contract_validator.js";
import { callLocalOllamaChat, callQwenChat } from "./local_llm_client.js";

function parseBoolEnv(value, defaultValue = false) {
  const raw = String(value ?? "").trim().toLowerCase();
  if (!raw) return defaultValue;
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  return defaultValue;
}

export function shouldUseLegacyChatToolPath(intent) {
  const toolName = String(intent?.tool_name || "").trim();
  if (!toolName) return false;
  return !toolName.startsWith("coding.");
}

export function buildApiDispatchJson(result) {
  if (!result?.ok) {
    return {
      ok: false,
      mode: "error",
      run_id: String(result?.run_id || ""),
      error: String(result?.error || "unknown error"),
      error_code: String(result?.error_code || "UNKNOWN_ERROR"),
      task_envelope: result?.task_envelope || null,
    };
  }

  if (result.response_mode === "direct_reply") {
    return {
      ok: true,
      mode: "direct_reply",
      run_id: result.run_id,
      reply: result.reply || "",
      task_envelope: result.task_envelope || null,
    };
  }

  if (result.execution?.workflow_run_id) {
    return {
      ok: true,
      mode: "workflow",
      run_id: result.run_id,
      workflow_run_id: result.execution.workflow_run_id,
      workflow_id: result.execution.workflow_id || "",
      first_step: result.execution.first_step || null,
      waiting_approval: Boolean(result.execution.first_step?.waiting_approval),
      task_envelope: result.task_envelope || null,
    };
  }

  if (result.execution?.task_id) {
    return {
      ok: true,
      mode: "task",
      run_id: result.run_id,
      task_id: result.execution.task_id,
      tool_name: result.execution.tool_name || "",
      waiting_approval: Boolean(result.execution.waiting_approval),
      task_envelope: result.task_envelope || null,
    };
  }

  return {
    ok: true,
    mode: "unknown",
    run_id: result.run_id,
    task_envelope: result.task_envelope || null,
  };
}

export async function generateBrainDirectReply({
  rawInput,
  modelPreference = "auto",
  forceLocalLlm = false,
  hasQwenKey = false,
  qwenBase = "",
  qwenModel = "",
  currentLocalModel = "",
  projectContext = "",
}) {
  const disableLocalFallback = parseBoolEnv(process.env.DISABLE_LOCAL_LLM_FALLBACK, false);
  const useCloudChat = modelPreference === "api" || (!forceLocalLlm && hasQwenKey);
  if (useCloudChat) {
    try {
      const reply = await callQwenChat([{ role: "user", content: rawInput }], { qwenBase, qwenModel });
      if (String(reply || "").trim()) return reply;
    } catch (err) {
      if (disableLocalFallback) {
        console.warn("[vnext] cloud direct reply failed, local fallback disabled:", err?.message || err);
        throw err;
      }
      console.warn("[vnext] cloud direct reply failed, fallback to local:", err?.message || err);
    }
  }
  return await callLocalOllamaChat(currentLocalModel, rawInput, undefined, projectContext) || "";
}

export function createHandleApiChat({
  uuidv4,
  ensureRun,
  planCompositeWorkflowFromText,
  pool,
  updateRunStatus,
  completeRunWithCostLedger,
  enqueueWorkflow,
  parseIntent,
  buildForcedIntentFromRule,
  executeVNextDispatch,
  buildVNextDispatchInput,
  forceLocalLlm = false,
  callBrainWithRetry,
  currentLocalModel = "",
  currentQwenModel = "",
}) {
  if (typeof uuidv4 !== "function") throw new Error("uuidv4 is required");
  if (typeof ensureRun !== "function") throw new Error("ensureRun is required");
  if (typeof planCompositeWorkflowFromText !== "function") throw new Error("planCompositeWorkflowFromText is required");
  if (!pool?.query || typeof pool.query !== "function") throw new Error("pool.query is required");
  if (typeof updateRunStatus !== "function") throw new Error("updateRunStatus is required");
  if (typeof completeRunWithCostLedger !== "function") throw new Error("completeRunWithCostLedger is required");
  if (typeof enqueueWorkflow !== "function") throw new Error("enqueueWorkflow is required");
  if (typeof parseIntent !== "function") throw new Error("parseIntent is required");
  if (typeof buildForcedIntentFromRule !== "function") throw new Error("buildForcedIntentFromRule is required");
  if (typeof executeVNextDispatch !== "function") throw new Error("executeVNextDispatch is required");
  if (typeof buildVNextDispatchInput !== "function") throw new Error("buildVNextDispatchInput is required");
  if (typeof callBrainWithRetry !== "function") throw new Error("callBrainWithRetry is required");

  return async function handleApiChat(req, res) {
    const { message } = req.body || {};
    const run_id = uuidv4();

    try {
      await ensureRun(run_id, {
        client_msg_id: `api-${run_id}`,
        user_id: "api-user",
        status: "starting",
        input_text: message || "",
      });
    } catch (e) {
      return res.status(500).json({ ok: false, error: `Failed to initialize run: ${e.message}` });
    }

    const compositePlan = await planCompositeWorkflowFromText(message || "", {});

    if (compositePlan) {
      try {
        await updateRunStatus(pool, run_id, "running");
        const wf = await enqueueWorkflow({
          name: compositePlan.name,
          steps: compositePlan.steps,
          run_id,
          context: null,
        });
        if (!wf?.ok) throw new Error(wf?.error || "workflow enqueue failed");
        return res.json({
          ok: true,
          mode: "workflow",
          workflow_id: wf.workflow_id,
          run_id,
          tasks: wf.tasks,
        });
      } catch (e) {
        await updateRunStatus(pool, run_id, "failed").catch(() => {});
        return res.status(500).json({ ok: false, error: e.message });
      }
    }

    let intent = await parseIntent(message || "");
    const forcedIntent = buildForcedIntentFromRule(message || "");
    if (
      forcedIntent &&
      (intent.mode_suggested === "chat" || !intent.requires_tools || intent.confidence < 0.6 || !intent.tool_name)
    ) {
      intent = forcedIntent;
    } else if (
      forcedIntent &&
      intent?.tool_name &&
      forcedIntent.tool_name === intent.tool_name
    ) {
      intent.payload = { ...(intent.payload || {}), ...(forcedIntent.payload || {}) };
    }

    if (!shouldUseLegacyChatToolPath(intent)) {
      try {
        const result = await executeVNextDispatch({
          requestBody: buildVNextDispatchInput({
            source: "api",
            rawInput: message || "",
            payload: {
              payload: req.body?.payload && typeof req.body.payload === "object" ? req.body.payload : undefined,
              provider: req.body?.provider || undefined,
              model: req.body?.model || undefined,
              fast_mode: Boolean(req.body?.fast_mode),
              model_preference: forceLocalLlm ? "local" : "auto",
            },
          }),
          run_id,
          client_msg_id: `api-${run_id}`,
          skipEnsureRun: true,
          analyzerResult: intent,
        });
        if (!result?.ok) {
          const status = String(result?.error_code || "") === "BAD_REQUEST" ? 400 : 500;
          return res.status(status).json(buildApiDispatchJson(result));
        }
        return res.json(buildApiDispatchJson(result));
      } catch (e) {
        await updateRunStatus(pool, run_id, "failed").catch(() => {});
        return res.status(500).json({ ok: false, error: e.message || "vnext dispatch failed", run_id });
      }
    }

    if (intent.confidence > 0.6 && intent.tool_name) {
      try {
        const immediateOpsQuery = /(本金|资金|空仓|仓位|怎(?:么|樣)操作|如何操作|明天怎么操作|明日どうする)/i.test(String(message || ""));
        if (intent.tool_name === "quant.discovery_workflow" && immediateOpsQuery) {
          intent.payload = {
            ...(intent.payload || {}),
            quick_mode: true,
            time_budget_s: 75,
            max_attempts: Math.min(Number(intent.payload?.max_attempts || 2), 2),
            min_candidates: Math.min(Number(intent.payload?.min_candidates || 2), 2),
          };
        }

        await updateRunStatus(pool, run_id, "running");
        const mode = intent.tool_name === "quant.discovery_workflow" ? "discovery" : "analysis";
        const toolPayload = { ...(intent.payload || {}) };
        if (mode === "discovery" && /(本金|资金|空仓|仓位|怎(?:么|樣)操作|如何操作|明天怎么操作|明日どうする)/i.test(String(message || ""))) {
          toolPayload.quick_mode = true;
          toolPayload.time_budget_s = 75;
          toolPayload.max_attempts = Math.min(Number(toolPayload.max_attempts || 2), 2);
          toolPayload.min_candidates = Math.min(Number(toolPayload.min_candidates || 2), 2);
        }
        const brainRetries = mode === "discovery" ? 0 : 2;
        const brainData = await callBrainWithRetry({
          symbol: intent.payload.symbol || "unknown",
          run_id,
          mode,
          tool_name: intent.tool_name,
          tool_payload: toolPayload,
          model_preference: forceLocalLlm ? "local_large" : "local_small",
          local_model: currentLocalModel,
          qwen_model: currentQwenModel,
        }, brainRetries);
        await completeRunWithCostLedger(pool, run_id, brainData?.cost_ledger || {});
        return res.json({
          ok: true,
          narrative: brainData.narrative,
          report_markdown: brainData.report_markdown || "",
          report_html_object_key: brainData.report_html_object_key || "",
          run_id,
        });
      } catch (e) {
        await updateRunStatus(pool, run_id, "failed").catch(() => {});
        return res.status(500).json({ ok: false, error: e.message });
      }
    }

    await updateRunStatus(pool, run_id, "completed").catch(() => {});
    return res.json(assertDispatchErrorResponse(makeErrorResponse({
      run_id,
      error: "no actionable route",
      error_code: "NO_ACTIONABLE_ROUTE",
      task_envelope: null,
    })));
  };
}
