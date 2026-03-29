import { normalizeInputRequest } from "./input_normalizer.js";
import { routeTaskRequest as defaultRouteTaskRequest } from "./brain_router.js";
import {
  makeDirectReplyResponse,
  makeTaskQueuedResponse,
  makeWorkflowQueuedResponse,
  makeErrorResponse,
} from "./response_protocol.js";
import {
  assertDispatchErrorResponse,
  assertDispatchSuccessResponse,
} from "./contract_validator.js";

export function createExecuteVNextDispatch({
  ensureRun,
  parseIntent,
  registry,
  generateBrainDirectReply,
  pool,
  updateRunStatus,
  enqueueTask,
  workflowEngine,
  routeTaskRequest = defaultRouteTaskRequest,
  coderProviderDefault = "opencode",
  coderModelDefault = "minimax-coding-plan/MiniMax-M2.7",
  waterfallTraceService = null,
}) {
  if (typeof ensureRun !== "function") throw new Error("ensureRun is required");
  if (typeof parseIntent !== "function") throw new Error("parseIntent is required");
  if (!registry) throw new Error("registry is required");
  if (typeof generateBrainDirectReply !== "function") throw new Error("generateBrainDirectReply is required");
  if (!pool?.query || typeof pool.query !== "function") throw new Error("pool.query is required");
  if (typeof updateRunStatus !== "function") throw new Error("updateRunStatus is required");
  if (typeof enqueueTask !== "function") throw new Error("enqueueTask is required");
  if (!workflowEngine?.startWorkflowRun || typeof workflowEngine.startWorkflowRun !== "function") {
    throw new Error("workflowEngine.startWorkflowRun is required");
  }

  return async function executeVNextDispatch({
    requestBody = {},
    run_id,
    client_msg_id = "",
    skipEnsureRun = false,
    analyzerResult = undefined,
    routeOverride = null,
    context = null,
  }) {
    // WS-30-02: intake stage start
    const intakeStart = new Date();

    const normalized = normalizeInputRequest(requestBody || {});
    if (!normalized.raw_input) {
      const err = new Error("raw_input/message is required");
      err.code = "BAD_REQUEST";
      throw err;
    }

    if (!skipEnsureRun) {
      await ensureRun(run_id, {
        client_msg_id: client_msg_id || `vnext-${run_id}`,
        user_id: normalized.context?.user_id || "vnext-user",
        status: "starting",
        input_text: normalized.raw_input,
      });
    }

    let finalAnalyzerResult = analyzerResult;
    if (finalAnalyzerResult === undefined) {
      try {
        finalAnalyzerResult = await parseIntent(normalized.raw_input, {});
      } catch (err) {
        console.warn("[vnext] parseIntent failed during dispatch:", err?.message || err);
        finalAnalyzerResult = null;
      }
    }

    // WS-30-02: routing stage — time the brain router classification
    const routingStart = new Date();
    const routed = routeOverride || routeTaskRequest({
      ...normalized,
      analyzerResult: finalAnalyzerResult,
      registry,
    });
    const routingEnd = new Date();

    // Persist intake and routing stages (fire-and-forget)
    if (waterfallTraceService) {
      waterfallTraceService.recordStage(run_id, "intake", intakeStart, routingStart)
        .catch((e) => console.warn("[waterfall_trace] intake:", e.message));
      waterfallTraceService.recordStage(run_id, "routing", routingStart, routingEnd)
        .catch((e) => console.warn("[waterfall_trace] routing:", e.message));
    }
    const plan = routed.task_envelope.execution_plan || {};

    if (routed.decision === "direct_reply") {
      const reply = await generateBrainDirectReply(
        normalized.raw_input,
        String(requestBody?.model_preference || "auto"),
      );
      await updateRunStatus(pool, run_id, "completed").catch(() => {});
      return assertDispatchSuccessResponse(
        makeDirectReplyResponse({
          run_id,
          task_envelope: routed.task_envelope,
          reply,
        }),
      );
    }

    if (routed.decision === "single_agent") {
      await updateRunStatus(pool, run_id, "running").catch(() => {});
      const payload = {
        ...(finalAnalyzerResult?.payload || {}),
        ...(requestBody?.payload && typeof requestBody.payload === "object" ? requestBody.payload : {}),
        task_envelope: routed.task_envelope,
        task_prompt: requestBody?.task_prompt || normalized.raw_input,
        prompt: requestBody?.prompt || normalized.raw_input,
        project_type: plan.project_type || undefined,
      };
      const queued = await enqueueTask({
        tool_name: String(plan.tool_name || "coding.delegate"),
        payload,
        run_id,
        context,
      });
      return assertDispatchSuccessResponse(
        makeTaskQueuedResponse({
          run_id,
          task_envelope: routed.task_envelope,
          task_id: queued.task_id,
          tool_name: String(plan.tool_name || "coding.delegate"),
          waiting_approval: Boolean(queued.waiting_approval),
        }),
      );
    }

    if (routed.decision === "orchestrated_workflow") {
      await updateRunStatus(pool, run_id, "running").catch(() => {});
      // WS-30-02: execution_dispatch stage
      const dispatchStart = new Date();
      const started = await workflowEngine.startWorkflowRun({
        workflow_id: String(plan.workflow_id || "coding_team_v0"),
        project_type: String(plan.project_type || "webapp_crm"),
        run_id,
        input: {
          goal: normalized.raw_input,
          provider: requestBody?.provider || coderProviderDefault,
          model: requestBody?.model || coderModelDefault,
          task_envelope: routed.task_envelope,
          fast_mode: Boolean(requestBody?.fast_mode),
        },
        context,
      });
      if (waterfallTraceService) {
        waterfallTraceService.recordStage(run_id, "execution_dispatch", dispatchStart, new Date(), {
          workflow_run_id: started.workflow_run_id,
        }).catch((e) => console.warn("[waterfall_trace] execution_dispatch:", e.message));
      }
      return assertDispatchSuccessResponse(
        makeWorkflowQueuedResponse({
          run_id,
          task_envelope: routed.task_envelope,
          workflow_run_id: started.workflow_run_id,
          workflow_id: started.workflow_id,
          first_step: started.first_step,
        }),
      );
    }

    await updateRunStatus(pool, run_id, "failed").catch(() => {});
    return assertDispatchErrorResponse(
      makeErrorResponse({
        run_id,
        error: "human review required",
        error_code: "UNKNOWN_ERROR",
        task_envelope: routed.task_envelope,
      }),
    );
  };
}
