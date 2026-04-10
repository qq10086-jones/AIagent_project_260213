import { normalizeInputRequest } from "./input_normalizer.js";
import { routeTaskRequest as defaultRouteTaskRequest } from "./brain_router.js";
import {
  makeDirectReplyResponse,
  makeTaskQueuedResponse,
  makeWorkflowQueuedResponse,
  makeProjectPlanPendingResponse,
  makeErrorResponse,
} from "./response_protocol.js";
import {
  assertDispatchErrorResponse,
  assertDispatchSuccessResponse,
} from "./contract_validator.js";
import { validateSingleAgentPayloadGuardrails } from "../../../shared/contracts/single_agent_guardrails.js";
import { decompose } from "./project_planner.js";
import { createProjectExecutor } from "./project_executor.js";

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
  runtimeConfig = {},
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
        evidence_id: routed.task_envelope.evidence_id || "",
        replay_tag: routed.task_envelope.replay_tag || "",
      };
      const guardrails = validateSingleAgentPayloadGuardrails(payload);
      if (!guardrails.ok) {
        await updateRunStatus(pool, run_id, "failed").catch(() => {});
        return assertDispatchErrorResponse(
          makeErrorResponse({
            run_id,
            error: guardrails.errors.join("; "),
            error_code: "SINGLE_AGENT_GUARDRAILS_INVALID",
            task_envelope: routed.task_envelope,
          }),
        );
      }
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
          advisory: queued.advisory || null,
        }),
      );
    }

    if (routed.decision === "orchestrated_workflow") {
      await updateRunStatus(pool, run_id, "running").catch(() => {});

      // v3.3 Project Planner: complex + flag → planner 路径
      const plannerEnabled = Boolean(runtimeConfig?.worker_coder?.project_planner_enabled);
      const complexity = routed.route?.complexity || "medium";

      if (plannerEnabled && complexity === "complex") {
        try {
          const taskClasses = runtimeConfig?.worker_coder?.task_classes || undefined;
          const plannerConfig = {
            project_planner_max_runs: runtimeConfig?.worker_coder?.project_planner_max_runs ?? 12,
            project_planner_max_parallel_runs: runtimeConfig?.worker_coder?.project_planner_max_parallel_runs ?? 2,
            project_planner_failure_policy: runtimeConfig?.worker_coder?.project_planner_failure_policy ?? "stop_dependents",
          };

          const projectPlan = await decompose({
            raw_input: normalized.raw_input,
            project_type: plan.project_type || undefined,
            classifier_result: routed.route?.classifier_result,
            registry,
            task_classes: taskClasses,
            config: plannerConfig,
          });

          // Multi-run: 走 project executor
          if (!projectPlan._fallback && projectPlan.runs?.length > 1) {
            // P-C2: 人工确认模式
            const confirmMode = runtimeConfig?.worker_coder?.project_planner_confirm_mode ?? "manual";
            if (confirmMode === "manual") {
              await updateRunStatus(pool, run_id, "pending_confirmation").catch(() => {});
              return assertDispatchSuccessResponse(
                makeProjectPlanPendingResponse({
                  run_id,
                  task_envelope: routed.task_envelope,
                  project_plan: projectPlan,
                }),
              );
            }

            // confirm_mode=auto: 校验后执行
            const { validateProjectPlan } = await import("./project_plan_contract.js");
            const autoValidation = validateProjectPlan(projectPlan, {
              taskClasses: taskClasses,
              maxRuns: plannerConfig.project_planner_max_runs,
            });
            if (!autoValidation.ok) {
              console.warn("[vnext] project plan failed validation in auto mode, falling back:", autoValidation.errors);
              // 校验不通过 → 退化为单 run
            } else {

            const summary = await executeProjectPlanFromDispatch({
              projectPlan, workflowEngine, runtimeConfig, waterfallTraceService,
              run_id, requestBody, coderProviderDefault, coderModelDefault, routed,
            });

            await updateRunStatus(pool, run_id, summary.failed > 0 ? "partial_failure" : "completed").catch(() => {});

            return assertDispatchSuccessResponse(
              makeWorkflowQueuedResponse({
                run_id,
                task_envelope: routed.task_envelope,
                workflow_run_id: summary.project_id,
                workflow_id: "project_plan",
                first_step: `wave_0 (${summary.total_runs} runs)`,
              }),
            );
            } // end else (autoValidation.ok)
          }
          // Single-run fallback: 退化为普通 workflow
        } catch (plannerErr) {
          console.warn("[vnext] project planner failed, falling back to single workflow:", plannerErr?.message);
          // 继续走普通 workflow 路径
        }
      }

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

/**
 * 内部: 从 dispatch 上下文执行 project plan。
 */
async function executeProjectPlanFromDispatch({
  projectPlan, workflowEngine, runtimeConfig, waterfallTraceService,
  run_id, requestBody, coderProviderDefault, coderModelDefault, routed,
  resume = false,
}) {
  const executor = createProjectExecutor({
    workflowEngine,
    recordEvent: (id, event, data) => {
      if (waterfallTraceService) {
        waterfallTraceService.recordStage(run_id, event, new Date(), new Date(), data)
          .catch(() => {});
      }
    },
    runtimeConfig: runtimeConfig?.worker_coder || {},
    artifactDir: `runtime/artifacts/release/project-${run_id}`,
  });

  return executor.executeProjectPlan(projectPlan, {
    provider: requestBody?.provider || coderProviderDefault,
    model: requestBody?.model || coderModelDefault,
    task_envelope: routed?.task_envelope || null,
    fast_mode: Boolean(requestBody?.fast_mode),
  }, { resume });
}

/**
 * P-C2: 人工确认后执行 project plan。
 * 用户审阅 plan 后调用此函数提交执行。
 */
export function createConfirmProjectPlan({
  pool,
  updateRunStatus,
  getRunById,
  workflowEngine,
  runtimeConfig = {},
  waterfallTraceService = null,
  coderProviderDefault = "opencode",
  coderModelDefault = "minimax-coding-plan/MiniMax-M2.7",
}) {
  return async function confirmProjectPlan({ run_id, project_plan, requestBody = {}, resume = false }) {
    // Fix #3: 状态绑定 — 只有 pending_confirmation 状态的 run 才能被 confirm
    if (typeof getRunById === "function") {
      const run = await getRunById(pool, run_id);
      if (!run) {
        return { ok: false, error: `run ${run_id} not found`, error_code: "RUN_NOT_FOUND" };
      }
      const allowedStates = resume ? ["partial_failure", "pending_confirmation"] : ["pending_confirmation"];
      if (!allowedStates.includes(run.status)) {
        return {
          ok: false,
          error: `run ${run_id} is in state '${run.status}', expected one of: ${allowedStates.join(", ")}`,
          error_code: "INVALID_RUN_STATE",
        };
      }
    }

    const { validateProjectPlan } = await import("./project_plan_contract.js");
    const taskClasses = runtimeConfig?.worker_coder?.task_classes || undefined;
    const validation = validateProjectPlan(project_plan, {
      taskClasses,
      maxRuns: runtimeConfig?.worker_coder?.project_planner_max_runs ?? 12,
    });
    if (!validation.ok) {
      return {
        ok: false,
        error: `Plan validation failed: ${validation.errors.join("; ")}`,
        error_code: "PROJECT_PLAN_INVALID",
      };
    }

    await updateRunStatus(pool, run_id, "running").catch(() => {});

    const summary = await executeProjectPlanFromDispatch({
      projectPlan: project_plan, workflowEngine, runtimeConfig, waterfallTraceService,
      run_id, requestBody, coderProviderDefault, coderModelDefault, routed: null,
      resume,
    });

    await updateRunStatus(pool, run_id, summary.failed > 0 ? "partial_failure" : "completed").catch(() => {});

    return {
      ok: true,
      project_summary: summary,
    };
  };
}
