/**
 * task_enqueuer.js
 *
 * enqueueTask and enqueueWorkflow — task lifecycle entry points.
 * Validates against registry, applies risk analysis, deduplicates via idempotency key,
 * and dispatches to the appropriate Redis stream.
 * Extracted from index.js as part of WS-11-05 decomposition.
 */

import { v4 as uuidv4 } from "uuid";

/**
 * @param {{ pool, redis, registry, analyzeTaskRisk, validateTaskInputAgainstRegistry,
 *           getToolSpec, upsertTask, getTaskStream, findTaskIdByIdempotencyKey,
 *           enqueueToStream, bindTaskToContext, recordEvent, insertWorkflowDefinition,
 *           makeIdempotencyKey, groupTask, evaluatePermissionAdvisory, insertPermissionAuditRecord }} deps
 */
export function createTaskEnqueuer({
  pool,
  redis,
  registry,
  analyzeTaskRisk,
  validateTaskInputAgainstRegistry,
  getToolSpec,
  upsertTask,
  getTaskStream,
  findTaskIdByIdempotencyKey,
  enqueueToStream,
  bindTaskToContext,
  recordEvent,
  insertWorkflowDefinition,
  makeIdempotencyKey,
  groupTask,
  evaluatePermissionAdvisory = null,
  insertPermissionAuditRecord = null,
}) {
  async function enqueueTask({ tool_name, payload, run_id, risk_level = null, idempotency_key, context }) {
    const fullPayload = { ...(payload || {}), run_id };
    const idem = idempotency_key || makeIdempotencyKey(run_id, tool_name, fullPayload);
    const spec = getToolSpec(tool_name);

    const regCheck = validateTaskInputAgainstRegistry({ registry, tool_name, payload: fullPayload });
    if (!regCheck.ok) {
      const err = new Error(`REGISTRY_INVALID: ${regCheck.errors.join("; ")}`);
      err.code = "REGISTRY_INVALID";
      throw err;
    }

    const risk = analyzeTaskRisk(tool_name, fullPayload);
    const finalRisk = risk_level || risk.risk_level || spec?.default_risk || "low";
    const requiresApproval = Boolean(risk.requires_approval);
    const advisory = typeof evaluatePermissionAdvisory === "function"
      ? evaluatePermissionAdvisory({ tool_name, payload: fullPayload, risk: { ...risk, risk_level: finalRisk } })
      : null;

    const existingTaskId = await findTaskIdByIdempotencyKey(pool, idem);
    if (existingTaskId) {
      bindTaskToContext(existingTaskId, context, tool_name);
      return { task_id: existingTaskId, deduplicated: true, advisory };
    }

    const task_id = uuidv4();
    await upsertTask({
      task_id,
      tool_name,
      status: requiresApproval ? "waiting_approval" : "queued",
      risk_level: finalRisk,
      payload: fullPayload,
      run_id,
      idempotency_key: idem,
      workflow_id: payload?.workflow_id,
      step_index: payload?.step_index,
    });

    await recordEvent(task_id, "task.created", {
      tool_name,
      run_id,
      risk_level: finalRisk,
      approval_reasons: risk.reasons || [],
    });
    if (advisory) {
      await recordEvent(task_id, "permission.council.advisory", advisory);
      if (typeof insertPermissionAuditRecord === "function") {
        await insertPermissionAuditRecord(pool, {
          run_id,
          task_id,
          tool_name,
          risk_level: finalRisk,
          ...advisory,
        });
      }
    }

    if (requiresApproval) {
      await recordEvent(task_id, "approval.requested", { tool_name, run_id, reasons: risk.reasons || [] });
    } else {
      const taskStream = getTaskStream(tool_name);
      await enqueueToStream(redis, taskStream, groupTask, {
        task_id,
        run_id,
        tool_name,
        payload: JSON.stringify(fullPayload),
        workflow_id: payload?.workflow_id || "",
        step_index: Number.isFinite(payload?.step_index) ? String(payload.step_index) : "",
      });
    }

    bindTaskToContext(task_id, context, tool_name);
    return { task_id, deduplicated: false, waiting_approval: requiresApproval, advisory };
  }

  async function enqueueWorkflow({ name, steps, run_id, context = null }) {
    const normalizedSteps = Array.isArray(steps) ? steps.filter(s => s && s.tool_name) : [];
    if (normalizedSteps.length === 0) return { ok: false, error: "No valid steps." };

    const workflow_id = uuidv4();
    await insertWorkflowDefinition(pool, workflow_id, String(name || "chat-workflow"), { steps: normalizedSteps });

    if (context) {
      context.totalTaskCount = normalizedSteps.length;
      context.completedTaskCount = 0;
    }

    const tasks = [];
    for (let i = 0; i < normalizedSteps.length; i++) {
      const step = normalizedSteps[i];
      const payload = { ...(step.payload || {}), workflow_id, step_index: i };
      const enq = await enqueueTask({
        tool_name: step.tool_name,
        payload,
        run_id,
        risk_level: step.risk_level,
        idempotency_key: makeIdempotencyKey(run_id, step.tool_name, payload),
        context,
      });
      tasks.push({ task_id: enq.task_id, tool_name: step.tool_name, waiting_approval: enq.waiting_approval });
    }
    return { ok: true, workflow_id, run_id, tasks };
  }

  return { enqueueTask, enqueueWorkflow };
}
