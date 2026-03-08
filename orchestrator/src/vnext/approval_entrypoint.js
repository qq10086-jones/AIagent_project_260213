export function createHandleApproveTask({
  approvalToken,
  pool,
  getTaskForApproval,
  markTaskQueued,
  recordEvent,
  workflowEngine,
  getTaskStream,
  redis,
}) {
  if (!pool?.query || typeof pool.query !== "function") throw new Error("pool.query is required");
  if (typeof getTaskForApproval !== "function") throw new Error("getTaskForApproval is required");
  if (typeof markTaskQueued !== "function") throw new Error("markTaskQueued is required");
  if (typeof recordEvent !== "function") throw new Error("recordEvent is required");
  if (!workflowEngine?.handleTaskApproved || typeof workflowEngine.handleTaskApproved !== "function") {
    throw new Error("workflowEngine.handleTaskApproved is required");
  }
  if (typeof getTaskStream !== "function") throw new Error("getTaskStream is required");
  if (!redis?.xadd || typeof redis.xadd !== "function") throw new Error("redis.xadd is required");

  return async function handleApproveTask(req, res) {
    const token = req.header("X-Approval-Token") || "";
    if (token !== approvalToken) {
      return res.status(403).json({ ok: false, error: "invalid approval token" });
    }

    const task_id = req.params.task_id;
    const task = await getTaskForApproval(pool, task_id);
    if (!task) {
      return res.status(404).json({ ok: false, error: "task not found" });
    }
    if (task.status !== "waiting_approval") {
      return res.status(409).json({ ok: false, error: `task status is ${task.status}` });
    }

    await markTaskQueued(pool, task_id);
    await recordEvent(task_id, "approval.approved", { task_id });
    await workflowEngine.handleTaskApproved(task_id).catch((err) => {
      console.warn(`[workflow] handleTaskApproved failed: ${err.message}`);
    });

    let payload = {};
    try {
      payload = JSON.parse(task.payload_json || "{}");
    } catch {
      payload = {};
    }

    const taskStream = getTaskStream(task.tool_name);
    await redis.xadd(
      taskStream,
      "*",
      "task_id",
      task_id,
      "run_id",
      task.run_id || "",
      "tool_name",
      task.tool_name,
      "payload",
      JSON.stringify(payload),
      "workflow_id",
      task.workflow_id || "",
      "step_index",
      Number.isFinite(task.step_index) ? String(task.step_index) : "",
    );

    return res.json({ ok: true, task_id });
  };
}

export function createHandleRejectTask({
  approvalToken,
  pool,
  updateRunStatus,
  getTaskForRejection,
  markTaskApprovalRejected,
  countPendingTasksForRun,
  recordEvent,
  workflowEngine,
  normalizeResultPayload,
  taskToContext,
  runToContext,
}) {
  if (!pool?.query || typeof pool.query !== "function") throw new Error("pool.query is required");
  if (typeof updateRunStatus !== "function") throw new Error("updateRunStatus is required");
  if (typeof getTaskForRejection !== "function") throw new Error("getTaskForRejection is required");
  if (typeof markTaskApprovalRejected !== "function") throw new Error("markTaskApprovalRejected is required");
  if (typeof countPendingTasksForRun !== "function") throw new Error("countPendingTasksForRun is required");
  if (typeof recordEvent !== "function") throw new Error("recordEvent is required");
  if (!workflowEngine?.handleTaskRejected || typeof workflowEngine.handleTaskRejected !== "function") {
    throw new Error("workflowEngine.handleTaskRejected is required");
  }
  if (typeof normalizeResultPayload !== "function") throw new Error("normalizeResultPayload is required");
  if (!taskToContext || typeof taskToContext.get !== "function") throw new Error("taskToContext is required");
  if (!runToContext || typeof runToContext.delete !== "function") throw new Error("runToContext is required");

  return async function handleRejectTask(req, res) {
    const token = req.header("X-Approval-Token") || "";
    if (token !== approvalToken) {
      return res.status(403).json({ ok: false, error: "invalid approval token" });
    }

    const task_id = req.params.task_id;
    const reason = String(req.body?.reason || "").trim();
    const task = await getTaskForRejection(pool, task_id);
    if (!task) {
      return res.status(404).json({ ok: false, error: "task not found" });
    }
    if (task.status !== "waiting_approval") {
      return res.status(409).json({ ok: false, error: `task status is ${task.status}` });
    }

    await markTaskApprovalRejected(
      pool,
      task_id,
      normalizeResultPayload("failed", { rejected: true, reason, approval: "rejected" }, "APPROVAL_REJECTED"),
    );
    await recordEvent(task_id, "approval.rejected", { task_id, reason });
    await workflowEngine.handleTaskRejected(task_id, reason).catch((err) => {
      console.warn(`[workflow] handleTaskRejected failed: ${err.message}`);
    });

    const ctx = taskToContext.get(task_id);
    if (ctx) {
      taskToContext.delete(task_id);
      if (ctx.pendingTaskIds instanceof Set) {
        ctx.pendingTaskIds.delete(task_id);
      }
      if ((ctx.pendingTaskIds?.size || 0) === 0 && ctx.closeRunOnTaskResult) {
        await updateRunStatus(pool, ctx.run_id, "failed").catch(() => {});
        runToContext.delete(ctx.run_id);
      }
    } else if (task.run_id) {
      const pendingCount = await countPendingTasksForRun(pool, task.run_id);
      if (pendingCount === 0) {
        await updateRunStatus(pool, task.run_id, "failed").catch(() => {});
      }
    }

    return res.json({ ok: true, task_id });
  };
}
