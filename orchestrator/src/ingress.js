import { v4 as uuidv4 } from "uuid";
import { pool, redis, recordEvent, upsertTask, getToolSpec, getTaskStream } from "./index.js";
import { analyzeTaskRisk } from "./policy.js";

/**
 * Ingress Layer
 * Handles external API requests and translates them into internal tasks.
 */

export async function handleExecuteTool(req, res) {
  const { tool_name, payload, run_id, risk_level, idempotency_key, context } = req.body || {};

  if (!tool_name || !payload) {
    return res.status(400).json({ ok: false, error: "tool_name and payload are required" });
  }

  try {
    const spec = getToolSpec(tool_name);
    const risk = analyzeTaskRisk(tool_name, payload);
    
    const finalRisk = risk_level || risk.risk_level;
    const requiresApproval = risk.requires_approval;

    const task_id = uuidv4();
    const fullPayload = { ...payload, run_id };

    await upsertTask({
      task_id,
      tool_name,
      status: requiresApproval ? "waiting_approval" : "queued",
      risk_level: finalRisk,
      payload: fullPayload,
      run_id,
      idempotency_key
    });

    await recordEvent(task_id, "task.created", { tool_name, run_id, risk_level: finalRisk });

    if (requiresApproval) {
      await recordEvent(task_id, "approval.requested", { reasons: risk.reasons });
    } else {
      const taskStream = getTaskStream(tool_name);
      await redis.xadd(taskStream, "*", "task_id", task_id, "payload", JSON.stringify(fullPayload), "tool_name", tool_name);
    }

    return res.json({ ok: true, task_id });
  } catch (err) {
    console.error(`[ingress] Failed to execute tool: ${err.message}`);
    return res.status(500).json({ ok: false, error: err.message });
  }
}

export async function handleApproveTask(req, res, APPROVAL_TOKEN) {
  const token = req.header("X-Approval-Token") || "";
  if (token !== APPROVAL_TOKEN) {
    return res.status(403).json({ ok: false, error: "invalid approval token" });
  }

  const { task_id } = req.params;
  try {
    const row = await pool.query("SELECT * FROM tasks WHERE task_id=$1", [task_id]);
    if (row.rows.length === 0) return res.status(404).json({ ok: false, error: "task not found" });
    
    const task = row.rows[0];
    if (task.status !== "waiting_approval") return res.status(409).json({ ok: false, error: `status is ${task.status}` });

    await pool.query("UPDATE tasks SET status='queued', updated_at=NOW() WHERE task_id=$1", [task_id]);
    await recordEvent(task_id, "approval.approved");

    const taskStream = getTaskStream(task.tool_name);
    await redis.xadd(taskStream, "*", "task_id", task_id, "payload", task.payload_json, "tool_name", task.tool_name);

    return res.json({ ok: true, task_id });
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
}

export async function handleRejectTask(req, res, APPROVAL_TOKEN) {
  const token = req.header("X-Approval-Token") || "";
  if (token !== APPROVAL_TOKEN) {
    return res.status(403).json({ ok: false, error: "invalid approval token" });
  }

  const { task_id } = req.params;
  try {
    const row = await pool.query("SELECT * FROM tasks WHERE task_id=$1", [task_id]);
    if (row.rows.length === 0) return res.status(404).json({ ok: false, error: "task not found" });
    
    const task = row.rows[0];
    if (task.status !== "waiting_approval") return res.status(409).json({ ok: false, error: `status is ${task.status}` });

    await pool.query("UPDATE tasks SET status='failed', updated_at=NOW() WHERE task_id=$1", [task_id]);
    await recordEvent(task_id, "approval.rejected");

    return res.json({ ok: true, task_id });
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
}
