import assert from "node:assert/strict";
import { createAuditHooks } from "../../shared/contracts/audit_hooks.js";

async function testAuditHooksInsertEvents() {
  const calls = [];
  const hooks = createAuditHooks({
    pool: {
      async query(sql, params) {
        calls.push({ sql: String(sql), params });
        return { rows: [] };
      },
    },
    logger: { warn() {} },
  });

  await hooks.onTaskStart("run-1", { task_id: "task-1", tool_name: "coding.delegate" }, { taskId: "task-1", workerName: "worker-coder" });
  await hooks.onToolCall("run-1", "coding.delegate", { prompt: "x" }, { taskId: "task-1", workerName: "worker-coder" });
  await hooks.onToolResult("run-1", "coding.delegate", { ok: true }, { taskId: "task-1", workerName: "worker-coder" });
  await hooks.onTaskComplete("run-1", { ok: true }, { taskId: "task-1", workerName: "worker-coder" });
  await hooks.onTaskError("run-1", { message: "boom" }, { taskId: "task-1", workerName: "worker-coder" });

  assert.ok(calls.length >= 6);
  assert.match(calls[0].sql, /CREATE TABLE IF NOT EXISTS execution_audit_log/);
  assert.match(calls[1].sql, /INSERT INTO execution_audit_log/);
  assert.equal(calls[1].params[0], "run-1");
  assert.equal(calls[1].params[1], "task-1");
}

await testAuditHooksInsertEvents();
console.log("audit_hooks.test.js: all tests passed");
