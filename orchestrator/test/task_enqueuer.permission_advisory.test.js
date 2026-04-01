import assert from "node:assert/strict";

import { createTaskEnqueuer } from "../src/vnext/task_enqueuer.js";

async function main() {
  const upserted = [];
  const streamWrites = [];
  const events = [];
  const auditRows = [];

  const enqueuer = createTaskEnqueuer({
    pool: {},
    redis: {},
    registry: {},
    analyzeTaskRisk: () => ({ risk_level: "medium", requires_approval: false, reasons: [] }),
    validateTaskInputAgainstRegistry: () => ({ ok: true, errors: [] }),
    getToolSpec: () => ({}),
    upsertTask: async (task) => { upserted.push(task); },
    getTaskStream: () => "stream:task:coding",
    findTaskIdByIdempotencyKey: async () => null,
    enqueueToStream: async (_redis, stream, group, message) => {
      streamWrites.push({ stream, group, message });
    },
    bindTaskToContext: () => {},
    recordEvent: async (taskId, eventType, payload) => {
      events.push({ taskId, eventType, payload });
    },
    insertWorkflowDefinition: async () => {},
    makeIdempotencyKey: () => "idem-1",
    groupTask: "cg:workers",
    evaluatePermissionAdvisory: () => ({
      council_advice: "allow",
      safety_verdict: "allow",
      context_verdict: "sufficient",
      risk_level: "medium",
      risk_score: 0.45,
      advisory_summary: "Routine coding task",
      duration_ms: 3,
    }),
    insertPermissionAuditRecord: async (_pool, record) => {
      auditRows.push(record);
    },
  });

  const queued = await enqueuer.enqueueTask({
    tool_name: "coding.delegate",
    payload: { task_prompt: "fix navbar spacing" },
    run_id: "run-1",
    context: null,
  });

  assert.equal(queued.waiting_approval, false);
  assert.equal(queued.advisory?.council_advice, "allow");
  assert.equal(upserted.length, 1);
  assert.equal(upserted[0].payload.permission_advisory.council_advice, "allow");
  assert.equal(auditRows.length, 1);
  assert.equal(auditRows[0].task_id, upserted[0].task_id);
  assert.equal(events.some((event) => event.eventType === "permission.council.advisory"), true);

  const streamPayload = JSON.parse(streamWrites[0].message.payload);
  assert.equal(streamPayload.permission_advisory.council_advice, "allow");

  console.log("task_enqueuer.permission_advisory.test.js: all tests passed");
}

main().catch((err) => {
  console.error("task_enqueuer.permission_advisory.test.js: failed");
  console.error(err);
  process.exit(1);
});
