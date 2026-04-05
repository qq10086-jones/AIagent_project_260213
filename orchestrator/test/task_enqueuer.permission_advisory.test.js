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

  // --- Council deny pre-intercept ---
  const denyEvents = [];
  const denyUpserted = [];
  const denyStreamWrites = [];

  const denyEnqueuer = createTaskEnqueuer({
    pool: {},
    redis: {},
    registry: {},
    analyzeTaskRisk: () => ({ risk_level: "high", requires_approval: true, reasons: ["destructive_command"] }),
    validateTaskInputAgainstRegistry: () => ({ ok: true, errors: [] }),
    getToolSpec: () => ({}),
    upsertTask: async (task) => { denyUpserted.push(task); },
    getTaskStream: () => "stream:task:coding",
    findTaskIdByIdempotencyKey: async () => null,
    enqueueToStream: async (_redis, stream, group, message) => {
      denyStreamWrites.push({ stream, group, message });
    },
    bindTaskToContext: () => {},
    recordEvent: async (taskId, eventType, payload) => {
      denyEvents.push({ taskId, eventType, payload });
    },
    insertWorkflowDefinition: async () => {},
    makeIdempotencyKey: () => "idem-deny-1",
    groupTask: "cg:workers",
    evaluatePermissionAdvisory: () => ({
      council_advice: "deny",
      safety_verdict: "deny",
      context_verdict: "sufficient",
      risk_level: "high",
      risk_score: 0.98,
      advisory_summary: "Critical risk indicators detected: destructive_command",
      reasons: ["destructive_command"],
      duration_ms: 2,
    }),
    insertPermissionAuditRecord: async () => {},
  });

  const denied = await denyEnqueuer.enqueueTask({
    tool_name: "coding.execute",
    payload: { command: "rm -rf /" },
    run_id: "run-deny-1",
    context: null,
  });

  assert.equal(denied.council_denied, true);
  assert.equal(denied.waiting_approval, false);
  assert.equal(denied.advisory?.council_advice, "deny");
  assert.equal(denyStreamWrites.length, 0, "deny should not enqueue to stream");
  assert.equal(denyEvents.some((e) => e.eventType === "permission.council.denied"), true);
  const denyEvent = denyEvents.find((e) => e.eventType === "permission.council.denied");
  assert.ok(denyEvent.payload.advisory_summary.includes("destructive_command"));

  // --- Approval event carries advisory summary ---
  const approvalEvents = [];

  const approvalEnqueuer = createTaskEnqueuer({
    pool: {},
    redis: {},
    registry: {},
    analyzeTaskRisk: () => ({ risk_level: "high", requires_approval: true, reasons: ["sensitive_path_or_secret"] }),
    validateTaskInputAgainstRegistry: () => ({ ok: true, errors: [] }),
    getToolSpec: () => ({}),
    upsertTask: async () => {},
    getTaskStream: () => "stream:task:coding",
    findTaskIdByIdempotencyKey: async () => null,
    enqueueToStream: async () => {},
    bindTaskToContext: () => {},
    recordEvent: async (taskId, eventType, payload) => {
      approvalEvents.push({ taskId, eventType, payload });
    },
    insertWorkflowDefinition: async () => {},
    makeIdempotencyKey: () => "idem-approval-1",
    groupTask: "cg:workers",
    evaluatePermissionAdvisory: () => ({
      council_advice: "review",
      safety_verdict: "review",
      context_verdict: "sufficient",
      risk_level: "high",
      risk_score: 0.75,
      advisory_summary: "Human approval recommended: sensitive_path_or_secret",
      reasons: ["sensitive_path_or_secret"],
      duration_ms: 1,
    }),
    insertPermissionAuditRecord: async () => {},
  });

  const approved = await approvalEnqueuer.enqueueTask({
    tool_name: "coding.execute",
    payload: { command: "cat /etc/passwd" },
    run_id: "run-approval-1",
    context: null,
  });

  assert.equal(approved.waiting_approval, true);
  const approvalRequestedEvent = approvalEvents.find((e) => e.eventType === "approval.requested");
  assert.ok(approvalRequestedEvent, "approval.requested event must exist");
  assert.equal(approvalRequestedEvent.payload.advisory_summary, "Human approval recommended: sensitive_path_or_secret");
  assert.equal(approvalRequestedEvent.payload.council_advice, "review");
  assert.equal(approvalRequestedEvent.payload.risk_score, 0.75);

  console.log("task_enqueuer.permission_advisory.test.js: all tests passed");
}

main().catch((err) => {
  console.error("task_enqueuer.permission_advisory.test.js: failed");
  console.error(err);
  process.exit(1);
});
