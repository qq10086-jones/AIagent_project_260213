import assert from "node:assert/strict";

import { createHandleApproveTask, createHandleRejectTask } from "../src/vnext/approval_entrypoint.js";

function createResponseHarness() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return this;
    },
  };
}

async function testApproveWritesHumanDecision() {
  const calls = [];
  const handler = createHandleApproveTask({
    approvalToken: "token",
    pool: { query: async () => ({ rows: [] }) },
    getTaskForApproval: async () => ({
      task_id: "task-1",
      status: "waiting_approval",
      payload_json: "{}",
      run_id: "run-1",
      tool_name: "coding.execute",
      workflow_id: "",
      step_index: "",
    }),
    markTaskQueued: async () => {},
    updatePermissionAuditHumanDecision: async (_pool, args) => {
      calls.push(args);
    },
    recordEvent: async () => {},
    workflowEngine: { handleTaskApproved: async () => {} },
    getTaskStream: () => "stream:task",
    redis: { xadd: async () => {} },
  });
  const res = createResponseHarness();
  await handler({ header: () => "token", params: { task_id: "task-1" } }, res);
  assert.equal(res.body.ok, true);
  assert.deepEqual(calls[0], { task_id: "task-1", final_human_decision: "approved" });
}

async function testRejectWritesHumanDecision() {
  const calls = [];
  const taskToContext = new Map();
  const runToContext = new Map();
  const handler = createHandleRejectTask({
    approvalToken: "token",
    pool: { query: async () => ({ rows: [] }) },
    updateRunStatus: async () => {},
    getTaskForRejection: async () => ({
      task_id: "task-2",
      status: "waiting_approval",
      run_id: "run-2",
    }),
    markTaskApprovalRejected: async () => {},
    countPendingTasksForRun: async () => 0,
    updatePermissionAuditHumanDecision: async (_pool, args) => {
      calls.push(args);
    },
    recordEvent: async () => {},
    workflowEngine: { handleTaskRejected: async () => {} },
    normalizeResultPayload: (...args) => ({ args }),
    taskToContext,
    runToContext,
  });
  const res = createResponseHarness();
  await handler({ header: () => "token", params: { task_id: "task-2" }, body: { reason: "too risky" } }, res);
  assert.equal(res.body.ok, true);
  assert.deepEqual(calls[0], { task_id: "task-2", final_human_decision: "rejected" });
}

async function main() {
  await testApproveWritesHumanDecision();
  await testRejectWritesHumanDecision();
  console.log("approval_entrypoint.permission_audit.test.js: all tests passed");
}

main().catch((err) => {
  console.error("approval_entrypoint.permission_audit.test.js: failed");
  console.error(err);
  process.exit(1);
});
