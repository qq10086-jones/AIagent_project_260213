import test from "node:test";
import assert from "node:assert/strict";

import { createHandleApproveTask, createHandleRejectTask } from "../src/vnext/approval_entrypoint.js";

function createResponseRecorder() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return payload;
    },
  };
}

test("approve entrypoint requeues waiting task and pushes it to stream", async () => {
  const calls = { pool: [], events: [], workflow: [], xadd: [] };
  const handleApproveTask = createHandleApproveTask({
    approvalToken: "token-1",
    pool: {
      query: async (...args) => {
        calls.pool.push(args);
        if (String(args[0]).startsWith("SELECT task_id, tool_name")) {
          return {
            rows: [{
              task_id: "task-1",
              tool_name: "coding.delegate",
              payload_json: JSON.stringify({ prompt: "fix bug" }),
              run_id: "run-1",
              status: "waiting_approval",
              workflow_id: "wf-1",
              step_index: 0,
            }],
          };
        }
        return { rows: [] };
      },
    },
    getTaskForApproval: async (_pool, task_id) => ({
      task_id,
      tool_name: "coding.delegate",
      payload_json: JSON.stringify({ prompt: "fix bug" }),
      run_id: "run-1",
      status: "waiting_approval",
      workflow_id: "wf-1",
      step_index: 0,
    }),
    markTaskQueued: async () => {},
    updatePermissionAuditHumanDecision: async () => {},
    recordEvent: async (...args) => {
      calls.events.push(args);
    },
    workflowEngine: {
      handleTaskApproved: async (...args) => {
        calls.workflow.push(args);
      },
    },
    getTaskStream: (toolName) => `stream:${toolName}`,
    redis: {
      xadd: async (...args) => {
        calls.xadd.push(args);
      },
    },
  });

  const req = {
    params: { task_id: "task-1" },
    header: (name) => (name === "X-Approval-Token" ? "token-1" : ""),
  };
  const res = createResponseRecorder();

  await handleApproveTask(req, res);

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, { ok: true, task_id: "task-1" });
  assert.deepEqual(calls.events[0], ["task-1", "approval.approved", { task_id: "task-1" }]);
  assert.deepEqual(calls.workflow[0], ["task-1"]);
  assert.equal(calls.xadd.length, 1);
  assert.equal(calls.xadd[0][0], "stream:coding.delegate");
});

test("reject entrypoint fails waiting task and closes run when last pending task is removed", async () => {
  const calls = { pool: [], events: [], workflow: [], updateRunStatus: [] };
  const taskToContext = new Map([
    ["task-2", { run_id: "run-2", closeRunOnTaskResult: true, pendingTaskIds: new Set(["task-2"]) }],
  ]);
  const runToContext = new Map([["run-2", { run_id: "run-2" }]]);

  const handleRejectTask = createHandleRejectTask({
    approvalToken: "token-2",
    pool: {
      query: async (...args) => {
        calls.pool.push(args);
        return { rows: [] };
      },
    },
    updateRunStatus: async (...args) => {
      calls.updateRunStatus.push(args);
    },
    getTaskForRejection: async () => ({ task_id: "task-2", tool_name: "coding.delegate", run_id: "run-2", status: "waiting_approval" }),
    markTaskApprovalRejected: async () => {},
    countPendingTasksForRun: async () => 0,
    updatePermissionAuditHumanDecision: async () => {},
    recordEvent: async (...args) => {
      calls.events.push(args);
    },
    workflowEngine: {
      handleTaskRejected: async (...args) => {
        calls.workflow.push(args);
      },
    },
    normalizeResultPayload: (status, output, errorCode) => ({ status, output, error_code: errorCode }),
    taskToContext,
    runToContext,
  });

  const req = {
    params: { task_id: "task-2" },
    body: { reason: "manual reject" },
    header: (name) => (name === "X-Approval-Token" ? "token-2" : ""),
  };
  const res = createResponseRecorder();

  await handleRejectTask(req, res);

  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.body, { ok: true, task_id: "task-2" });
  assert.deepEqual(calls.events[0], ["task-2", "approval.rejected", { task_id: "task-2", reason: "manual reject" }]);
  assert.deepEqual(calls.workflow[0], ["task-2", "manual reject"]);
  assert.equal(taskToContext.has("task-2"), false);
  assert.equal(runToContext.has("run-2"), false);
  assert.equal(calls.updateRunStatus[0][1], "run-2");
  assert.equal(calls.updateRunStatus[0][2], "failed");
});

test("reject entrypoint returns 403 on invalid token", async () => {
  const handleRejectTask = createHandleRejectTask({
    approvalToken: "expected-token",
    pool: { query: async () => ({ rows: [] }) },
    updateRunStatus: async () => {},
    getTaskForRejection: async () => null,
    markTaskApprovalRejected: async () => {},
    countPendingTasksForRun: async () => 0,
    updatePermissionAuditHumanDecision: async () => {},
    recordEvent: async () => {},
    workflowEngine: { handleTaskRejected: async () => {} },
    normalizeResultPayload: () => ({}),
    taskToContext: new Map(),
    runToContext: new Map(),
  });

  const req = {
    params: { task_id: "task-3" },
    body: { reason: "manual reject" },
    header: () => "wrong-token",
  };
  const res = createResponseRecorder();

  await handleRejectTask(req, res);

  assert.equal(res.statusCode, 403);
  assert.deepEqual(res.body, { ok: false, error: "invalid approval token" });
});
