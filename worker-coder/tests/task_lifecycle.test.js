import assert from "node:assert/strict";

import { createTaskLifecycle } from "../task_lifecycle.js";

async function testTimeoutWinsSingleFinalization() {
  const events = [];
  const lifecycle = createTaskLifecycle({
    taskId: "task-1",
    msgId: "msg-1",
    toolName: "coding.delegate",
    runId: "run-1",
    emitResult: async (...args) => events.push(["emit", ...args]),
    writeFact: async (...args) => events.push(["fact", ...args]),
    ackMessage: async (...args) => events.push(["ack", ...args]),
  });

  await lifecycle.emitClaimed();
  assert.equal(lifecycle.isAborted(), false);

  const timeoutApplied = await lifecycle.finalizeTimeout(new Error("GLOBAL_TASK_TIMEOUT"));
  const resultApplied = await lifecycle.finalizeResult({
    ok: true,
    output: { ok: true },
    error: null,
  });

  assert.equal(timeoutApplied, true);
  assert.equal(resultApplied, false);
  assert.equal(lifecycle.isAborted(), true);
  assert.deepEqual(events, [
    ["emit", "task-1", "claimed"],
    ["fact", "run-1", "coder", { tool_name: "coding.delegate", error: "GLOBAL_TASK_TIMEOUT", success: false, timed_out: true }],
    ["emit", "task-1", "failed", { error: "GLOBAL_TASK_TIMEOUT", plan: "failed_during_execution" }, "GLOBAL_TASK_TIMEOUT"],
    ["ack", "msg-1"],
  ]);
}

async function testSuccessWinsSingleFinalization() {
  const events = [];
  const lifecycle = createTaskLifecycle({
    taskId: "task-2",
    msgId: "msg-2",
    toolName: "coding.execute",
    runId: "run-2",
    emitResult: async (...args) => events.push(["emit", ...args]),
    writeFact: async (...args) => events.push(["fact", ...args]),
    ackMessage: async (...args) => events.push(["ack", ...args]),
  });

  const successApplied = await lifecycle.finalizeResult({
    ok: true,
    output: { exit_code: 0 },
    error: null,
  });
  const failureApplied = await lifecycle.finalizeExecutionFailure(new Error("late failure"));

  assert.equal(successApplied, true);
  assert.equal(failureApplied, false);
  assert.deepEqual(events, [
    ["fact", "run-2", "coder", { tool_name: "coding.execute", output: { exit_code: 0 }, success: true }],
    ["emit", "task-2", "succeeded", { exit_code: 0 }, null],
    ["ack", "msg-2"],
  ]);
}

async function testFinalizeFallsBackWhenPrimaryEmitFails() {
  const events = [];
  let emitCalls = 0;
  const lifecycle = createTaskLifecycle({
    taskId: "task-3",
    msgId: "msg-3",
    toolName: "coding.delegate",
    runId: "run-3",
    emitResult: async (...args) => {
      emitCalls += 1;
      if (emitCalls === 1) {
        throw new Error("stream result payload too large");
      }
      events.push(["emit", ...args]);
    },
    writeFact: async (...args) => events.push(["fact", ...args]),
    ackMessage: async (...args) => events.push(["ack", ...args]),
  });

  const applied = await lifecycle.finalizeResult({
    ok: false,
    output: { summary: "oversized result" },
    error: "delegate failed",
  });

  assert.equal(applied, true);
  assert.deepEqual(events, [
    ["fact", "run-3", "coder", { tool_name: "coding.delegate", output: { summary: "oversized result" }, success: false }],
    ["emit", "task-3", "failed", { error: "stream result payload too large", plan: "failed_during_result_emit", original_status: "failed" }, "stream result payload too large"],
    ["ack", "msg-3"],
  ]);
}

async function main() {
  await testTimeoutWinsSingleFinalization();
  await testSuccessWinsSingleFinalization();
  await testFinalizeFallsBackWhenPrimaryEmitFails();
  console.log("task_lifecycle.test.js: all tests passed");
}

main().catch((err) => {
  console.error("task_lifecycle.test.js: failed");
  console.error(err);
  process.exit(1);
});
