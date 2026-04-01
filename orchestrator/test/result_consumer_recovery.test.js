import test from "node:test";
import assert from "node:assert/strict";

import { createResultConsumer } from "../src/vnext/result_consumer.js";

function createHarness({
  xautoclaimResponse = ["0-0", []],
  xreadgroupResponse = null,
} = {}) {
  const calls = {
    xack: [],
    markTaskRunning: [],
    updateTaskTerminalResult: [],
    workflowClaimed: [],
    workflowTerminal: [],
    runStatus: [],
  };

  const consumer = createResultConsumer({
    pool: {},
    redis: {
      xautoclaim: async () => xautoclaimResponse,
      xreadgroup: async () => xreadgroupResponse,
      xack: async (...args) => {
        calls.xack.push(args);
      },
    },
    workflowEngine: {
      handleTaskClaimed: async (taskId) => {
        calls.workflowClaimed.push(taskId);
      },
      handleTaskTerminal: async (payload) => {
        calls.workflowTerminal.push(payload);
        return { handled: true };
      },
    },
    normalizeResultPayload: (status, output, errorCode) => ({ status, output, error_code: errorCode }),
    normalizeErrorCode: (status) => (status === "failed" ? "TASK_FAILED" : null),
    getRunInputText: async () => "",
    findRunIdByTaskId: async () => "run-1",
    countPendingTasksForRun: async () => 0,
    countFailedWorkflowRunsForRun: async () => 0,
    countFailedTasksForRun: async () => 0,
    updateRunStatusIfNotFailed: async (...args) => {
      calls.runStatus.push(args);
    },
    updateTaskTerminalResult: async (...args) => {
      calls.updateTaskTerminalResult.push(args);
    },
    forceTaskFailed: async () => {},
    markTaskRunning: async (...args) => {
      calls.markTaskRunning.push(args);
    },
    recordEvent: async () => {},
    taskToContext: new Map(),
    runToContext: new Map(),
    discord: { channels: { fetch: async () => null } },
    replyChunked: async () => [],
    safeTranslate: async (text) => text,
    createResultEmbed: () => ({}),
    createBinaryAttachment: () => ({}),
    insertTrace: async () => {},
    s3: { send: async () => ({}) },
    callLocalOllamaReply: async () => "",
    detectProject: () => "general",
    formatCodingDelegateResult: () => "ok",
    summarizeOutputBrief: () => "ok",
    deliverWorkflowRuntimeNotification: async () => ({ delivered: false }),
    streamResult: "stream:result",
    groupResult: "cg:orchestrator",
  });

  return { consumer, calls };
}

test("result consumer reclaims stale claimed messages before reading new ones", async () => {
  const { consumer, calls } = createHarness({
    xautoclaimResponse: [
      "0-0",
      [
        ["177-0", ["task_id", "task-1", "status", "claimed"]],
      ],
    ],
  });

  const result = await consumer.tick();

  assert.deepEqual(result, { reclaimed: 1 });
  assert.equal(calls.markTaskRunning.length, 1);
  assert.equal(calls.workflowClaimed.length, 1);
  assert.equal(calls.xack.length, 1);
  assert.equal(calls.xack[0][0], "stream:result");
});

test("result consumer processes fresh succeeded messages and updates task status", async () => {
  const { consumer, calls } = createHarness({
    xautoclaimResponse: ["0-0", []],
    xreadgroupResponse: [
      [
        "stream:result",
        [
          ["188-0", ["task_id", "task-2", "status", "succeeded", "output", "{\"ok\":true}"]],
        ],
      ],
    ],
  });

  const result = await consumer.tick();

  assert.deepEqual(result, { processed: 1 });
  assert.equal(calls.updateTaskTerminalResult.length, 1);
  assert.equal(calls.workflowTerminal.length, 1);
  assert.equal(calls.xack.length, 1);
  assert.equal(calls.runStatus.length, 1);
});

test("result consumer routes pi session updates through progress path without terminal status update", async () => {
  const taskToContext = new Map([
    ["task-pi", { channelId: "c-pi", lang: "zh", tool_name: "coding.delegate" }],
  ]);
  const sent = [];
  const calls = {
    recordEvent: [],
    terminal: [],
  };

  const consumer = createResultConsumer({
    pool: {},
    redis: {
      xautoclaim: async () => ["0-0", []],
      xreadgroup: async () => [[
        "stream:result",
        [[
          "199-0",
          [
            "task_id", "task-pi",
            "status", "pi_session_update",
            "output", JSON.stringify({
              tool_name: "coding.delegate",
              event: {
                type: "text_delta",
                tag: "agent_message_chunk",
                text: "streamed delegate update",
              },
            }),
          ],
        ]],
      ]],
      xack: async () => {},
    },
    workflowEngine: {
      handleTaskClaimed: async () => {},
      handleTaskTerminal: async () => null,
    },
    normalizeResultPayload: (status, output, errorCode) => ({ status, output, error_code: errorCode }),
    normalizeErrorCode: () => null,
    getRunInputText: async () => "",
    findRunIdByTaskId: async () => "run-pi",
    countPendingTasksForRun: async () => 0,
    countFailedWorkflowRunsForRun: async () => 0,
    countFailedTasksForRun: async () => 0,
    updateRunStatusIfNotFailed: async () => {},
    updateTaskTerminalResult: async (...args) => {
      calls.terminal.push(args);
    },
    forceTaskFailed: async () => {},
    markTaskRunning: async () => {},
    recordEvent: async (...args) => {
      calls.recordEvent.push(args);
    },
    taskToContext,
    runToContext: new Map(),
    discord: {
      channels: {
        fetch: async () => ({
          send: async (text) => {
            sent.push(text);
            return { id: `m-${sent.length}` };
          },
        }),
      },
    },
    replyChunked: async () => [],
    safeTranslate: async (text) => text,
    createResultEmbed: () => ({}),
    createBinaryAttachment: () => ({}),
    insertTrace: async () => {},
    s3: { send: async () => ({}) },
    callLocalOllamaReply: async () => "",
    detectProject: () => "general",
    formatCodingDelegateResult: () => "ok",
    summarizeOutputBrief: () => "ok",
    deliverWorkflowRuntimeNotification: async () => ({ delivered: false }),
    streamResult: "stream:result",
    groupResult: "cg:orchestrator",
  });

  const result = await consumer.tick();
  assert.deepEqual(result, { processed: 1 });
  assert.equal(calls.terminal.length, 0);
  assert.equal(calls.recordEvent.length, 1);
  assert.match(String(calls.recordEvent[0][1]), /task\.pi_session_update/);
  assert.equal(sent.length, 1);
  assert.match(String(sent[0]), /agent update/);
});
