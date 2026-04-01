import test from "node:test";
import assert from "node:assert/strict";

import {
  buildPiRuntimeUpdateMessage,
  buildRuntimeUpdateMessage,
  createStreamAdapter,
} from "../../shared/stream_adapter.js";

test("stream adapter builds quant terminal messages and requests default skip", () => {
  const built = buildRuntimeUpdateMessage({
    taskId: "task-q1",
    status: "succeeded",
    toolName: "quant.deep_analysis",
    output: { briefing: "Top candidate: 5020.T" },
  });
  assert.equal(built.kind, "quant_terminal");
  assert.equal(built.shouldSkipDefault, true);
  assert.match(built.text, /5020\.T/);
});

test("stream adapter throttles duplicate claimed updates", async () => {
  const sent = [];
  const adapter = createStreamAdapter({
    discord: {
      channels: {
        fetch: async () => ({
          send: async (text) => {
            sent.push(text);
            return { id: `msg-${sent.length}` };
          },
        }),
      },
    },
    safeTranslate: async (text) => text,
    replyChunked: async () => [],
    throttleMs: 1000,
    logger: { warn() {} },
  });

  const ctx = { channelId: "c1", lang: "zh" };
  const first = await adapter.sendTaskUpdate({ ctx, taskId: "task-1", status: "claimed", toolName: "coding.delegate" });
  const second = await adapter.sendTaskUpdate({ ctx, taskId: "task-1", status: "claimed", toolName: "coding.delegate" });

  assert.equal(first.delivered, true);
  assert.equal(second.reason, "throttled");
  assert.equal(sent.length, 1);
});

test("stream adapter builds progress messages for tool_call and tool_result", () => {
  const callUpdate = buildRuntimeUpdateMessage({
    taskId: "task-1",
    status: "tool_call",
    toolName: "coding.execute",
    output: { input_summary: "npm test" },
  });
  assert.match(callUpdate.text, /running/);
  assert.match(callUpdate.text, /npm test/);

  const resultUpdate = buildRuntimeUpdateMessage({
    taskId: "task-1",
    status: "tool_result",
    toolName: "coding.execute",
    output: { summary: "tests passed" },
  });
  assert.match(resultUpdate.text, /progress/);
  assert.match(resultUpdate.text, /tests passed/);
});

test("stream adapter builds pi-native session update messages", () => {
  const toolCall = buildPiRuntimeUpdateMessage({
    taskId: "task-2",
    event: {
      type: "tool_call",
      tag: "tool_call",
      title: "bash ls -la",
    },
  });
  assert.match(toolCall.text, /tool running/);
  assert.match(toolCall.text, /bash ls -la/);

  const toolProgress = buildPiRuntimeUpdateMessage({
    taskId: "task-2",
    event: {
      type: "status",
      tag: "tool_call_update",
      text: "bash completed in 2.1s",
    },
  });
  assert.match(toolProgress.text, /tool progress/);
  assert.match(toolProgress.text, /2\.1s/);

  const agentChunk = buildPiRuntimeUpdateMessage({
    taskId: "task-2",
    event: {
      type: "text_delta",
      tag: "agent_message_chunk",
      text: "I found the bug in auth middleware.",
    },
  });
  assert.match(agentChunk.text, /agent update/);
  assert.match(agentChunk.text, /auth middleware/);
});

test("stream adapter can send pi session updates with throttling", async () => {
  const sent = [];
  const adapter = createStreamAdapter({
    discord: {
      channels: {
        fetch: async () => ({
          send: async (text) => {
            sent.push(text);
            return { id: `msg-${sent.length}` };
          },
        }),
      },
    },
    safeTranslate: async (text) => text,
    replyChunked: async () => [],
    throttleMs: 1000,
    logger: { warn() {} },
  });

  const ctx = { channelId: "c2", lang: "zh" };
  const event = { type: "text_delta", tag: "agent_message_chunk", text: "stream chunk" };
  const first = await adapter.sendPiSessionUpdate({ ctx, taskId: "task-pi", event });
  const second = await adapter.sendPiSessionUpdate({ ctx, taskId: "task-pi", event });

  assert.equal(first.delivered, true);
  assert.equal(second.reason, "throttled");
  assert.equal(sent.length, 1);
});
