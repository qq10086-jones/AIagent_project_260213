import test from "node:test";
import assert from "node:assert/strict";

import { deliverWorkflowRuntimeNotification } from "../src/vnext/workflow_notification_delivery.js";

test("workflow notification delivery sends transition message for succeeded workflow step", async () => {
  const sent = [];
  const channel = {
    send: async (message) => {
      sent.push(message);
      return { id: "msg-1" };
    },
  };

  const result = await deliverWorkflowRuntimeNotification({
    workflowTerminal: { handled: true, workflow_run_id: "wf-1", step_index: 0, step_id: "pm_spec" },
    notifyCtx: { channelId: "chan-1", lang: "zh" },
    discord: {
      channels: {
        fetch: async () => channel,
      },
    },
    workflowEngine: {
      getWorkflowRunStatus: async () => ({
        steps: [
          { step_index: 0, step_id: "pm_spec", role_name: "pm", status: "succeeded" },
          { step_index: 1, step_id: "arch_design", role_name: "architect", status: "queued" },
        ],
      }),
    },
    status: "succeeded",
    safeTranslate: async (text) => text,
  });

  assert.equal(result.delivered, true);
  assert.equal(result.kind, "transition");
  assert.equal(sent.length, 1);
  assert.match(String(sent[0]), /^\[NEXUS\] /);
});

test("workflow notification delivery uses chunked reply for failure message", async () => {
  const chunked = [];
  const channel = {
    send: async () => {
      throw new Error("should not use direct send for failure");
    },
  };

  const result = await deliverWorkflowRuntimeNotification({
    workflowTerminal: { handled: true, workflow_run_id: "wf-2", step_index: 1, step_id: "impl_be" },
    notifyCtx: { channelId: "chan-2", lang: "zh" },
    discord: {
      channels: {
        fetch: async () => channel,
      },
    },
    workflowEngine: {
      getWorkflowRunStatus: async () => ({
        steps: [
          { step_index: 0, step_id: "pm_spec", role_name: "pm", status: "succeeded" },
          { step_index: 1, step_id: "impl_be", role_name: "backend", status: "failed" },
        ],
      }),
    },
    status: "failed",
    normalizedErrorCode: "STEP_FAILED",
    streamError: "compiler exploded",
    output: { stderr: "traceback" },
    safeTranslate: async (text) => text,
    replyChunked: async (_channel, text) => {
      chunked.push(text);
      return [{ id: "msg-2" }];
    },
  });

  assert.equal(result.delivered, true);
  assert.equal(result.kind, "failure");
  assert.equal(chunked.length, 1);
  assert.match(String(chunked[0]), /^\[NEXUS\] /);
});

test("workflow notification delivery skips when runtime context is missing", async () => {
  const result = await deliverWorkflowRuntimeNotification({
    workflowTerminal: { handled: true, workflow_run_id: "wf-3" },
    notifyCtx: null,
    discord: { channels: { fetch: async () => null } },
    workflowEngine: { getWorkflowRunStatus: async () => null },
  });

  assert.deepEqual(result, { delivered: false, reason: "missing_runtime_dependencies" });
});
