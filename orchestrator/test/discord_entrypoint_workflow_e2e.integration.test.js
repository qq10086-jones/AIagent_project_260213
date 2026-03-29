import test from "node:test";
import assert from "node:assert/strict";

import { createDiscordGateway } from "../src/adapters/discord_gateway.js";
import { createDiscordMessageHandler } from "../src/adapters/discord_message_handler.js";

test("discord coder directive can simulate full workflow notification chain", async () => {
  const replies = [];
  const channelMessages = [];
  const progressEdits = [];
  const calls = {
    ensureRun: [],
    executeVNextDispatch: [],
  };

  const gateway = createDiscordGateway({
    translate: async (text) => text,
  });

  const progressMessage = {
    id: "progress-1",
    content: "",
    async edit(nextContent) {
      this.content = nextContent;
      progressEdits.push(nextContent);
      return this;
    },
  };

  const channel = {
    id: "chan-1",
    messages: {
      fetch: async (messageId) => (messageId === progressMessage.id ? progressMessage : null),
    },
    async sendTyping() {
      return true;
    },
    async send(content) {
      channelMessages.push(content);
      return { id: `channel-msg-${channelMessages.length}`, content };
    },
  };

  gateway.client.channels.fetch = async (channelId) => (channelId === channel.id ? channel : null);

  const handler = createDiscordMessageHandler({
    redis: {
      set: async () => "OK",
    },
    discord: {
      user: { id: "bot-1" },
    },
    approvalToken: "approval-token",
    coderProviderDefault: "opencode",
    coderModelDefault: "minimax-coding-plan/MiniMax-M2.7",
    ensureRun: async (...args) => {
      calls.ensureRun.push(args);
    },
    updateRunStatus: async () => {},
    findRunIdByClientMsgId: async () => null,
    enqueueTask: async () => ({ ok: true }),
    makeIdempotencyKey: () => "dedupe-key",
    getToolSpec: () => null,
    executeVNextDispatch: async (args) => {
      calls.executeVNextDispatch.push(args);
      return {
        ok: true,
        response_mode: "task",
        execution: {
          workflow_run_id: "wf-discord-1",
          first_step: {
            step_id: "pm_spec",
            waiting_approval: false,
          },
        },
      };
    },
    appState: {
      currentLocalModel: "local-model",
      forceLocalLlm: false,
    },
    currentQwenModel: () => "qwen-max",
    setQwenModel: () => {},
    translate: async (text) => text,
    safeTranslate: async (text) => text,
    replyChunked: gateway.replyChunked,
    runToContext: gateway.runToContext,
    workflowRunToContext: gateway.workflowRunToContext,
    registry: {
      project_types: {
        generic_coding_task: {
          default_workflow: "coding_team_v0",
        },
      },
    },
  });

  const msg = {
    id: "discord-msg-1",
    content: "/coder: Build a CRM MVP with frontend and backend",
    author: {
      bot: false,
      id: "user-1",
      username: "lin",
    },
    channel,
    attachments: [],
    async reply(content) {
      replies.push(content);
      if (String(content).includes("Waiting for workflow to start")) {
        return progressMessage;
      }
      return { id: `reply-${replies.length}`, content };
    },
  };

  await handler.handleDiscordMessage(msg);

  assert.equal(calls.ensureRun.length, 1);
  assert.equal(calls.executeVNextDispatch.length, 1);
  assert.equal(replies.length, 2);
  assert.match(String(replies[0]), /Coding Team/);
  assert.match(String(replies[1]), /Waiting for workflow to start/);

  const notifyCtx = gateway.workflowRunToContext.get("wf-discord-1");
  assert.deepEqual(notifyCtx, {
    channelId: "chan-1",
    lang: "en",
    progressMessageId: "progress-1",
    runId: calls.executeVNextDispatch[0].run_id,
  });

  await gateway.sendStepTransitionNotification({
    event: "workflow.started",
    workflow_run_id: "wf-discord-1",
    step_count: 7,
    step_ids: ["pm_spec", "arch_design", "impl_be", "impl_fe", "smoke_test", "qa_verify", "release_pack"],
  });

  await gateway.sendStepTransitionNotification({
    event: "step.started",
    workflow_run_id: "wf-discord-1",
    step_id: "pm_spec",
    step_index: 0,
  });

  await gateway.sendStepTransitionNotification({
    event: "workflow.completed",
    workflow_run_id: "wf-discord-1",
    result_url: "https://preview.example.test",
    run_summary: "npm install\nnode server.js",
  });

  assert.equal(channelMessages.length, 2);
  assert.match(String(channelMessages[0]), /^\[NEXUS\] /);
  assert.match(String(channelMessages[0]), /1\/7/);
  assert.match(String(channelMessages[1]), /preview\.example\.test/);
  assert.match(String(channelMessages[1]), /npm install/);
  assert.equal(progressEdits.length, 1);
  assert.match(String(progressEdits[0]), /preview\.example\.test/);
});
