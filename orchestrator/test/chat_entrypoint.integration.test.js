import test from "node:test";
import assert from "node:assert/strict";

import { createHandleApiChat } from "../src/vnext/chat_entrypoint.js";

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

function createHarness({
  compositePlan = null,
  parsedIntent = { confidence: 0.2, requires_tools: false, mode_suggested: "chat", tool_name: "" },
  forcedIntent = null,
  vnextResult = null,
  legacyBrainData = null,
} = {}) {
  const calls = {
    ensureRun: [],
    pool: [],
    updateRunStatus: [],
    completeRunWithCostLedger: [],
    enqueueWorkflow: [],
    parseIntent: [],
    forcedIntent: [],
    executeVNextDispatch: [],
    callBrain: [],
  };

  const handleApiChat = createHandleApiChat({
    uuidv4: () => "run-chat-1",
    ensureRun: async (...args) => {
      calls.ensureRun.push(args);
    },
    planCompositeWorkflowFromText: async () => compositePlan,
    pool: {
      query: async (...args) => {
        calls.pool.push(args);
        return { rows: [] };
      },
    },
    updateRunStatus: async (...args) => {
      calls.updateRunStatus.push(args);
    },
    completeRunWithCostLedger: async (...args) => {
      calls.completeRunWithCostLedger.push(args);
    },
    enqueueWorkflow: async (args) => {
      calls.enqueueWorkflow.push(args);
      return { ok: true, workflow_id: "legacy-workflow", tasks: [{ task_id: "legacy-task-1" }] };
    },
    parseIntent: async (...args) => {
      calls.parseIntent.push(args);
      return structuredClone(parsedIntent);
    },
    buildForcedIntentFromRule: (...args) => {
      calls.forcedIntent.push(args);
      return forcedIntent ? structuredClone(forcedIntent) : null;
    },
    executeVNextDispatch: async (args) => {
      calls.executeVNextDispatch.push(args);
      return vnextResult || {
        ok: true,
        response_mode: "direct_reply",
        run_id: "run-chat-1",
        reply: "brain direct",
        task_envelope: {
          task_id: "env-chat-1",
          source: "api",
          raw_input: args.requestBody.raw_input,
          normalized_input: { text: args.requestBody.raw_input },
          intent: "chat",
          requires_orchestration: false,
          target_team: "brain",
          expected_outputs: ["direct_reply"],
          constraints: {},
          context: {},
        },
      };
    },
    buildVNextDispatchInput: ({ source, rawInput, payload }) => ({ source, raw_input: rawInput, ...payload }),
    forceLocalLlm: false,
    callBrainWithRetry: async (...args) => {
      calls.callBrain.push(args);
      return legacyBrainData || { narrative: "legacy brain", report_markdown: "", report_html_object_key: "", cost_ledger: {} };
    },
    currentLocalModel: "local-model",
    currentQwenModel: "qwen-model",
  });

  return { handleApiChat, calls };
}

test("/chat direct chat bypass uses vnext direct reply path", async () => {
  const { handleApiChat, calls } = createHarness({
    parsedIntent: { confidence: 0.1, requires_tools: false, mode_suggested: "chat", tool_name: "" },
  });
  const req = { body: { message: "你好，帮我总结一下今天进展" } };
  const res = createResponseRecorder();

  await handleApiChat(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.mode, "direct_reply");
  assert.equal(calls.executeVNextDispatch.length, 1);
  assert.equal(calls.callBrain.length, 0);
  assert.equal(calls.enqueueWorkflow.length, 0);
});

test("/chat coding intent can return approval_request through vnext dispatch", async () => {
  const { handleApiChat, calls } = createHarness({
    parsedIntent: {
      confidence: 0.9,
      requires_tools: true,
      mode_suggested: "tool",
      tool_name: "coding.delegate",
      payload: { destructive: true },
    },
    vnextResult: {
      ok: true,
      response_mode: "approval_request",
      run_id: "run-chat-1",
      task_envelope: {
        task_id: "env-approval-1",
        source: "api",
        raw_input: "删除旧文件并重建",
        normalized_input: { text: "删除旧文件并重建" },
        intent: "coding",
        requires_orchestration: false,
        target_team: "coding_team",
        expected_outputs: ["repo_changes"],
        constraints: { approval_mode: "manual" },
        context: {},
      },
      execution: {
        task_id: "task-approval-1",
        tool_name: "coding.delegate",
        waiting_approval: true,
      },
    },
  });
  const req = { body: { message: "删除旧文件并重建", payload: { destructive: true } } };
  const res = createResponseRecorder();

  await handleApiChat(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.mode, "task");
  assert.equal(res.body.waiting_approval, true);
  assert.equal(calls.executeVNextDispatch.length, 1);
  assert.equal(calls.callBrain.length, 0);
});

test("/chat quant intent stays on legacy analysis path", async () => {
  const { handleApiChat, calls } = createHarness({
    parsedIntent: {
      confidence: 0.95,
      requires_tools: true,
      mode_suggested: "tool",
      tool_name: "quant.deep_analysis",
      payload: { symbol: "AAPL" },
    },
  });
  const req = { body: { message: "分析 AAPL 现在是否值得买入" } };
  const res = createResponseRecorder();

  await handleApiChat(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body.ok, true);
  assert.equal(typeof res.body.narrative, "string");
  assert.equal(calls.executeVNextDispatch.length, 0);
  assert.equal(calls.callBrain.length, 1);
});
