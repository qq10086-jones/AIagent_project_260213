import test from "node:test";
import assert from "node:assert/strict";

import { createExecuteVNextDispatch } from "../src/vnext/runtime_dispatch.js";
import { createTaskEnvelope } from "../src/vnext/task_envelope.js";

function createHarness({ routeDecision = "direct_reply", analyzerResult = null, enqueueResult = null } = {}) {
  const calls = {
    ensureRun: [],
    parseIntent: [],
    pool: [],
    updateRunStatus: [],
    directReply: [],
    enqueueTask: [],
    workflow: [],
  };

  const routeTaskRequest = ({ raw_input, analyzerResult: finalAnalyzerResult, context }) => ({
    decision: routeDecision,
    task_envelope: createTaskEnvelope({
      task_id: `env-${routeDecision}`,
      source: "api",
      raw_input,
      normalized_input: { text: raw_input },
      intent: routeDecision === "direct_reply" ? "chat" : "coding",
      requires_orchestration: routeDecision === "orchestrated_workflow",
      target_team: routeDecision === "direct_reply" ? "brain" : "coding_team",
      expected_outputs: routeDecision === "direct_reply" ? ["direct_reply"] : ["repo_changes", "execution_log"],
      constraints: { local_only: true, approval_mode: "manual", risk_level: "medium" },
      context,
      execution_plan:
        routeDecision === "orchestrated_workflow"
          ? { workflow_id: "coding_team_v0", project_type: "webapp_crm" }
          : routeDecision === "single_agent"
            ? { tool_name: "coding.delegate", project_type: "webapp_crm" }
            : { mode: "direct_reply" },
      decision: routeDecision,
    }),
    finalAnalyzerResult,
  });

  const dispatcher = createExecuteVNextDispatch({
    ensureRun: async (...args) => {
      calls.ensureRun.push(args);
    },
    parseIntent: async (...args) => {
      calls.parseIntent.push(args);
      return analyzerResult;
    },
    registry: { workflows: { coding_team_v0: { project_type: "webapp_crm" } } },
    generateBrainDirectReply: async (...args) => {
      calls.directReply.push(args);
      return "brain reply";
    },
    pool: {
      query: async (...args) => {
        calls.pool.push(args);
        return { rows: [] };
      },
    },
    updateRunStatus: async (...args) => {
      calls.updateRunStatus.push(args);
    },
    enqueueTask: async (args) => {
      calls.enqueueTask.push(args);
      return enqueueResult || { task_id: "task-1", waiting_approval: false };
    },
    workflowEngine: {
      startWorkflowRun: async (args) => {
        calls.workflow.push(args);
        return {
          workflow_run_id: "wf-1",
          workflow_id: args.workflow_id,
          first_step: { step_id: "pm.plan", waiting_approval: false },
        };
      },
    },
    routeTaskRequest,
    coderProviderDefault: "opencode",
    coderModelDefault: "minimax-m2.5",
  });

  return { dispatcher, calls };
}

test("runtime dispatch direct reply completes run without task enqueue", async () => {
  const { dispatcher, calls } = createHarness({ routeDecision: "direct_reply" });

  const result = await dispatcher({
    requestBody: { source: "api", raw_input: "你好，帮我总结一下今天的工作" },
    run_id: "run-direct",
  });

  assert.equal(result.ok, true);
  assert.equal(result.response_mode, "direct_reply");
  assert.equal(result.reply, "brain reply");
  assert.equal(calls.directReply.length, 1);
  assert.equal(calls.enqueueTask.length, 0);
  assert.equal(calls.workflow.length, 0);
  assert.equal(calls.updateRunStatus.at(-1)?.[1], "run-direct");
  assert.equal(calls.updateRunStatus.at(-1)?.[2], "completed");
});

test("runtime dispatch single-agent path preserves approval_request behavior", async () => {
  const { dispatcher, calls } = createHarness({
    routeDecision: "single_agent",
    analyzerResult: { payload: { risk: "high" } },
    enqueueResult: { task_id: "task-risky", waiting_approval: true },
  });

  const result = await dispatcher({
    requestBody: { source: "api", raw_input: "删除生产环境旧文件", payload: { destructive: true } },
    run_id: "run-approval",
    context: { channelId: "chan-1" },
  });

  assert.equal(result.ok, true);
  assert.equal(result.response_mode, "approval_request");
  assert.equal(result.execution.task_id, "task-risky");
  assert.equal(result.execution.waiting_approval, true);
  assert.equal(calls.enqueueTask.length, 1);
  assert.equal(calls.enqueueTask[0].context.channelId, "chan-1");
  assert.equal(calls.enqueueTask[0].payload.task_prompt, "删除生产环境旧文件");
  assert.equal(calls.enqueueTask[0].payload.destructive, true);
});

test("runtime dispatch workflow path passes context into workflow engine", async () => {
  const { dispatcher, calls } = createHarness({ routeDecision: "orchestrated_workflow" });

  const result = await dispatcher({
    requestBody: {
      source: "api",
      raw_input: "Build a CRM MVP with frontend backend QA and tests",
      provider: "codex",
      model: "gpt-5-codex",
      fast_mode: true,
    },
    run_id: "run-workflow",
    context: { channelId: "chan-2", reply: async () => {} },
  });

  assert.equal(result.ok, true);
  assert.equal(result.response_mode, "progress_update");
  assert.equal(result.execution.workflow_run_id, "wf-1");
  assert.equal(calls.workflow.length, 1);
  assert.equal(calls.workflow[0].context.channelId, "chan-2");
  assert.equal(calls.workflow[0].input.provider, "codex");
  assert.equal(calls.workflow[0].input.model, "gpt-5-codex");
  assert.equal(calls.workflow[0].input.fast_mode, true);
});
