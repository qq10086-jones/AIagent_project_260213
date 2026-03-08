import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

import { loadRegistryOrThrow } from "../src/registry.js";
import { routeTaskRequest } from "../src/vnext/brain_router.js";
import { applyRoutingPolicy } from "../src/vnext/brain_router_policy.js";
import { createTaskEnvelope } from "../src/vnext/task_envelope.js";

import { fileURLToPath } from "url";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const registry = loadRegistryOrThrow(path.resolve(__dirname, "..", "..", "configs", "registry", "capability_registry.json"));

test("brain router returns direct_reply for chat input", () => {
  const routed = routeTaskRequest({
    source: "discord",
    raw_input: "你好，最近怎么样？",
    normalized_input: { text: "你好，最近怎么样？" },
    context: { channel_id: "chat-1" },
    registry,
  });

  assert.equal(routed.decision, "direct_reply");
  assert.equal(routed.task_envelope.intent, "chat");
  assert.equal(routed.task_envelope.requires_orchestration, false);
});

test("brain router returns orchestrated workflow for complex coding input", () => {
  const routed = routeTaskRequest({
    source: "discord",
    raw_input: "Build a full-stack CRM MVP with PM, architecture, frontend, backend, QA, and tests.",
    normalized_input: { text: "Build a full-stack CRM MVP with PM, architecture, frontend, backend, QA, and tests." },
    context: { channel_id: "code-1" },
    registry,
  });

  assert.equal(routed.decision, "orchestrated_workflow");
  assert.equal(routed.task_envelope.intent, "coding");
  assert.equal(routed.task_envelope.execution_plan.workflow_id, "coding_team_v0");
});

// WS-13-03: Policy override integration tests

test("policy P-01: /coder prefix forces orchestrated_workflow regardless of analyzerResult", () => {
  const routed = routeTaskRequest({
    source: "api",
    raw_input: "/coder build a login page",
    normalized_input: { text: "/coder build a login page" },
    context: {},
    analyzerResult: { intent: "chat", confidence: 0.9 },
    registry,
  });
  assert.equal(routed.decision, "orchestrated_workflow");
  assert.equal(routed.task_envelope.intent, "coding");
});

test("policy P-02: trivial input (< 3 tokens) forces direct_reply", () => {
  const routed = routeTaskRequest({
    source: "api",
    raw_input: "hi there",
    normalized_input: { text: "hi there" },
    context: {},
    registry,
  });
  assert.equal(routed.decision, "direct_reply");
  assert.equal(routed.task_envelope.intent, "chat");
});

test("policy P-05: null analyzerResult (LLM failed) forces direct_reply", () => {
  const result = applyRoutingPolicy(
    "Build me a full stock trading platform with alerts",
    null
  );
  assert.equal(result.applied_rule, "P-05");
  assert.equal(result.decision, "direct_reply");
  assert.equal(result.override, true);
  assert.equal(result.error_note, "LLM_CLASSIFICATION_FAILED");
});

test("policy P-04: unknown intent downgrades to direct_reply", () => {
  const result = applyRoutingPolicy(
    "Do the thing with the stuff for the project",
    { intent: "unknown", confidence: 0.3 }
  );
  assert.equal(result.applied_rule, "P-04");
  assert.equal(result.decision, "direct_reply");
  assert.equal(result.override, true);
});

test("policy P-06: unknown execution cue returns clarification_required", () => {
  const result = applyRoutingPolicy(
    "build a customer portal with approvals",
    undefined,
    "unknown"
  );
  assert.equal(result.applied_rule, "P-06");
  assert.equal(result.decision, "clarification_required");
  assert.equal(Boolean(result.clarification_prompt), true);
});

test("policy unknown without execution cues downgrades to direct_reply", () => {
  const result = applyRoutingPolicy(
    "what is the weather over there",
    undefined,
    "unknown"
  );
  assert.equal(result.override, true);
  assert.equal(result.applied_rule, "P-04");
  assert.equal(result.decision, "direct_reply");
});

test("policy P-01 still takes priority over P-06", () => {
  const result = applyRoutingPolicy(
    "/coder build a customer portal",
    undefined,
    "unknown"
  );
  assert.equal(result.applied_rule, "P-01");
  assert.equal(result.decision, "orchestrated_workflow");
});

test("policy no-override: valid coding intent passes through unchanged", () => {
  const result = applyRoutingPolicy(
    "Build a full-stack CRM MVP with PM, architecture, frontend, backend, QA, and tests.",
    { intent: "coding", confidence: 0.85 }
  );
  assert.equal(result.override, false);
  assert.equal(result.applied_rule, null);
});

test("task envelope creation preserves explicit contracts", () => {
  const envelope = createTaskEnvelope({
    source: "discord",
    raw_input: "implement login page",
    intent: "coding",
    requires_orchestration: false,
    target_team: "coding_team",
    expected_outputs: ["repo_changes"],
    constraints: { local_only: true, approval_mode: "manual", risk_level: "medium" },
    context: { channel_id: "abc" },
  });

  assert.equal(envelope.target_team, "coding_team");
  assert.deepEqual(envelope.expected_outputs, ["repo_changes"]);
});
