import test from "node:test";
import assert from "node:assert/strict";

import { normalizeInputRequest } from "../src/vnext/input_normalizer.js";
import { createTaskEnvelope, validateTaskEnvelope } from "../src/vnext/task_envelope.js";
import { routeTaskRequest } from "../src/vnext/brain_router.js";
import { loadRegistryOrThrow } from "../src/registry.js";

const registry = loadRegistryOrThrow(new URL("../../configs/registry/capability_registry.json", import.meta.url));

test("normalizeInputRequest extracts discord metadata", () => {
  const result = normalizeInputRequest({
    discord_event: {
      content: "build a crm mvp",
      channel_id: "c1",
      thread_id: "t1",
      author: { id: "u1", username: "lin" },
      attachments: [{ filename: "spec.png", url: "https://example.com/spec.png", content_type: "image/png" }],
    },
  });

  assert.equal(result.source, "discord");
  assert.equal(result.raw_input, "build a crm mvp");
  assert.equal(result.context.channel_id, "c1");
  assert.equal(result.normalized_input.attachments.length, 1);
});

test("createTaskEnvelope validates required fields", () => {
  const envelope = createTaskEnvelope({
    source: "discord",
    raw_input: "implement login page",
    intent: "coding",
    requires_orchestration: false,
    target_team: "coding_team",
    expected_outputs: ["repo_changes"],
    constraints: { local_only: true },
    context: { channel_id: "abc" },
  });

  const validation = validateTaskEnvelope(envelope);
  assert.equal(validation.ok, true);
});

test("routeTaskRequest chooses orchestrated workflow for complex coding requests", () => {
  const routed = routeTaskRequest({
    source: "discord",
    raw_input: "Build a full-stack CRM MVP with PM, architecture, frontend, backend, QA, and deliver tests.",
    normalized_input: { text: "Build a full-stack CRM MVP with PM, architecture, frontend, backend, QA, and deliver tests." },
    context: { channel_id: "123" },
    registry,
  });

  assert.equal(routed.decision, "orchestrated_workflow");
  assert.equal(routed.task_envelope.intent, "coding");
  assert.equal(routed.task_envelope.execution_plan.workflow_id, "coding_team_v0");
});

test("routeTaskRequest keeps chat requests in direct reply path", () => {
  const routed = routeTaskRequest({
    source: "api",
    raw_input: "你好，今天过得怎么样？",
    normalized_input: { text: "你好，今天过得怎么样？" },
    context: {},
    registry,
  });

  assert.equal(routed.decision, "direct_reply");
  assert.equal(routed.task_envelope.target_team, "brain");
});
