import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

import { loadRegistryOrThrow } from "../src/registry.js";
import { routeTaskRequest } from "../src/vnext/brain_router.js";
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
