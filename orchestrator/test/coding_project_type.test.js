import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadRegistryOrThrow, validateTaskInputAgainstRegistry } from "../src/registry.js";
import { inferCodingProjectType, resolveCodingProjectType, resolveCodingWorkflowId } from "../src/vnext/coding_project_type.js";
import { routeTaskRequest } from "../src/vnext/brain_router.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const registry = loadRegistryOrThrow(path.resolve(__dirname, "..", "..", "configs", "registry", "capability_registry.json"));

test("coding project type infers single_file_html for simple html requests", () => {
  assert.equal(inferCodingProjectType("帮我写一个 html，内容就写：hello, world"), "single_file_html");
});

test("coding project type infers webapp_crm for crm requests", () => {
  assert.equal(inferCodingProjectType("Build a CRM MVP with customer portal and sales pipeline"), "webapp_crm");
});

test("coding project type falls back to generic_coding_task when request is broad", () => {
  assert.equal(inferCodingProjectType("帮我做一个 coding team 补丁"), "generic_coding_task");
});

test("coding workflow defaults resolve through generic_coding_task", () => {
  assert.equal(resolveCodingWorkflowId(registry), "coding_team_v0");
  assert.equal(resolveCodingProjectType("做一个管理后台", registry), "generic_app");
});

test("brain router assigns single_file_html project type for html workflow requests", () => {
  const routed = routeTaskRequest({
    source: "discord",
    raw_input: "/coder 帮我写一个 html，内容就写：hello, world",
    normalized_input: { text: "/coder 帮我写一个 html，内容就写：hello, world" },
    context: { channel_id: "chan-html" },
    registry,
  });

  assert.equal(routed.decision, "orchestrated_workflow");
  assert.equal(routed.task_envelope.execution_plan.workflow_id, "coding_team_v0");
  assert.equal(routed.task_envelope.execution_plan.project_type, "single_file_html");
});

test("brain router keeps crm requests on crm project type", () => {
  const routed = routeTaskRequest({
    source: "discord",
    raw_input: "Build a CRM MVP with frontend backend QA and tests",
    normalized_input: { text: "Build a CRM MVP with frontend backend QA and tests" },
    context: { channel_id: "chan-crm" },
    registry,
  });

  assert.equal(routed.decision, "orchestrated_workflow");
  assert.equal(routed.task_envelope.execution_plan.project_type, "webapp_crm");
});

test("registry validator allows specialized project type on shared workflow", () => {
  const checked = validateTaskInputAgainstRegistry({
    registry,
    tool_name: "coding.delegate",
    payload: {
      workflow_id: "coding_team_v0",
      project_type: "single_file_html",
      role: "frontend",
      step_id: "impl_fe",
    },
  });

  assert.equal(checked.ok, true);
});
