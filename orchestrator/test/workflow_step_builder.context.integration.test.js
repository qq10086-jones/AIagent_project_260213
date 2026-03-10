import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { createStepBuilder } from "../src/domain/workflow_step_builder.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-step-context-"));
}

function writeFile(workspaceRoot, rel, content) {
  const abs = path.join(workspaceRoot, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function makeBuilder(workspaceRoot) {
  return createStepBuilder({
    workspaceRoot,
    registry: {
      project_types: { webapp_crm: { acceptance_suite: "webapp_crm_v0" } },
      acceptance_suites: { webapp_crm_v0: { commands: ["node --version"], required_reports: [] } },
    },
    promptScriptRegistry: {
      scripts: {
        "backend.impl.v2": { script_id: "backend.impl.v2", role: "backend", llm_role: "backend", validation: {} },
      },
    },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {
      execution: { diff_first_enabled: true },
      worker_coder: {
        max_attempts_default: 2,
        same_error_repeat_limit_default: 2,
        wall_clock_timeout_s_default: 480,
      },
    },
  });
}

test("impl_be payload includes context packet, repo map, and coding context block", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "package.json", JSON.stringify({ name: "crm-app" }, null, 2));
  writeFile(workspaceRoot, "sandbox/crm_site/server.js", "function startServer() { return 'ok'; }\n");
  writeFile(workspaceRoot, "sandbox/crm_site/server.test.js", "test('server', () => {})\n");

  const { buildStepPayload } = makeBuilder(workspaceRoot);
  const payload = buildStepPayload({
    run: {
      run_id: "run-ctx",
      workflow_run_id: "wf-run-ctx",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Implement backend API", provider: "opencode", model: "qwen-coder-next" }),
    },
    stepDef: {
      id: "impl_be",
      role: "backend",
      tool: "coding.delegate",
      gate: "policy",
      prompt_script_id: "backend.impl.v2",
    },
    stepIndex: 2,
  });

  assert.equal(payload.context_packet.step_id, "impl_be");
  assert.equal(payload.context_packet.role, "backend");
  assert.match(JSON.stringify(payload.repo_map.candidate_files), /sandbox\/crm_site\/server\.js/);
  assert.match(payload.task_prompt, /\[Coding Context Packet\]/);
  assert.match(payload.task_prompt, /Target Paths:/);
  assert.match(payload.tool_adapter_request.payload.task_prompt, /\[Coding Context Packet\]/);
  assert.equal(payload.tool_adapter_request.payload.context_packet.step_id, "impl_be");
  assert.equal(payload.verification_command, "node --check sandbox/crm_site/server.js");
  assert.equal(payload.max_attempts, 2);
  assert.equal(payload.same_error_repeat_limit, 2);
  assert.equal(payload.wall_clock_timeout_s, 480);
  assert.equal(payload.tool_adapter_request.payload.verification_command, "node --check sandbox/crm_site/server.js");
  assert.equal(payload.tool_adapter_request.payload.max_attempts, 2);
  assert.equal(payload.tool_adapter_request.payload.same_error_repeat_limit, 2);
  assert.equal(payload.tool_adapter_request.payload.wall_clock_timeout_s, 480);
});
