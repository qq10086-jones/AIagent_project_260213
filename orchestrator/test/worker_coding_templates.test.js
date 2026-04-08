import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { createStepBuilder } from "../src/domain/workflow_step_builder.js";

async function main() {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "worker-coding-template-"));
  fs.mkdirSync(path.join(workspaceRoot, "sandbox", "crm_site"), { recursive: true });
  fs.writeFileSync(path.join(workspaceRoot, "sandbox", "crm_site", "app.js"), "export const app = 1;\n", "utf8");
  fs.writeFileSync(path.join(workspaceRoot, "sandbox", "crm_site", "package.json"), JSON.stringify({
    name: "worker-coding-template-test",
    scripts: {
      lint: "node --check workspace/sandbox/crm_site/app.js",
      typecheck: "node --version",
      build: "node --version",
    },
  }, null, 2), "utf8");
  const registry = {
    project_types: {
      webapp_crm: {
        acceptance_suite: "suite_v1",
      },
    },
    acceptance_suites: {
      suite_v1: {
        commands: ["node --version"],
        required_reports: [],
      },
    },
  };
  const promptScriptRegistry = {
    scripts: {
      "frontend.impl.v1": {
        script_id: "frontend.impl.v1",
        role: "frontend",
        llm_role: "frontend",
        artifact_type: "implementation",
      },
      "frontend.impl.v2": {
        script_id: "frontend.impl.v2",
        role: "frontend",
        llm_role: "frontend",
        artifact_type: "implementation",
      },
    },
  };
  const handoffContracts = { handoffs: {} };
  const { buildStepPayload } = createStepBuilder({
    registry,
    promptScriptRegistry,
    handoffContracts,
    workspaceRoot,
    runtimeConfig: {},
  });

  const payload = await buildStepPayload({
    run: {
      run_id: "run-1",
      workflow_run_id: "wf-1",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({
        goal: "update checkout page",
        step_payloads: {
          impl_fe: {
            beta_template_id: "wc.fe_modify.v1",
            target_paths: ["sandbox/crm_site/app.js"],
          },
        },
      }),
    },
    stepDef: {
      id: "impl_fe",
      role: "frontend",
      tool: "coding.delegate",
      prompt_script_id: "frontend.impl.v1",
      gate: "policy",
    },
    stepIndex: 3,
  });

  assert.equal(payload.beta_template_id, "wc.fe_modify.v1");
  assert.equal(payload.task_class, "fe_modify");
  assert.deepEqual(payload.context_envelope, {
    max_files: 15,
    max_tokens: 24000,
    dependency_depth: 2,
    context_source: "template",
  });
  assert.deepEqual(payload.template_verification_tiers, ["lint", "type_check", "build"]);
  assert.deepEqual(payload.verification_plan, [
    {
      tier: "syntax_check",
      command: "node --check sandbox/crm_site/app.js",
      required: true,
      source: "inferred_target_paths",
    },
    {
      tier: "lint",
      command: "npm --prefix sandbox/crm_site run lint",
      required: false,
      source: "package_json:sandbox/crm_site/package.json",
    },
    {
      tier: "type_check",
      command: "npm --prefix sandbox/crm_site run typecheck",
      required: false,
      source: "package_json:sandbox/crm_site/package.json",
    },
    {
      tier: "build",
      command: "npm --prefix sandbox/crm_site run build",
      required: false,
      source: "package_json:sandbox/crm_site/package.json",
    },
  ]);
  assert.equal(payload.verification_command, "node --check sandbox/crm_site/app.js");
  assert.ok(Array.isArray(payload.human_acceptance_criteria));

  await assert.rejects(async () => {
    await buildStepPayload({
      run: {
        run_id: "run-2",
        workflow_run_id: "wf-2",
        workflow_id: "coding_team_v0",
        project_type: "webapp_crm",
        input_json: JSON.stringify({
          goal: "update checkout page",
          step_payloads: {
            impl_fe: {
              beta_template_id: "wc.fe_modify.v1",
              task_class: "be_create",
              target_paths: ["sandbox/crm_site/app.js"],
            },
          },
        }),
      },
      stepDef: {
        id: "impl_fe",
        role: "frontend",
        tool: "coding.delegate",
        prompt_script_id: "frontend.impl.v1",
        gate: "policy",
      },
      stepIndex: 3,
    });
  }, /task_class mismatch/);
  console.log("worker_coding_templates.test.js: all tests passed");
}

main().catch((err) => {
  console.error("worker_coding_templates.test.js: failed");
  console.error(err);
  process.exit(1);
});
