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
        execution_lane_default: "stable_cloud_lane",
        execution_lanes: {
          stable_cloud_lane: {
            provider: "opencode",
            model: "minimax-coding-plan/MiniMax-M2.7",
          },
        },
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
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/server.js", "function startServer() { return 'ok'; }\n");
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/server.test.js", "test('server', () => {})\n");

  const { buildStepPayload } = makeBuilder(workspaceRoot);
  const payload = buildStepPayload({
    run: {
      run_id: "run-ctx",
      workflow_run_id: "wf-run-ctx",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Implement backend API", provider: "opencode", model: "minimax-coding-plan/MiniMax-M2.7" }),
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
  assert.equal(payload.execution_lane, "stable_cloud_lane");
  assert.equal(payload.provider, "opencode");
  assert.equal(payload.model, "minimax-coding-plan/MiniMax-M2.7");
  assert.match(JSON.stringify(payload.repo_map.candidate_files), /sandbox\/crm_site\/server\.js/);
  assert.match(payload.task_prompt, /\[Coding Context Packet\]/);
  assert.match(payload.task_prompt, /Target Paths:/);
  assert.match(payload.tool_adapter_request.payload.task_prompt, /\[Coding Context Packet\]/);
  assert.equal(payload.tool_adapter_request.payload.context_packet.step_id, "impl_be");
  assert.equal(payload.verification_command, "node --check workspace/sandbox/crm_site/server.js");
  assert.equal(payload.max_attempts, 2);
  assert.equal(payload.same_error_repeat_limit, 2);
  assert.equal(payload.wall_clock_timeout_s, 480);
  assert.equal(payload.tool_adapter_request.payload.verification_command, "node --check workspace/sandbox/crm_site/server.js");
  assert.equal(payload.tool_adapter_request.payload.max_attempts, 2);
  assert.equal(payload.tool_adapter_request.payload.same_error_repeat_limit, 2);
  assert.equal(payload.tool_adapter_request.payload.wall_clock_timeout_s, 480);
});

// wall_clock_timeout_s clamp boundary tests
// The clamp is: clampInt(configured, max(30, max_runtime_s), 3600, max(max_runtime_s, 300))
// impl_be max_runtime_s=240, arch_design=480 (from runtimeByStep in workflow_step_builder.js)

test("wall_clock_timeout_s is clamped UP when config is below max_runtime_s", () => {
  const workspaceRoot = makeWorkspace();
  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: { scripts: {} },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {
      worker_coder: {
        execution_lane_default: "stable_cloud_lane",
        execution_lanes: { stable_cloud_lane: { provider: "opencode", model: "m" } },
        wall_clock_timeout_s_default: 60,  // below impl_be max_runtime_s=240
        max_attempts_default: 1,
        same_error_repeat_limit_default: 1,
      },
    },
  });
  const payload = builder.buildStepPayload({
    run: { run_id: "r1", workflow_run_id: "w1", workflow_id: "coding_team_v0", project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "impl", provider: "opencode" }) },
    stepDef: { id: "impl_be", role: "backend", tool: "coding.delegate", gate: "policy", prompt_script_id: "" },
    stepIndex: 2,
  });
  // clamp(60, max(30,240), 3600, max(240,300)) = clamp(60, 240, 3600, 300) ↁEclamped up to 240
  assert.equal(payload.wall_clock_timeout_s, 240,
    "wall_clock below max_runtime_s must be clamped up to max_runtime_s");
});

test("wall_clock_timeout_s=900 passes through for arch_design (max_runtime_s=480)", () => {
  // Production scenario: runtime_defaults.json wall_clock_timeout_s_default=900
  // arch_design max_runtime_s=480 ↁEclamp(900, max(30,480), 3600, max(480,300)) = 900
  const workspaceRoot = makeWorkspace();
  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: { scripts: {} },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {
      worker_coder: {
        execution_lane_default: "stable_local_lane",
        execution_lanes: { stable_local_lane: { provider: "opencode", model: "ollama/glm-4.7-flash:latest" } },
        wall_clock_timeout_s_default: 900,
        max_attempts_default: 2,
        same_error_repeat_limit_default: 2,
      },
    },
  });
  const payload = builder.buildStepPayload({
    run: { run_id: "r2", workflow_run_id: "w2", workflow_id: "coding_team_v0", project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "arch", provider: "opencode" }) },
    stepDef: { id: "arch_design", role: "architect", tool: "coding.delegate", gate: "policy", prompt_script_id: "" },
    stepIndex: 1,
  });
  assert.equal(payload.wall_clock_timeout_s, 900,
    "wall_clock=900 must not be clamped down when max_runtime_s=480");
});

test("wall_clock_timeout_s defaults to max(max_runtime_s, 300) when not configured", () => {
  // No wall_clock_timeout_s_default ↁEfallback = max(max_runtime_s, 300)
  // impl_be max_runtime_s=240 ↁEdefault = max(240, 300) = 300
  const workspaceRoot = makeWorkspace();
  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: { scripts: {} },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {
      worker_coder: {
        execution_lane_default: "stable_cloud_lane",
        execution_lanes: { stable_cloud_lane: { provider: "opencode", model: "m" } },
        max_attempts_default: 1,
        same_error_repeat_limit_default: 1,
      },
    },
  });
  const payload = builder.buildStepPayload({
    run: { run_id: "r3", workflow_run_id: "w3", workflow_id: "coding_team_v0", project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "impl", provider: "opencode" }) },
    stepDef: { id: "impl_be", role: "backend", tool: "coding.delegate", gate: "policy", prompt_script_id: "" },
    stepIndex: 2,
  });
  assert.equal(payload.wall_clock_timeout_s, 300,
    "missing wall_clock config must default to max(max_runtime_s=240, 300)=300");
});


test("stable_cloud_lane impl_be avoids structured_patch and keeps full-file outputs", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "package.json", JSON.stringify({ name: "crm-app" }, null, 2));
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/server.js", "function startServer() { return 'ok'; }\n");

  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: {
      scripts: {
        "backend.impl.v2": { script_id: "backend.impl.v2", role: "backend", llm_role: "backend", validation: {} },
        "backend.impl.v1": { script_id: "backend.impl.v1", role: "backend", llm_role: "backend", validation: {} },
      },
    },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {
      execution: { diff_first_enabled: true },
      worker_coder: {
        execution_lane_default: "stable_cloud_lane",
        execution_lanes: {
          stable_cloud_lane: { provider: "opencode", model: "minimax-coding-plan/MiniMax-M2.7" },
          
        },
        wall_clock_timeout_s_default: 900,
        max_attempts_default: 2,
        same_error_repeat_limit_default: 2,
      },
    },
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "mini-cloud-run",
      workflow_run_id: "mini-cloud-wf",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Implement backend API", provider: "opencode", model: "minimax-coding-plan/MiniMax-M2.7", execution_lane: "stable_cloud_lane" }),
    },
    stepDef: {
      id: "impl_be",
      role: "backend",
      tool: "coding.delegate",
      gate: "policy",
      prompt_script_id: "backend.impl.v1",
    },
    stepIndex: 2,
  });

  assert.equal(payload.execution_lane, "stable_cloud_lane");
  assert.equal(payload.execution_mode_requested, "full_file_fallback");
  assert.equal(payload.prompt_script_id, "backend.impl.v1");
  assert.deepEqual(payload.expected_artifacts, ["impl/be_changes/server.js", "impl/be_changes/package.json", "impl/be_notes.md", "handoff/be_to_fe.json"]);
});

test("generic_app impl_be injects primary_minimax_lane and uses full_file_fallback", () => {
  const workspaceRoot = makeWorkspace();
  // workspace/sandbox/app is empty — no target files — natural full_file_fallback
  fs.mkdirSync(path.join(workspaceRoot, "workspace/sandbox/app"), { recursive: true });

  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: {
      scripts: {
        "backend.impl.v1": { script_id: "backend.impl.v1", role: "backend", llm_role: "backend", validation: {} },
      },
    },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {
      execution: { diff_first_enabled: true },
      worker_coder: {
        execution_lane_default: "stable_cloud_lane",
        execution_lanes: {
          stable_cloud_lane: { provider: "opencode", model: "minimax-coding-plan/MiniMax-M2.7" },
          primary_minimax_lane: { provider: "opencode", model: "minimax-coding-plan/MiniMax-M2.7" },
        },
        wall_clock_timeout_s_default: 900,
        max_attempts_default: 2,
        same_error_repeat_limit_default: 2,
      },
    },
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "generic-run",
      workflow_run_id: "generic-wf",
      workflow_id: "coding_team_v0",
      project_type: "generic_app",
      input_json: JSON.stringify({ goal: "Build todo app" }),
    },
    stepDef: {
      id: "impl_be",
      role: "backend",
      tool: "coding.delegate",
      gate: "policy",
      prompt_script_id: "backend.impl.v1",
    },
    stepIndex: 2,
  });

  assert.equal(payload.execution_lane, "primary_minimax_lane", "generic_app impl_be should use primary_minimax_lane");
  assert.equal(payload.model, "minimax-coding-plan/MiniMax-M2.7", "model should match primary_minimax_lane");
  assert.equal(payload.execution_mode_requested, "full_file_fallback", "primary_minimax_lane forced to full_file_fallback");
});

test("deploy_preview payload includes release metadata and preview defaults", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/index.html", "<!doctype html><html></html>");

  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: { scripts: {} },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {},
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "preview-run",
      workflow_run_id: "preview-wf",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Build CRM preview" }),
    },
    stepDef: {
      id: "deploy_preview",
      role: "release",
      tool: "ops.deploy_preview",
      gate: "finalize",
    },
    stepIndex: 6,
  });

  assert.equal(payload.preview_provider, "local");
  assert.equal(payload.preview_ttl_hours, 24);
  assert.equal(payload.release_manifest_path, "artifacts/release/preview-run/meta/run_manifest.json");
  assert.equal(payload.release_notes_path, "artifacts/release/preview-run/release/release_notes.md");
  assert.deepEqual(payload.target_paths, ["workspace/sandbox/crm_site/"]);
  assert.deepEqual(payload.project_root_candidates, [
    "artifacts/release/preview-run/impl/be_changes",
    "artifacts/release/preview-run/impl/fe_changes/public",
    "workspace/sandbox/crm_site/",
  ]);
  assert.equal(payload.render_service_id, "");
  assert.equal(payload.model_override, "minimax-coding-plan/MiniMax-M2.7");
});

test("release_pack payload pins fast model lane for webapp_crm", () => {
  const workspaceRoot = makeWorkspace();
  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: {
      scripts: {
        "release.pack.v1": { script_id: "release.pack.v1", role: "release", llm_role: "release", validation: {} },
      },
    },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {},
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "release-run",
      workflow_run_id: "release-wf",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Ship release pack" }),
    },
    stepDef: {
      id: "release_pack",
      role: "release",
      tool: "coding.delegate",
      gate: "finalize",
      prompt_script_id: "release.pack.v1",
    },
    stepIndex: 5,
  });

  assert.equal(payload.execution_lane, "primary_minimax_lane");
  assert.equal(payload.model, "minimax-coding-plan/MiniMax-M2.7");
  assert.equal(payload.model_override, "minimax-coding-plan/MiniMax-M2.7");
});

test("smoke_test payload includes executable command and smoke artifact", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "artifacts/release/run-smoke/handoff/be_to_fe.json", JSON.stringify({
    api_contracts: [{ method: "GET", path: "/api/books" }],
  }, null, 2));

  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: { scripts: {} },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {},
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "run-smoke",
      workflow_run_id: "wf-smoke",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Smoke test app" }),
    },
    stepDef: {
      id: "smoke_test",
      role: "qa",
      tool: "coding.execute",
      gate: null,
    },
    stepIndex: 4,
  });

  assert.match(payload.command, /run_smoke_test\.mjs/);
  assert.match(payload.command, /--artifact-root/);
  assert.match(payload.command, /--api-endpoint "\/api\/books"/);
  assert.equal(payload.max_runtime_s, 120);
  assert.deepEqual(payload.expected_artifacts, ["smoke/smoke_result.json"]);
});

test("impl_be prefers structured workplan json over markdown fallback", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "artifacts/release/run-workplan/plan/workplan.json", JSON.stringify({
    be_tasks: [
      { id: "T-BE-1", description: "Implement auth route", verify: "GET /api/health returns 200" },
    ],
    fe_tasks: [],
  }, null, 2));

  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: {
      scripts: {
        "backend.impl.v1": { script_id: "backend.impl.v1", role: "backend", llm_role: "backend", validation: {} },
      },
    },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {},
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "run-workplan",
      workflow_run_id: "wf-workplan",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Build CRM backend" }),
    },
    stepDef: {
      id: "impl_be",
      role: "backend",
      tool: "coding.delegate",
      gate: "policy",
      prompt_script_id: "backend.impl.v1",
    },
    stepIndex: 2,
  });

  assert.match(payload.task_prompt, /\[Structured Workplan — BE Tasks\]/);
  assert.match(payload.task_prompt, /T-BE-1: Implement auth route/);
  assert.match(payload.task_prompt, /verify: GET \/api\/health returns 200/);
  assert.equal(payload.workplan_validation.ok, true);
});

test("impl_be falls back to markdown when structured workplan json is invalid", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "artifacts/release/run-workplan-bad/plan/workplan.json", JSON.stringify({
    be_tasks: [{ id: "T-BE-1", description: "Missing verify field" }],
    fe_tasks: [],
  }, null, 2));
  writeFile(workspaceRoot, "artifacts/release/run-workplan-bad/plan/workplan.md", "## BE Tasks\n- [ ] T-BE-9: Fallback task | verify: node --check server.js\n");

  const builder = createStepBuilder({
    workspaceRoot,
    registry: { project_types: {}, acceptance_suites: {} },
    promptScriptRegistry: {
      scripts: {
        "backend.impl.v1": { script_id: "backend.impl.v1", role: "backend", llm_role: "backend", validation: {} },
      },
    },
    handoffContracts: { handoffs: {} },
    runtimeConfig: {},
  });

  const payload = builder.buildStepPayload({
    run: {
      run_id: "run-workplan-bad",
      workflow_run_id: "wf-workplan-bad",
      workflow_id: "coding_team_v0",
      project_type: "webapp_crm",
      input_json: JSON.stringify({ goal: "Build CRM backend" }),
    },
    stepDef: {
      id: "impl_be",
      role: "backend",
      tool: "coding.delegate",
      gate: "policy",
      prompt_script_id: "backend.impl.v1",
    },
    stepIndex: 2,
  });

  assert.equal(payload.workplan_validation.ok, false);
  assert.equal(payload.workplan_validation.code, "WORKPLAN_JSON_INVALID");
  assert.match(payload.task_prompt, /\[Workplan JSON Fallback\]/);
  assert.match(payload.task_prompt, /T-BE-9: Fallback task/);
});

