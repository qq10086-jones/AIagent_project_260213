/**
 * canary_coding_team_e2e.js
 *
 * v1.1 coding-team workflow canary:
 * PM -> Arch -> BE -> FE -> Smoke -> QA -> Release -> Deploy Preview
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";
import { createWorkflowEngine } from "../src/workflow_engine.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { loadPromptScriptRegistryOrThrow, getDefaultPromptScriptRegistryPath } from "../src/prompt_script_registry.js";
import { loadHandoffContractsOrThrow, getDefaultHandoffContractPath } from "../src/handoff_contract_registry.js";
import { analyzeTaskRisk } from "../src/policy.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function assert(condition, label) {
  if (!condition) throw new Error(`ASSERTION FAILED: ${label}`);
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) throw new Error(`${label}: expected='${expected}' actual='${actual}'`);
}

function ensureDir(targetPath) {
  fs.mkdirSync(targetPath, { recursive: true });
}

function writeText(targetPath, text) {
  ensureDir(path.dirname(targetPath));
  fs.writeFileSync(targetPath, text, "utf8");
}

function writeJson(targetPath, value) {
  ensureDir(path.dirname(targetPath));
  fs.writeFileSync(targetPath, JSON.stringify(value, null, 2), "utf8");
}

function createMemoryPool() {
  const state = {
    workflow_runs: [],
    workflow_steps: [],
    tasks: [],
    runs: [],
    workflow_checkpoints: [],
  };

  function findRun(id) {
    return state.workflow_runs.find((item) => item.workflow_run_id === id) || null;
  }

  function findStep(workflowRunId, stepIndex) {
    return state.workflow_steps.find(
      (item) => item.workflow_run_id === workflowRunId && Number(item.step_index) === Number(stepIndex)
    ) || null;
  }

  return {
    state,
    async query(sql, params = []) {
      const text = String(sql).replace(/\s+/g, " ").trim();

      if (text.startsWith("INSERT INTO workflow_runs")) {
        const [workflow_run_id, run_id, workflow_id, project_type, input_json] = params;
        state.workflow_runs.push({
          workflow_run_id,
          run_id,
          workflow_id,
          project_type,
          status: "running",
          current_step_index: 0,
          input_json,
          error_code: null,
          error_message: null,
          last_checkpoint_id: null,
        });
        return { rows: [] };
      }

      if (text.startsWith("INSERT INTO workflow_steps")) {
        const [workflow_run_id, step_index, step_id, role_name, tool_name, gate_name] = params;
        state.workflow_steps.push({
          workflow_run_id,
          step_index,
          step_id,
          role_name,
          tool_name,
          gate_name,
          status: "pending",
          task_id: null,
          risk_level: null,
          approval_required: false,
          approval_reasons_json: "[]",
          result_json: null,
          error_code: null,
          checkpoint_id: null,
        });
        return { rows: [] };
      }

      if (text === "SELECT * FROM workflow_runs WHERE workflow_run_id=$1") {
        const row = findRun(params[0]);
        return { rows: row ? [row] : [] };
      }

      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC") {
        return {
          rows: state.workflow_steps
            .filter((item) => item.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }

      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2") {
        const row = findStep(params[0], params[1]);
        return { rows: row ? [row] : [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status=$3, task_id=$4")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = params[2];
          step.task_id = params[3];
          step.risk_level = params[4];
          step.approval_required = params[5];
          step.approval_reasons_json = params[6];
        }
        return { rows: [] };
      }

      if (text === "UPDATE workflow_runs SET current_step_index=$2, status='running', updated_at=NOW() WHERE workflow_run_id=$1") {
        const run = findRun(params[0]);
        if (run) {
          run.current_step_index = params[1];
          run.status = "running";
        }
        return { rows: [] };
      }

      if (text === "UPDATE workflow_runs SET status='partial_failure', error_code=$2, error_message=$3, updated_at=NOW() WHERE workflow_run_id=$1") {
        const run = findRun(params[0]);
        if (run) {
          run.status = "partial_failure";
          run.error_code = params[1];
          run.error_message = params[2];
        }
        return { rows: [] };
      }

      if (text === "SELECT payload_json FROM tasks WHERE task_id=$1") {
        const task = state.tasks.find((item) => item.task_id === params[0]) || null;
        return { rows: task ? [{ payload_json: task.payload_json }] : [] };
      }

      if (text === "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1") {
        const task = state.tasks.find((item) => item.task_id === params[0]) || null;
        return { rows: task ? [task] : [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='running'")) {
        const step = findStep(params[0], params[1]);
        if (step) step.status = "running";
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='queued'")) {
        const step = findStep(params[0], params[1]);
        if (step) step.status = "queued";
        return { rows: [] };
      }

      if (text === "SELECT gate_name FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2 LIMIT 1") {
        const step = findStep(params[0], params[1]);
        return { rows: [{ gate_name: step?.gate_name || "" }] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='failed', result_json=$3, error_code=$4")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "failed";
          step.result_json = params[2];
          step.error_code = params[3];
        }
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='succeeded', result_json=$3, error_code=NULL")) {
        const step = findStep(params[0], params[1]);
        if (step) {
          step.status = "succeeded";
          step.result_json = params[2];
          step.error_code = null;
          step.checkpoint_id = params[3];
        }
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_runs SET status='failed', error_code=$2, error_message=$3")) {
        const run = findRun(params[0]);
        if (run) {
          run.status = "failed";
          run.error_code = params[1];
          run.error_message = params[2];
        }
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET status='failed', error_code=$3")) {
        const step = findStep(params[0], params[1]);
        if (step && step.status !== "succeeded") {
          step.status = "failed";
          step.error_code = params[2];
        }
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_runs SET status='succeeded'")) {
        const run = findRun(params[0]);
        if (run) run.status = "succeeded";
        return { rows: [] };
      }

      if (text === "UPDATE runs SET status='failed' WHERE run_id=$1") return { rows: [] };
      if (text === "UPDATE runs SET status='succeeded' WHERE run_id=$1") return { rows: [] };
      if (text === "UPDATE runs SET status=$1 WHERE run_id=$2") return { rows: [] };

      if (text.startsWith("INSERT INTO workflow_checkpoints")) {
        const [checkpoint_id, workflow_run_id, step_index, step_id, task_id, workspace_hash, artifact_refs_json, checkpoint_json] = params;
        state.workflow_checkpoints.push({
          checkpoint_id,
          workflow_run_id,
          step_index,
          step_id,
          task_id,
          workspace_hash,
          artifact_refs_json,
          checkpoint_json,
        });
        return { rows: [] };
      }

      if (text.startsWith("UPDATE workflow_steps SET checkpoint_id=$3")) {
        const step = findStep(params[0], params[1]);
        if (step) step.checkpoint_id = params[2];
        return { rows: [] };
      }

      if (text === "UPDATE workflow_runs SET last_checkpoint_id=$2, updated_at=NOW() WHERE workflow_run_id=$1") {
        const run = findRun(params[0]);
        if (run) run.last_checkpoint_id = params[1];
        return { rows: [] };
      }

      if (text.startsWith("SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash")) {
        return {
          rows: state.workflow_checkpoints
            .filter((item) => item.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }

      if (
        text.startsWith("INSERT INTO assets") ||
        text.startsWith("INSERT INTO waterfall_stage_log") ||
        text.startsWith("INSERT INTO routing_decision_log") ||
        text.startsWith("UPDATE runs SET")
      ) {
        return { rows: [] };
      }

      throw new Error(`Unhandled SQL in memory pool: ${text}`);
    },
  };
}

function createEngineHarness(workspaceRoot) {
  const pool = createMemoryPool();
  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const promptScriptRegistry = loadPromptScriptRegistryOrThrow(getDefaultPromptScriptRegistryPath());
  const handoffContracts = loadHandoffContractsOrThrow(getDefaultHandoffContractPath());
  const events = [];
  let nextTaskId = 0;

  const engine = createWorkflowEngine({
    pool,
    registry,
    promptScriptRegistry,
    handoffContracts,
    workspaceRoot,
    strictStepArtifacts: false,
    auditStepArtifacts: true,
    enqueueTask: async ({ payload, run_id, tool_name }) => {
      const task_id = `task-${++nextTaskId}`;
      const risk = analyzeTaskRisk(tool_name, payload || {});
      pool.state.tasks.push({ task_id, run_id, tool_name, payload_json: JSON.stringify(payload) });
      return { task_id, waiting_approval: Boolean(risk.requires_approval) };
    },
    recordEvent: async (stream_id, event_name, payload) => {
      events.push({ stream_id, event_name, payload });
    },
    makeIdempotencyKey: (run_id, tool_name, payload) =>
      crypto.createHash("sha256").update(JSON.stringify({ run_id, tool_name, payload })).digest("hex"),
  });

  return { engine, pool, events };
}

function writePmArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "spec.md"), "# Scope\n\n## User Stories\n\nAs a user, I want to log in.\n\n## Acceptance Criteria\n\nLogin works.\n\n## Non-Goals\n\nNo email 2FA.\n\n## Artifact List\n\n- plan/spec.md\n");
  writeText(path.join(rootAbs, "plan", "milestones.md"), "scope user_stories acceptance_criteria non_goals artifact_list");
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: ["AC-001: User can log in"],
    artifacts: ["plan/spec.md", "plan/milestones.md"],
    owner: "pm_agent",
    version: "v1",
  });
  writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM MVP auth only",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: { criteria: ["User can log in"] },
  });
}

function writeArchArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "arch.md"), "# Module Breakdown\n\nAuth service.\n\n## Interfaces\n\nSee interfaces.md.\n\n## Dependency Choices\n\nPostgres.\n\n## Risk Notes\n\nAuth migration risk.\n");
  writeText(path.join(rootAbs, "plan", "interfaces.md"), "# GET /api/login\n\nRequest: none\nResponse: { ok, token }\nAuth: none\n");
  writeText(path.join(rootAbs, "plan", "workplan.md"), "module breakdown interfaces dependency choices risk notes");
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "high", title: "auth", mitigation: "staged rollout" }],
    decision_log: ["Use postgres for auth storage"],
  });
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["auth-service"],
    interfaces: ["GET /api/login"],
    decisions: [{ adr_id: "ADR-001", title: "Use Postgres for auth storage", status: "accepted" }],
    risks: ["auth migration"],
  });
}

function writeBeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "be_changes", "server.js"), "const express = require('express');\nconst path = require('path');\nconst app = express();\napp.use(express.json());\napp.use(express.static(path.join(__dirname, 'public')));\napp.get('/api/login', (_req, res) => res.json({ ok: true, token: 'demo-token' }));\napp.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));\nconst port = Number(process.env.PORT || 3000);\napp.listen(port, () => console.log(`listening:${port}`));\n");
  writeJson(path.join(rootAbs, "impl", "be_changes", "package.json"), {
    name: "crm-e2e-demo",
    version: "1.0.0",
    main: "server.js",
    dependencies: { express: "^4.19.2" },
  });
  writeText(path.join(rootAbs, "impl", "be_notes.md"), "# Backend Notes\n\n## API Contracts\n\nGET /api/login implemented.\n\n## Shared Types\n\n- User: { id, email }\n\n## Scope Constraints\n\n- No email verification in this sprint.\n\nRun: node server.js\n");
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "login", method: "GET", path: "/api/login", response_shape: "{ ok, token }" }],
    shared_types: [{ name: "User", description: "{ id, email }" }],
    scope_constraints: ["no email verification"],
  });
}

function writeFeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "fe_changes", "public", "index.html"), "<!doctype html>\n<html><body><main><h1>CRM MVP</h1><p id=\"status\">booting</p></main><script type=\"module\" src=\"./app.js\"></script></body></html>\n");
  writeText(path.join(rootAbs, "impl", "fe_changes", "public", "app.js"), "async function boot() {\n  const status = document.getElementById('status');\n  const response = await fetch('/api/login');\n  const data = await response.json();\n  status.textContent = data.ok ? 'ready' : 'failed';\n}\nboot();\n");
  writeText(path.join(rootAbs, "impl", "fe_changes", "public", "styles.css"), "body { font-family: sans-serif; }\n");
  writeText(path.join(rootAbs, "impl", "fe_notes.md"), "# Frontend Notes\n\nConsumed GET /api/login from be_to_fe.json.\nRun: npm start\n");
}

function writeSmokeArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "smoke", "smoke_result.json"), {
    install_ok: true,
    server_started: true,
    root_check: { status: 200, content_type: "text/html", passed: true },
    api_check: {
      endpoint: "/api/login",
      status: 200,
      response_sample: "{\"ok\":true,\"token\":\"demo-token\"}",
      passed: true,
      skipped: false,
    },
    errors: [],
    verdict: "pass",
    evidence_level: "l1_l2",
  });
}

function writeQaArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [
      { check_id: "D-01", layer: "deterministic", description: "BE artifacts present", status: "pass", detail: "impl/be_changes/server.js and package.json present" },
      { check_id: "D-02", layer: "deterministic", description: "FE artifacts present", status: "pass", detail: "impl/fe_changes/public/index.html and app.js present" },
      { check_id: "D-03", layer: "deterministic", description: "Smoke result captured", status: "pass", detail: "smoke/smoke_result.json root_check.status=200 api_check.status=200" },
      { check_id: "S-01", layer: "semantic", description: "API contracts consistent", status: "pass", detail: "GET /api/login matches FE consumption" },
    ],
    journey_checks: [
      {
        journey_id: "J-001",
        description: "User can open the app and hit the login endpoint",
        status: "pass",
        evidence: [
          "smoke/smoke_result.json root_check.status=200",
          "smoke/smoke_result.json api_check.endpoint=/api/login status=200",
        ],
      },
    ],
    rubric_citations: [
      {
        term: "demo_usable",
        criterion: "service boots and main API responds",
        evidence: "smoke/smoke_result.json verdict=pass evidence_level=l1_l2",
        pass: true,
      },
    ],
    verified_artifacts: ["A1"],
    generated_at: new Date().toISOString(),
    step_id: "qa_verify",
  });
  writeJson(path.join(rootAbs, "handoff", "qa_to_release.json"), {
    from_step: "qa_verify",
    to_step: "release_pack",
    qa_report_path: "verify/qa_report.json",
    overall_status: "pass",
    verified_artifacts: ["A1"],
    run_id: path.basename(rootAbs),
    workflow_run_id: path.basename(rootAbs),
  });
}

function writeReleaseArtifacts(rootAbs, runId) {
  writeText(path.join(rootAbs, "release", "release_notes.md"), `# Release Notes\n\nCRM MVP auth endpoint delivered and verified.\n\n## What was built\n\n- GET /api/login backend + frontend\n\n## QA Status\n\nAll acceptance criteria (AC-001) verified.\n\nRun ID: ${runId}\n`);
  writeText(path.join(rootAbs, "release", "README.md"), "# Run Instructions\n\n1. `cd impl/be_changes`\n2. `npm install`\n3. `node server.js`\n\nOpen `http://localhost:3000/` after the server starts.\n");
  writeText(path.join(rootAbs, "release", "start.sh"), "#!/usr/bin/env sh\nset -eu\ncd \"$(dirname \"$0\")/../impl/be_changes\"\nnpm install\nnode server.js\n");
  writeJson(path.join(rootAbs, "release", "artifact_manifest.json"), {
    run_id: runId,
    workflow_id: "coding_team_v0",
    completed_at: new Date().toISOString(),
    artifacts: [
      { path: "plan/spec.md", type: "markdown", size_bytes: 120 },
      { path: "plan/acceptance.json", type: "json", size_bytes: 80 },
      { path: "risk/risk_report.json", type: "json", size_bytes: 90 },
      { path: "smoke/smoke_result.json", type: "json", size_bytes: 220 },
      { path: "verify/qa_report.json", type: "json", size_bytes: 200 },
      { path: "release/release_notes.md", type: "markdown", size_bytes: 180 },
      { path: "release/README.md", type: "markdown", size_bytes: 220 },
      { path: "release/start.sh", type: "text/x-shellscript", size_bytes: 96 },
    ],
  });
}

function writeDeployArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "preview", "deployment_result.json"), {
    preview_url: "",
    fallback_reason: "preview_disabled_in_test",
    deployed_at: new Date().toISOString(),
  });
}

function buildSuperpowersDiagnostics() {
  return {
    superpowers_plugin: {
      configured: true,
      available: true,
      config_path: "/root/.config/opencode/opencode.json",
      configured_entries: ["/root/.config/opencode/plugins/superpowers.js"],
      detected_paths: ["/root/.config/opencode/plugins/superpowers.js"],
    },
  };
}

async function advanceStep({ engine, pool, taskIndex, writeArtifacts, workspaceRoot, buildOutput = null }) {
  const task = pool.state.tasks[taskIndex];
  if (!task) throw new Error(`No task at index ${taskIndex}`);
  const payload = JSON.parse(task.payload_json);
  const artifactRootAbs = path.resolve(workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  writeArtifacts(artifactRootAbs);
  await engine.handleTaskClaimed(task.task_id);
  const output = typeof buildOutput === "function" ? buildOutput({ payload, task }) : { artifacts: [] };
  return engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output });
}

async function runHappyPath({ workspaceRoot, runName }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${runName}-run`, status: "running" });
  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${runName}-run`,
    input: { goal: "Build CRM webapp with auth", provider: "opencode", model: "qwen-max" },
  });

  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 0, writeArtifacts: writePmArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 1, writeArtifacts: writeArchArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 2, writeArtifacts: writeBeArtifacts, workspaceRoot, buildOutput: () => ({ artifacts: [], diagnostics: buildSuperpowersDiagnostics() }) });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 3, writeArtifacts: writeFeArtifacts, workspaceRoot, buildOutput: () => ({ artifacts: [], diagnostics: buildSuperpowersDiagnostics() }) });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 4, writeArtifacts: writeSmokeArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 5, writeArtifacts: writeQaArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 6, writeArtifacts: (rootAbs) => writeReleaseArtifacts(rootAbs, `${runName}-run`), workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 7, writeArtifacts: writeDeployArtifacts, workspaceRoot });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  assertEqual(workflowRun.status, "succeeded", "happy_path: workflow status");
  const steps = harness.pool.state.workflow_steps.filter((item) => item.workflow_run_id === started.workflow_run_id).sort((a, b) => Number(a.step_index) - Number(b.step_index));
  assertEqual(steps.length, 8, "happy_path: step count");
  for (const step of steps) assertEqual(step.status, "succeeded", `happy_path: step ${step.step_id}`);
  const checkpointCount = harness.pool.state.workflow_checkpoints.filter((item) => item.workflow_run_id === started.workflow_run_id).length;
  assertEqual(checkpointCount, 8, "happy_path: checkpoint count");

  const releaseRoot = path.resolve(workspaceRoot, "artifacts", "release", `${runName}-run`);
  assert(fs.existsSync(path.join(releaseRoot, "smoke", "smoke_result.json")), "happy_path: smoke_result.json");
  assert(fs.existsSync(path.join(releaseRoot, "release", "README.md")), "happy_path: README.md");
  assert(fs.existsSync(path.join(releaseRoot, "release", "start.sh")), "happy_path: start.sh");
  assert(fs.existsSync(path.join(releaseRoot, "preview", "deployment_result.json")), "happy_path: deployment_result.json");
  const manifest = JSON.parse(fs.readFileSync(path.join(releaseRoot, "meta", "run_manifest.json"), "utf8"));
  assertEqual(manifest?.runtime_evidence_summary?.smoke_verdict, "pass", "happy_path: smoke verdict summary");
  assertEqual(Number(manifest?.runtime_evidence_summary?.smoke_root_status || 0), 200, "happy_path: smoke root status summary");
  assertEqual(Number(manifest?.runtime_evidence_summary?.smoke_api_status || 0), 200, "happy_path: smoke api status summary");
  assertEqual(Number(manifest?.runtime_evidence_summary?.superpowers_configured_steps || 0), 2, "happy_path: superpowers configured steps summary");
  assertEqual(Number(manifest?.runtime_evidence_summary?.superpowers_available_steps || 0), 2, "happy_path: superpowers available steps summary");
  assert(harness.events.find((item) => item.event_name === "workflow.succeeded"), "happy_path: workflow.succeeded emitted");

  return {
    workflow_run_id: started.workflow_run_id,
    steps_succeeded: steps.filter((item) => item.status === "succeeded").length,
    checkpoints: checkpointCount,
    release_root: releaseRoot,
  };
}

async function runBeFailureInjection({ workspaceRoot, runName }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${runName}-run`, status: "running" });
  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${runName}-run`,
    input: { goal: "Build CRM webapp with auth", provider: "opencode", model: "qwen-max" },
  });

  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 0, writeArtifacts: writePmArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 1, writeArtifacts: writeArchArtifacts, workspaceRoot });

  const beTask = harness.pool.state.tasks[2];
  if (!beTask) throw new Error("be_failure: impl_be task not dispatched");
  const payload = JSON.parse(beTask.payload_json);
  const artifactRootAbs = path.resolve(workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  writeText(path.join(artifactRootAbs, "impl", "be_notes.md"), "# Incomplete backend notes\n");
  await harness.engine.handleTaskClaimed(beTask.task_id);
  await harness.engine.handleTaskTerminal({ task_id: beTask.task_id, status: "succeeded", output: { artifacts: [] } });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  assertEqual(workflowRun.status, "failed", "be_failure: workflow status");
  const steps = harness.pool.state.workflow_steps.filter((item) => item.workflow_run_id === started.workflow_run_id).sort((a, b) => Number(a.step_index) - Number(b.step_index));
  assertEqual(steps[2].status, "failed", "be_failure: impl_be failed");
  assertEqual(steps[2].error_code, "STEP_IMPL_BE_ARTIFACTS_MISSING", "be_failure: error code");
  assertEqual(harness.pool.state.tasks.length, 3, "be_failure: downstream not dispatched");

  const releaseRoot = path.resolve(workspaceRoot, "artifacts", "release", `${runName}-run`);
  assert(fs.existsSync(path.join(releaseRoot, "plan", "spec.md")), "be_failure: pm spec preserved");
  assert(fs.existsSync(path.join(releaseRoot, "plan", "arch.md")), "be_failure: arch doc preserved");
  assert(harness.events.find((item) => item.event_name === "workflow.failed"), "be_failure: workflow.failed emitted");

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    be_error_code: steps[2].error_code,
    tasks_dispatched: harness.pool.state.tasks.length,
  };
}

async function runQaFailureInjection({ workspaceRoot, runName }) {
  const harness = createEngineHarness(workspaceRoot);
  harness.pool.state.runs.push({ run_id: `${runName}-run`, status: "running" });
  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${runName}-run`,
    input: { goal: "Build CRM webapp with auth", provider: "opencode", model: "qwen-max" },
  });

  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 0, writeArtifacts: writePmArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 1, writeArtifacts: writeArchArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 2, writeArtifacts: writeBeArtifacts, workspaceRoot, buildOutput: () => ({ artifacts: [], diagnostics: buildSuperpowersDiagnostics() }) });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 3, writeArtifacts: writeFeArtifacts, workspaceRoot, buildOutput: () => ({ artifacts: [], diagnostics: buildSuperpowersDiagnostics() }) });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIndex: 4, writeArtifacts: writeSmokeArtifacts, workspaceRoot });

  const qaTask = harness.pool.state.tasks[5];
  if (!qaTask) throw new Error("qa_failure: qa_verify task not dispatched");
  await harness.engine.handleTaskClaimed(qaTask.task_id);
  await harness.engine.handleTaskTerminal({
    task_id: qaTask.task_id,
    status: "failed",
    output: {
      ok: false,
      error: "QA verification failed: login flow test returned exit code 1",
      diagnostics: { error_code: "E_VERIFICATION_FAILED" },
    },
  });

  const workflowRun = harness.pool.state.workflow_runs.find((item) => item.workflow_run_id === started.workflow_run_id);
  assertEqual(workflowRun.status, "failed", "qa_failure: workflow status");
  const steps = harness.pool.state.workflow_steps.filter((item) => item.workflow_run_id === started.workflow_run_id).sort((a, b) => Number(a.step_index) - Number(b.step_index));
  assertEqual(steps[4].status, "succeeded", "qa_failure: smoke_test succeeded");
  assertEqual(steps[5].status, "failed", "qa_failure: qa_verify failed");
  assertEqual(harness.pool.state.tasks.length, 6, "qa_failure: release/deploy not dispatched");
  assert(harness.events.find((item) => item.event_name === "workflow.failed"), "qa_failure: workflow.failed emitted");

  return {
    workflow_run_id: started.workflow_run_id,
    workflow_status: workflowRun.status,
    qa_status: steps[5].status,
    tasks_dispatched: harness.pool.state.tasks.length,
  };
}

async function main() {
  const baseDir = resolveOrchestratorArtifactPath("canary", "coding_team_e2e");
  ensureDir(baseDir);

  console.log("# Coding Team E2E Canary v1.1");

  const happyWorkspace = path.join(baseDir, "happy_path");
  ensureDir(happyWorkspace);
  const happyResult = await runHappyPath({ workspaceRoot: happyWorkspace, runName: "e2e-happy" });
  console.log(`- happy_path: all 8 steps succeeded, workflow=succeeded`);
  console.log(`  checkpoints=${happyResult.checkpoints} release_root=${happyResult.release_root}`);

  const failWorkspace = path.join(baseDir, "be_failure");
  ensureDir(failWorkspace);
  const failResult = await runBeFailureInjection({ workspaceRoot: failWorkspace, runName: "e2e-be-fail" });
  console.log(`- be_failure: workflow=${failResult.workflow_status}, be_error=${failResult.be_error_code}, tasks_dispatched=${failResult.tasks_dispatched}`);

  const qaFailWorkspace = path.join(baseDir, "qa_failure");
  ensureDir(qaFailWorkspace);
  const qaFailResult = await runQaFailureInjection({ workspaceRoot: qaFailWorkspace, runName: "e2e-qa-fail" });
  console.log(`- qa_failure: workflow=${qaFailResult.workflow_status}, qa_step=${qaFailResult.qa_status}, tasks_dispatched=${qaFailResult.tasks_dispatched}`);

  const report = {
    ok: true,
    generated_at: new Date().toISOString(),
    cases: {
      happy_path: { verdict: "pass", steps_succeeded: happyResult.steps_succeeded, checkpoints: happyResult.checkpoints },
      be_failure: { verdict: "pass", workflow_status: failResult.workflow_status, be_error_code: failResult.be_error_code, tasks_dispatched: failResult.tasks_dispatched },
      qa_failure: { verdict: "pass", workflow_status: qaFailResult.workflow_status, qa_status: qaFailResult.qa_status, tasks_dispatched: qaFailResult.tasks_dispatched },
    },
  };

  const reportPath = path.join(baseDir, "coding_team_e2e_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`- report: ${reportPath}`);
  console.log("exit: 0");
}

main().catch((err) => {
  console.error("[canary_coding_team_e2e] FAILED:", err.message || err);
  process.exit(1);
});



