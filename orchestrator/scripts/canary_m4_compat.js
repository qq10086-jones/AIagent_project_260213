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
  if (actual !== expected) {
    throw new Error(`${label}: expected='${expected}' actual='${actual}'`);
  }
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeText(p, text) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, text, "utf8");
}

function writeJson(p, value) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, JSON.stringify(value, null, 2), "utf8");
}

function parseResultJson(value) {
  if (value && typeof value === "object") return value;
  try {
    return JSON.parse(value || "{}");
  } catch {
    return {};
  }
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
    return state.workflow_runs.find((r) => r.workflow_run_id === id) || null;
  }

  function findStep(wrid, idx) {
    return state.workflow_steps.find(
      (s) => s.workflow_run_id === wrid && Number(s.step_index) === Number(idx)
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
        const r = findRun(params[0]);
        return { rows: r ? [r] : [] };
      }
      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 ORDER BY step_index ASC") {
        return {
          rows: state.workflow_steps
            .filter((s) => s.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }
      if (text === "SELECT * FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2") {
        const s = findStep(params[0], params[1]);
        return { rows: s ? [s] : [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status=$3, task_id=$4")) {
        const s = findStep(params[0], params[1]);
        if (s) {
          s.status = params[2];
          s.task_id = params[3];
          s.risk_level = params[4];
          s.approval_required = params[5];
          s.approval_reasons_json = params[6];
        }
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET current_step_index=$2, status='running', updated_at=NOW() WHERE workflow_run_id=$1") {
        const r = findRun(params[0]);
        if (r) {
          r.current_step_index = params[1];
          r.status = "running";
        }
        return { rows: [] };
      }
      if (text === "SELECT payload_json FROM tasks WHERE task_id=$1") {
        const t = state.tasks.find((item) => item.task_id === params[0]) || null;
        return { rows: t ? [{ payload_json: t.payload_json }] : [] };
      }
      if (text === "SELECT task_id, run_id, payload_json FROM tasks WHERE task_id=$1") {
        const t = state.tasks.find((item) => item.task_id === params[0]) || null;
        return { rows: t ? [t] : [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='running'")) {
        const s = findStep(params[0], params[1]);
        if (s) s.status = "running";
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='queued'")) {
        const s = findStep(params[0], params[1]);
        if (s) s.status = "queued";
        return { rows: [] };
      }
      if (text === "SELECT gate_name FROM workflow_steps WHERE workflow_run_id=$1 AND step_index=$2 LIMIT 1") {
        const s = findStep(params[0], params[1]);
        return { rows: [{ gate_name: s?.gate_name || "" }] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='failed', result_json=$3, error_code=$4")) {
        const s = findStep(params[0], params[1]);
        if (s) {
          s.status = "failed";
          s.result_json = params[2];
          s.error_code = params[3];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='succeeded', result_json=$3, error_code=NULL")) {
        const s = findStep(params[0], params[1]);
        if (s) {
          s.status = "succeeded";
          s.result_json = params[2];
          s.error_code = null;
          s.checkpoint_id = params[3];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_runs SET status='failed', error_code=$2, error_message=$3")) {
        const r = findRun(params[0]);
        if (r) {
          r.status = "failed";
          r.error_code = params[1];
          r.error_message = params[2];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_steps SET status='failed', error_code=$3")) {
        const s = findStep(params[0], params[1]);
        if (s && s.status !== "succeeded") {
          s.status = "failed";
          s.error_code = params[2];
        }
        return { rows: [] };
      }
      if (text.startsWith("UPDATE workflow_runs SET status='succeeded'")) {
        const r = findRun(params[0]);
        if (r) r.status = "succeeded";
        return { rows: [] };
      }
      if (text === "UPDATE runs SET status='failed' WHERE run_id=$1") return { rows: [] };
      if (text === "UPDATE runs SET status='succeeded' WHERE run_id=$1") return { rows: [] };
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
        const s = findStep(params[0], params[1]);
        if (s) s.checkpoint_id = params[2];
        return { rows: [] };
      }
      if (text === "UPDATE workflow_runs SET last_checkpoint_id=$2, updated_at=NOW() WHERE workflow_run_id=$1") {
        const r = findRun(params[0]);
        if (r) r.last_checkpoint_id = params[1];
        return { rows: [] };
      }
      if (text.startsWith("SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash")) {
        return {
          rows: state.workflow_checkpoints
            .filter((c) => c.workflow_run_id === params[0])
            .sort((a, b) => Number(a.step_index) - Number(b.step_index)),
        };
      }
      if (text.startsWith("INSERT INTO assets") || text.startsWith("UPDATE runs SET")) return { rows: [] };
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
    runtimeConfig: {
      execution: {
        diff_first_enabled: false,
      },
    },
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
  writeText(path.join(rootAbs, "plan", "spec.md"),
    "# Scope\n\n## User Stories\n\nAs a user, I want to log in.\n\n## Acceptance Criteria\n\nLogin works.\n\n## Non-Goals\n\nNo email 2FA.\n\n## Artifact List\n\n- plan/spec.md\n");
  writeText(path.join(rootAbs, "plan", "milestones.md"),
    "scope user_stories acceptance_criteria non_goals artifact_list");
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: ["AC-001: User can log in"],
    artifacts: ["plan/spec.md", "plan/milestones.md"],
    owner: "pm_agent",
    version: "v1",
  });
  writeJson(path.join(rootAbs, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "CRM MVP - auth only",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: { criteria: ["User can log in"] },
  });
}

function writeArchArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "plan", "arch.md"),
    "# Module Breakdown\n\nAuth service.\n\n## Interfaces\n\nSee interfaces.md.\n\n## Dependency Choices\n\nPostgres.\n\n## Risk Notes\n\nAuth migration risk.\n");
  writeText(path.join(rootAbs, "plan", "interfaces.md"),
    "# POST /api/login\n\nRequest: { email, password }\nResponse: { token }\nAuth: none\n");
  writeText(path.join(rootAbs, "plan", "workplan.md"),
    "module breakdown interfaces dependency choices risk notes");
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ level: "high", title: "auth", mitigation: "staged rollout" }],
    decision_log: ["Use postgres for auth storage"],
  });
  writeJson(path.join(rootAbs, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe"],
    modules: ["auth-service"],
    interfaces: ["POST /api/login"],
    decisions: [{ adr_id: "ADR-001", title: "Use Postgres for auth storage", status: "accepted" }],
    risks: ["auth migration"],
  });
}

function writeBeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "be_changes", "server.js"),
    "// stub backend server\nconst express = require('express');\n");
  writeText(path.join(rootAbs, "impl", "be_notes.md"),
    "# Backend Notes\n\n## API Contracts\n\nPOST /api/login implemented.\n\n## Shared Types\n\n- User: { id, email }\n\n## Scope Constraints\n\n- No email verification in this sprint.\n\nRun: node server.js\n");
  writeJson(path.join(rootAbs, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [{ name: "login", method: "POST", path: "/api/login", response_shape: "{ token }" }],
    shared_types: [{ name: "User", description: "{ id, email }" }],
    scope_constraints: ["no email verification"],
  });
}

function writeFeArtifacts(rootAbs) {
  writeText(path.join(rootAbs, "impl", "fe_changes", "app.js"),
    "// stub frontend app\nfetch('/api/login', { method: 'POST', body: JSON.stringify({ email, password }) });\n");
  writeText(path.join(rootAbs, "impl", "fe_notes.md"),
    "# Frontend Notes\n\nConsumed POST /api/login from be_to_fe.json.\nRun: npm start\n");
  writeJson(path.join(rootAbs, "handoff", "impl_to_qa.json"), {
    from_steps: ["impl_be", "impl_fe"],
    to_step: "qa_verify",
    be_changes_path: "impl/be_changes",
    fe_changes_path: "impl/fe_changes",
    run_instructions: "Start backend, start frontend, verify login flow.",
    known_limitations: ["no email verification"],
    api_contracts_path: "handoff/be_to_fe.json",
  });
}

function writeQaArtifacts(rootAbs) {
  writeJson(path.join(rootAbs, "verify", "qa_report.json"), {
    overall_status: "pass",
    checks: [
      { check_id: "D-01", layer: "deterministic", description: "BE artifacts present", status: "pass", detail: "impl/be_changes/ non-empty" },
      { check_id: "D-02", layer: "deterministic", description: "FE artifacts present", status: "pass", detail: "impl/fe_changes/ non-empty" },
      { check_id: "S-01", layer: "semantic", description: "API contracts consistent", status: "pass", detail: "POST /api/login matches FE consumption" },
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
  writeText(path.join(rootAbs, "release", "release_notes.md"),
    `# Release Notes\n\nCRM MVP - auth endpoint delivered and verified.\n\n## What was built\n\n- POST /api/login backend + frontend\n\n## QA Status\n\nAll acceptance criteria (AC-001) verified.\n\nRun ID: ${runId}\n`);
  writeJson(path.join(rootAbs, "release", "artifact_manifest.json"), {
    run_id: runId,
    workflow_id: "coding_team_v0",
    completed_at: new Date().toISOString(),
    artifacts: [
      { path: "plan/spec.md", type: "markdown", size_bytes: 120 },
      { path: "plan/acceptance.json", type: "json", size_bytes: 80 },
      { path: "risk/risk_report.json", type: "json", size_bytes: 90 },
      { path: "verify/qa_report.json", type: "json", size_bytes: 200 },
      { path: "release/release_notes.md", type: "markdown", size_bytes: 180 },
    ],
  });
}

async function advanceStep({ engine, pool, taskIdx, writeArtifacts, workspaceRoot }) {
  const task = pool.state.tasks[taskIdx];
  if (!task) throw new Error(`No task at index ${taskIdx}`);
  const payload = JSON.parse(task.payload_json);
  const artifactRootAbs = path.resolve(workspaceRoot, payload.artifact_root);
  ensureDir(artifactRootAbs);
  writeArtifacts(artifactRootAbs);
  await engine.handleTaskClaimed(task.task_id);
  return engine.handleTaskTerminal({ task_id: task.task_id, status: "succeeded", output: { artifacts: [] } });
}

async function main() {
  const baseDir = resolveOrchestratorArtifactPath("canary", "m4_compat");
  ensureDir(baseDir);
  const workspaceRoot = path.join(baseDir, "happy_path");
  ensureDir(workspaceRoot);

  const harness = createEngineHarness(workspaceRoot);
  const runName = "m4-compat";
  harness.pool.state.runs.push({ run_id: `${runName}-run`, status: "running" });

  const started = await harness.engine.startWorkflowRun({
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    run_id: `${runName}-run`,
    input: { goal: "Build CRM webapp with auth", provider: "opencode", model: "qwen-max" },
  });

  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIdx: 0, writeArtifacts: writePmArtifacts, workspaceRoot });
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIdx: 1, writeArtifacts: writeArchArtifacts, workspaceRoot });

  const beTaskPayload = JSON.parse(harness.pool.state.tasks[2].payload_json);
  assertEqual(beTaskPayload.prompt_script_id, "backend.impl.v1", "impl_be prompt_script_id");
  assertEqual(beTaskPayload.execution_mode_requested, "full_file_fallback", "impl_be execution_mode_requested");
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIdx: 2, writeArtifacts: writeBeArtifacts, workspaceRoot });

  const feTaskPayload = JSON.parse(harness.pool.state.tasks[3].payload_json);
  assertEqual(feTaskPayload.prompt_script_id, "frontend.impl.v1", "impl_fe prompt_script_id");
  assertEqual(feTaskPayload.execution_mode_requested, "full_file_fallback", "impl_fe execution_mode_requested");
  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIdx: 3, writeArtifacts: writeFeArtifacts, workspaceRoot });

  await advanceStep({ engine: harness.engine, pool: harness.pool, taskIdx: 4, writeArtifacts: writeQaArtifacts, workspaceRoot });
  await advanceStep({
    engine: harness.engine,
    pool: harness.pool,
    taskIdx: 5,
    writeArtifacts: (rootAbs) => writeReleaseArtifacts(rootAbs, `${runName}-run`),
    workspaceRoot,
  });

  const workflowRun = harness.pool.state.workflow_runs.find((r) => r.workflow_run_id === started.workflow_run_id);
  assertEqual(workflowRun.status, "succeeded", "workflow status");

  const steps = harness.pool.state.workflow_steps
    .filter((s) => s.workflow_run_id === started.workflow_run_id)
    .sort((a, b) => Number(a.step_index) - Number(b.step_index));
  assertEqual(steps.length, 6, "workflow step count");
  for (const step of steps) {
    assertEqual(step.status, "succeeded", `step ${step.step_id} status`);
  }

  const beResult = parseResultJson(steps.find((step) => step.step_id === "impl_be")?.result_json);
  const feResult = parseResultJson(steps.find((step) => step.step_id === "impl_fe")?.result_json);
  const beMode = String(beResult.execution_mode_used || beResult?.impl_validation?.execution_mode_used || "");
  const feMode = String(feResult.execution_mode_used || feResult?.impl_validation?.execution_mode_used || "");
  assertEqual(beMode, "full_file_fallback", "impl_be execution_mode_used");
  assertEqual(feMode, "full_file_fallback", "impl_fe execution_mode_used");

  const releaseRoot = path.resolve(workspaceRoot, "artifacts", "release", `${runName}-run`);
  assert(fs.existsSync(path.join(releaseRoot, "impl", "be_changes", "server.js")), "impl/be_changes/server.js exists");
  assert(fs.existsSync(path.join(releaseRoot, "impl", "fe_changes", "app.js")), "impl/fe_changes/app.js exists");
  assert(!fs.existsSync(path.join(releaseRoot, "impl", "be_patch_bundle.json")), "be patch bundle absent");
  assert(!fs.existsSync(path.join(releaseRoot, "impl", "fe_patch_bundle.json")), "fe patch bundle absent");
  assert(fs.existsSync(path.join(releaseRoot, "handoff", "be_to_fe.json")), "be_to_fe handoff exists");
  assert(fs.existsSync(path.join(releaseRoot, "handoff", "impl_to_qa.json")), "impl_to_qa handoff exists");
  assert(fs.existsSync(path.join(releaseRoot, "handoff", "qa_to_release.json")), "qa_to_release handoff exists");
  assert(fs.existsSync(path.join(releaseRoot, "release", "release_notes.md")), "release notes exist");
  assert(fs.existsSync(path.join(releaseRoot, "release", "artifact_manifest.json")), "artifact manifest exists");

  const report = {
    ok: true,
    generated_at: new Date().toISOString(),
    workflow_run_id: started.workflow_run_id,
    run_id: `${runName}-run`,
    checks: [
      { id: "full_sequential_workflow", ok: true, steps: 6, workflow_status: workflowRun.status },
      { id: "impl_be_full_file_mode", ok: true, prompt_script_id: beTaskPayload.prompt_script_id, execution_mode_used: beMode },
      { id: "impl_fe_full_file_mode", ok: true, prompt_script_id: feTaskPayload.prompt_script_id, execution_mode_used: feMode },
      { id: "m4_handoff_paths_present", ok: true },
      { id: "release_pack_structure_preserved", ok: true },
    ],
    release_root: releaseRoot.replace(/\\/g, "/"),
  };

  const reportPath = path.join(baseDir, "m4_compat_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log("# M4 Compatibility Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main().catch((err) => {
  console.error("[canary_m4_compat] FAILED:", err.message || err);
  process.exit(1);
});
