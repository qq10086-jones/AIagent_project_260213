import fs from "fs";
import os from "os";
import path from "path";

import { createStepBuilder } from "../src/domain/workflow_step_builder.js";
import { createArtifactPackService } from "../src/domain/workflow_artifact_pack.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { getDefaultPromptScriptRegistryPath, loadPromptScriptRegistryOrThrow } from "../src/prompt_script_registry.js";
import { getDefaultHandoffContractPath, loadHandoffContractsOrThrow } from "../src/handoff_contract_registry.js";
import { resolveOrchestratorArtifactPath } from "./_paths.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-canary-m9-"));
}

function writeText(targetPath, text) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, text, "utf8");
}

async function main() {
  const workspaceRoot = makeWorkspace();
  writeText(path.join(workspaceRoot, "sandbox", "crm_site", "server.js"), "export function start() { return 'ok'; }\n");
  writeText(path.join(workspaceRoot, "sandbox", "crm_site", "server.test.js"), "test('server', () => {})\n");

  const registry = loadRegistryOrThrow(getDefaultRegistryPath());
  const promptScriptRegistry = loadPromptScriptRegistryOrThrow(getDefaultPromptScriptRegistryPath());
  const handoffContracts = loadHandoffContractsOrThrow(getDefaultHandoffContractPath());
  const builder = createStepBuilder({
    registry,
    promptScriptRegistry,
    handoffContracts,
    workspaceRoot,
    runtimeConfig: {
      execution: { diff_first_enabled: true },
      worker_coder: {
        max_attempts_default: 2,
        same_error_repeat_limit_default: 2,
      },
    },
  });

  const run = {
    run_id: "run-m9-canary",
    workflow_run_id: "wf-m9-canary",
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    input_json: JSON.stringify({
      goal: "Implement backend CRM API",
      provider: "opencode",
      model: "qwen3-coder-plus-2025-07-22",
    }),
  };
  const stepDef = registry.workflows.coding_team_v0.steps.find((item) => String(item.id) === "impl_be");
  const payload = builder.buildStepPayload({ run, stepDef, stepIndex: 3 });

  assert(payload.verification_command === "node --check workspace/sandbox/crm_site/server.js", "step builder did not infer backend verification_command");
  assert(Number(payload.max_attempts) === 2, "step builder did not set max_attempts");
  assert(Number(payload.same_error_repeat_limit) === 2, "step builder did not set same_error_repeat_limit");
  assert(payload.tool_adapter_request?.payload?.verification_command === payload.verification_command, "tool adapter request dropped verification_command");

  const testLogRel = `artifacts/runs/${run.run_id}/task_task-impl-be/verification_attempt2.json`;
  const failureLatestRel = `artifacts/runs/${run.run_id}/memory/coding_failure_latest.json`;
  const failureJsonlRel = `artifacts/runs/${run.run_id}/memory/coding_failures.jsonl`;
  for (const rel of [testLogRel, failureLatestRel, failureJsonlRel]) {
    const abs = path.join(workspaceRoot, rel);
    writeText(abs, rel.endsWith(".jsonl") ? "{}\n" : JSON.stringify({ ok: true }, null, 2));
  }

  const steps = [
    {
      step_index: 3,
      step_id: "impl_be",
      role_name: "backend",
      tool_name: "coding.delegate",
      gate_name: "policy",
      task_id: "task-impl-be",
      status: "succeeded",
      checkpoint_id: "cp-impl-be",
      result_json: JSON.stringify({
        ok: true,
        test_result: "passed",
        artifacts: { test_log: testLogRel },
        diagnostics: {
          verification: {
            checked: true,
            ok: true,
            command: payload.verification_command,
          },
          retry_summary: {
            attempts_used: 2,
            max_attempts: 2,
            same_error_repeat_limit: 2,
            repairs_attempted: 1,
            repaired_after_retry: true,
          },
          coding_failure_memory: {
            latest_path: failureLatestRel,
            jsonl_path: failureJsonlRel,
          },
        },
      }),
    },
  ];

  const archivedPaths = [];
  const artifactPack = createArtifactPackService({
    pool: {
      async query(sql) {
        const text = String(sql).replace(/\s+/g, " ").trim();
        if (text.startsWith("SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash FROM workflow_checkpoints")) {
          return { rows: [{ checkpoint_id: "cp-impl-be", step_index: 3, step_id: "impl_be", task_id: "task-impl-be", workspace_hash: "hash-1" }] };
        }
        throw new Error(`Unhandled SQL in canary pool: ${text}`);
      },
    },
    workspaceRoot,
    registry,
    archiveReleasePackToMinio: async ({ extraPaths = [] }) => {
      archivedPaths.push(...extraPaths.map((item) => item.replace(/\\/g, "/")));
      return [];
    },
    indexReleasePackToDb: async () => {},
    minioBucket: "canary-bucket",
    recordEvent: async () => {},
    getSteps: async () => steps,
  });

  const pack = await artifactPack.generateArtifactPack(run);
  const manifest = JSON.parse(fs.readFileSync(pack.run_manifest_path, "utf8"));

  assert(Array.isArray(manifest.coding_execution_evidence), "manifest missing coding_execution_evidence");
  assert(manifest.coding_execution_evidence.length === 1, "unexpected coding_execution_evidence length");
  assert(manifest.coding_execution_summary.retry_attempted_steps === 1, "retry_attempted_steps not aggregated");
  assert(manifest.coding_execution_summary.failure_memory_entries === 1, "failure_memory_entries not aggregated");
  assert(archivedPaths.some((item) => item.endsWith(testLogRel)), "test log path not archived");
  assert(archivedPaths.some((item) => item.endsWith(failureLatestRel)), "failure latest path not archived");
  assert(archivedPaths.some((item) => item.endsWith(failureJsonlRel)), "failure jsonl path not archived");

  const outDir = resolveOrchestratorArtifactPath("canary", "m9_coding_guardrails");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "m9_coding_guardrails_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify({
    ok: true,
    generated_at: new Date().toISOString(),
    workspace_root: workspaceRoot.replace(/\\/g, "/"),
    payload_summary: {
      verification_command: payload.verification_command,
      max_attempts: payload.max_attempts,
      same_error_repeat_limit: payload.same_error_repeat_limit,
    },
    manifest_paths: {
      run_manifest_path: pack.run_manifest_path.replace(/\\/g, "/"),
      summary_path: pack.summary_path.replace(/\\/g, "/"),
    },
    coding_execution_summary: manifest.coding_execution_summary,
    archived_paths: archivedPaths,
  }, null, 2), "utf8");

  console.log("# M9 Coding Guardrails Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

main().catch((err) => {
  console.error("canary_m9_coding_guardrails.js failed");
  console.error(err);
  process.exit(1);
});
