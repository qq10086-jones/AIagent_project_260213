import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { createArtifactPackService } from "../src/domain/workflow_artifact_pack.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-artifact-pack-coding-"));
}

async function main() {
  const workspaceRoot = makeWorkspace();
  const run = {
    workflow_run_id: "wf-coding-evidence",
    run_id: "run-coding-evidence",
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
  };

  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", run.run_id);
  fs.mkdirSync(path.join(releaseRoot, "summary"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "smoke"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "impl", "fe_changes", "public"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "impl", "be_changes", "public"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "release"), { recursive: true });

  fs.writeFileSync(
    path.join(releaseRoot, "impl", "fe_changes", "app.js"),
    "document.querySelector('#cta').addEventListener('click', () => console.log('ok'));\n",
    "utf8"
  );
  fs.writeFileSync(
    path.join(releaseRoot, "impl", "fe_changes", "public", "app.js"),
    "// auto-generated scaffold replace with actual implementation\nexport function placeholderRender() { return 'pending human review'; }\n",
    "utf8"
  );
  fs.writeFileSync(
    path.join(releaseRoot, "impl", "be_changes", "public", "app.js"),
    "// auto-generated scaffold replace with actual implementation\nexport function placeholderRender() { return 'pending human review'; }\n",
    "utf8"
  );
  fs.writeFileSync(
    path.join(releaseRoot, "smoke", "smoke_result.json"),
    JSON.stringify({
      install_ok: true,
      server_started: true,
      root_check: { status: 200, content_type: "text/html", passed: true },
      api_check: { endpoint: "/api/login", status: 200, response_sample: "{\"ok\":true}", passed: true, skipped: false },
      errors: [],
      verdict: "pass",
      evidence_level: "l1_l2",
    }, null, 2),
    "utf8"
  );
  fs.writeFileSync(
    path.join(releaseRoot, "release", "release_notes.md"),
    "# Release Notes\n\nInitial package summary.\n",
    "utf8"
  );

  const testLogRel = `artifacts/runs/${run.run_id}/task_task-1/verification_1.json`;
  const promptContractRel = `artifacts/runs/${run.run_id}/task_task-1/prompt_contract_attempt1.json`;
  const failureLatestRel = `artifacts/runs/${run.run_id}/memory/coding_failure_latest.json`;
  const failureJsonlRel = `artifacts/runs/${run.run_id}/memory/coding_failures.jsonl`;

  for (const rel of [testLogRel, promptContractRel, failureLatestRel, failureJsonlRel]) {
    const abs = path.join(workspaceRoot, rel);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    fs.writeFileSync(abs, rel.endsWith(".jsonl") ? "{}\n" : JSON.stringify({ ok: true }, null, 2), "utf8");
  }

  const steps = [
    {
      step_index: 2,
      step_id: "impl_be",
      role_name: "backend",
      tool_name: "coding.delegate",
      gate_name: "policy",
      task_id: "task-1",
      status: "succeeded",
      checkpoint_id: "cp-1",
      result_json: JSON.stringify({
        ok: true,
        test_result: "passed",
        artifacts: {
          test_log: testLogRel,
          prompt_contract: promptContractRel,
        },
        diagnostics: {
          verification: {
            checked: true,
            ok: true,
            command: "node --check workspace/sandbox/crm_site/server.js",
          },
          retry_summary: {
            attempts_used: 2,
            max_attempts: 3,
            same_error_repeat_limit: 2,
            repairs_attempted: 1,
            repaired_after_retry: true,
          },
          coding_failure_memory: {
            latest_path: failureLatestRel,
            jsonl_path: failureJsonlRel,
          },
          superpowers_plugin: {
            configured: true,
            available: true,
            config_path: "artifacts/config/opencode.json",
            configured_entries: ["/root/.config/opencode/plugins/superpowers.js"],
            detected_paths: ["/root/.config/opencode/plugins/superpowers.js"],
          },
        },
      }),
    },
  ];

  const archivedExtraPaths = [];
  const service = createArtifactPackService({
    pool: {
      async query(sql) {
        const text = String(sql).replace(/\s+/g, " ").trim();
        if (text.startsWith("SELECT checkpoint_id, step_index, step_id, task_id, workspace_hash FROM workflow_checkpoints")) {
          return { rows: [{ checkpoint_id: "cp-1", step_index: 2, step_id: "impl_be", task_id: "task-1", workspace_hash: "hash-1" }] };
        }
        throw new Error(`Unhandled SQL: ${text}`);
      },
    },
    workspaceRoot,
    registry: { project_types: { webapp_crm: { required_artifacts: [] } } },
    archiveReleasePackToMinio: async ({ extraPaths = [] }) => {
      archivedExtraPaths.push(...extraPaths.map((item) => item.replace(/\\/g, "/")));
      return [];
    },
    indexReleasePackToDb: async () => {},
    minioBucket: "test-bucket",
    recordEvent: async () => {},
    getSteps: async () => steps,
  });

  const result = await service.generateArtifactPack(run);
  const manifest = JSON.parse(fs.readFileSync(result.run_manifest_path, "utf8"));
  const summary = fs.readFileSync(result.summary_path, "utf8");
  const releaseNotes = fs.readFileSync(path.join(releaseRoot, "release", "release_notes.md"), "utf8");

  assert.ok(Array.isArray(manifest.coding_execution_evidence));
  assert.equal(manifest.coding_execution_evidence.length, 1);
  assert.equal(manifest.frontend_assembly_repair?.repaired, true);
  assert.deepEqual(manifest.frontend_assembly_repair?.targets_written, [
    "impl/fe_changes/public/app.js",
    "impl/be_changes/public/app.js",
  ]);
  assert.equal(manifest.coding_execution_evidence[0].verification_checked, true);
  assert.equal(manifest.coding_execution_evidence[0].test_log_path, testLogRel);
  assert.equal(manifest.coding_execution_evidence[0].prompt_contract_path, promptContractRel);
  assert.equal(manifest.coding_execution_summary.retry_attempted_steps, 1);
  assert.equal(manifest.coding_execution_summary.failure_memory_entries, 1);
  assert.equal(manifest.runtime_evidence_summary.superpowers_configured_steps, 1);
  assert.equal(manifest.runtime_evidence_summary.superpowers_available_steps, 1);
  assert.equal(manifest.runtime_evidence_summary.superpowers_steps_used, 1);
  assert.equal(manifest.release_notes_runtime_evidence_appended, true);
  assert.equal(manifest.runtime_evidence_summary.smoke_verdict, "pass");
  assert.equal(manifest.runtime_evidence_summary.smoke_root_status, 200);
  assert.equal(manifest.runtime_evidence_summary.smoke_api_status, 200);
  assert.equal(manifest.runtime_evidence.smoke.path, `artifacts/release/${run.run_id}/smoke/smoke_result.json`);
  assert.match(summary, /Runtime Evidence/);
  assert.match(summary, /superpowers_configured_steps: 1/);
  assert.match(summary, /superpowers_steps_used: 1/);
  assert.match(releaseNotes, /## Runtime Evidence/);
  assert.match(releaseNotes, /superpowers_steps_used: 1/);
  assert.match(summary, /smoke_verdict: pass/);
  assert.match(summary, /smoke_root_status: 200/);
  assert.ok(archivedExtraPaths.some((item) => item.endsWith(testLogRel.replace(/\\/g, "/"))));
  assert.ok(archivedExtraPaths.some((item) => item.endsWith(promptContractRel.replace(/\\/g, "/"))));
  assert.ok(archivedExtraPaths.some((item) => item.endsWith(failureLatestRel.replace(/\\/g, "/"))));
  assert.ok(archivedExtraPaths.some((item) => item.endsWith(failureJsonlRel.replace(/\\/g, "/"))));
  assert.match(fs.readFileSync(path.join(releaseRoot, "impl", "fe_changes", "public", "app.js"), "utf8"), /querySelector/);
  assert.match(fs.readFileSync(path.join(releaseRoot, "impl", "be_changes", "public", "app.js"), "utf8"), /querySelector/);

  console.log("workflow_artifact_pack.coding_evidence.test.js: all tests passed");
}

main().catch((err) => {
  console.error("workflow_artifact_pack.coding_evidence.test.js: failed");
  console.error(err);
  process.exit(1);
});
