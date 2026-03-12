import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { CodingService } from "../coding_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "worker-coder-delegate-scope-"));
}

function writeFile(workspaceRoot, relPath, content) {
  const abs = path.join(workspaceRoot, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function shouldSkipSpawn(result) {
  return String(result?.error || "").includes("spawn EPERM");
}

async function testArtifactOnlyStepAllowsEmptyTargetPaths() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "sandbox/crm_site/app.js", "const value = 1;\n");

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "write pm spec artifact only",
    artifact_root: "orchestrator/artifacts/test/pm_spec_scope",
    expected_artifacts: ["pm_spec.md"],
    step_id: "pm_spec",
    target_paths: [],
    provider: "opencode",
    run_id: "run-pm-spec",
    task_id: "task-pm-spec",
    opencode_command: [process.execPath, "-e", ""],
  });

  if (shouldSkipSpawn(result)) {
    console.log("delegate_scope_policy pm_spec test skipped due sandbox EPERM");
    return;
  }

  assert.equal(result.ok, true, JSON.stringify(result));
  assert.ok(result.artifacts);
  assert.equal(result.diagnostics.error_code, undefined);
  assert.ok(fs.existsSync(path.join(workspaceRoot, "orchestrator", "artifacts", "test", "pm_spec_scope", "pm_spec.md")));
}

async function testImplementationStepStillRequiresTargetPaths() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "sandbox/crm_site/app.js", "const value = 1;\n");

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "update frontend file",
    step_id: "impl_fe",
    target_paths: [],
    provider: "opencode",
    run_id: "run-impl-fe",
    task_id: "task-impl-fe",
    opencode_command: [process.execPath, "-e", ""],
  });

  assert.equal(result.ok, false, JSON.stringify(result));
  assert.equal(result.diagnostics.error_code, "E_UNAUTHORIZED_WRITE");
  assert.match(String(result.error || ""), /target_paths required/);
}

async function main() {
  await testArtifactOnlyStepAllowsEmptyTargetPaths();
  await testImplementationStepStillRequiresTargetPaths();
  console.log("delegate_scope_policy.test.js: all tests passed");
}

main().catch((err) => {
  console.error("delegate_scope_policy.test.js: failed");
  console.error(err);
  process.exit(1);
});
