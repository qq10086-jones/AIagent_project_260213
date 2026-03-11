import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { CodingService } from "../coding_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "worker-coder-verification-"));
}

function writeFile(workspaceRoot, relPath, content) {
  const abs = path.join(workspaceRoot, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function shouldSkipSpawn(result) {
  return String(result?.error || "").includes("spawn EPERM");
}

async function testVerificationCommandPasses() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "sandbox/crm_site/app.js", "export const value = 1;\n");

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "update frontend file",
    step_id: "impl_fe",
    target_paths: ["sandbox/crm_site/"],
    provider: "opencode",
    run_id: "run-pass",
    task_id: "task-pass",
    opencode_command: [
      process.execPath,
      "-e",
      "require('fs').appendFileSync('sandbox/crm_site/app.js', '\\nexport const nextValue = 2;\\n');",
    ],
    verification_command: "node --check sandbox/crm_site/app.js",
  });

  if (shouldSkipSpawn(result)) {
    console.log("verification_command pass test skipped due sandbox EPERM");
    return;
  }

  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.test_result, "passed");
  assert.ok(result.artifacts.test_log);
  assert.equal(result.diagnostics.verification.checked, true);
  assert.equal(result.diagnostics.verification.ok, true);
  assert.equal(result.diagnostics.verification.command, "node --check sandbox/crm_site/app.js");
}

async function testVerificationCommandFails() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "sandbox/crm_site/app.js", "export const value = 1;\n");

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "update frontend file",
    step_id: "impl_fe",
    target_paths: ["sandbox/crm_site/"],
    provider: "opencode",
    run_id: "run-fail",
    task_id: "task-fail",
    opencode_command: [
      process.execPath,
      "-e",
      "require('fs').appendFileSync('sandbox/crm_site/app.js', '\\nexport const nextValue = 2;\\n');",
    ],
    verification_command: "node missing_entry.js",
  });

  if (shouldSkipSpawn(result)) {
    console.log("verification_command fail test skipped due sandbox EPERM");
    return;
  }

  assert.equal(result.ok, false, JSON.stringify(result));
  assert.equal(result.test_result, "failed");
  assert.equal(result.diagnostics.error_code, "E_VERIFICATION_FAILED");
  assert.equal(result.diagnostics.verification.checked, true);
  assert.equal(result.diagnostics.verification.ok, false);
  assert.ok(result.artifacts.test_log);
}

async function testVerificationPlanPasses() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "sandbox/crm_site/app.js", "export const value = 1;\n");

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "update frontend file",
    step_id: "impl_fe",
    target_paths: ["sandbox/crm_site/"],
    provider: "opencode",
    run_id: "run-plan-pass",
    task_id: "task-plan-pass",
    opencode_command: [
      process.execPath,
      "-e",
      "require('fs').appendFileSync('sandbox/crm_site/app.js', '\\nexport const nextValue = 2;\\n');",
    ],
    verification_plan: [
      { tier: "syntax_check", command: "node --check sandbox/crm_site/app.js" },
      { tier: "custom_smoke", command: "node --version" },
    ],
  });

  if (shouldSkipSpawn(result)) {
    console.log("verification_plan pass test skipped due sandbox EPERM");
    return;
  }

  assert.equal(result.ok, true, JSON.stringify(result));
  assert.deepEqual(result.diagnostics.verification.achieved_tiers, ["syntax_check", "custom_smoke"]);
  assert.equal(result.diagnostics.verification.records.length, 2);
}

async function main() {
  await testVerificationCommandPasses();
  await testVerificationCommandFails();
  await testVerificationPlanPasses();
  console.log("verification_command.test.js: all tests passed");
}

main().catch((err) => {
  console.error("verification_command.test.js: failed");
  console.error(err);
  process.exit(1);
});
