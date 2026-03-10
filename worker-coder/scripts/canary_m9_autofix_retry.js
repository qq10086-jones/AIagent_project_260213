import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { CodingService } from "../coding_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "worker-coder-autofix-"));
}

function writeFile(workspaceRoot, relPath, content) {
  const abs = path.join(workspaceRoot, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function writeReport(payload) {
  const reportDir = path.resolve(process.cwd(), "artifacts", "canary", "m9_autofix_retry");
  fs.mkdirSync(reportDir, { recursive: true });
  const reportPath = path.join(reportDir, "m9_autofix_retry_canary.json");
  fs.writeFileSync(reportPath, JSON.stringify(payload, null, 2), "utf8");
  console.log("# M9 Auto-Fix Retry Canary");
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
}

async function main() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "sandbox/crm_site/app.js", "const status = 'initial';\nmodule.exports = { status };\n");

  const result = await CodingService.delegateTask({
    workspaceRoot,
    task_prompt: "implement frontend update with verification",
    step_id: "impl_fe",
    target_paths: ["sandbox/crm_site/"],
    provider: "opencode",
    run_id: "run-autofix",
    task_id: "task-autofix",
    max_attempts: 2,
    same_error_repeat_limit: 2,
    wall_clock_timeout_s: 300,
    opencode_command: ["mock-inline-autofix", "sandbox/crm_site/app.js", "{{task_prompt}}"],
    verification_command: "node --check sandbox/crm_site/app.js",
  });

  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.test_result, "passed");
  assert.equal(result.diagnostics.retry_summary.attempts_used, 2);
  assert.equal(result.diagnostics.retry_summary.repairs_attempted, 1);
  assert.equal(result.diagnostics.retry_summary.repaired_after_retry, true);
  assert.equal(result.diagnostics.verification.checked, true);
  assert.equal(result.diagnostics.verification.ok, true);

  const finalContent = fs.readFileSync(path.join(workspaceRoot, "sandbox/crm_site/app.js"), "utf8");
  assert.match(finalContent, /fixed/);

  writeReport({
    ok: true,
    generated_at: new Date().toISOString(),
    workspace_root: workspaceRoot.replace(/\\/g, "/"),
    retry_summary: result.diagnostics.retry_summary,
    verification: result.diagnostics.verification,
    prompt_contract: result.diagnostics.prompt_contract || null,
    test_log: result.artifacts?.test_log || null,
    files_changed: result.files_changed || [],
  });
}

main().catch((err) => {
  console.error("canary_m9_autofix_retry.js failed");
  console.error(err);
  process.exit(1);
});
