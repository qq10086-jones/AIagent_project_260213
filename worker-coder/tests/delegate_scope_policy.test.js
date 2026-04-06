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
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.js", "const value = 1;\n");

  // Pre-create the PM artifacts that role validation expects
  // Role validation checks files under artifact_root/plan/...
  const artBase = "orchestrator/artifacts/test/pm_spec_scope";
  writeFile(workspaceRoot, `${artBase}/plan/spec.md`, [
    "# Scope",
    "Test scope for delegate policy test.",
    "",
    "# User Stories",
    "As a user I want to test scope policy.",
    "",
    "# Acceptance Criteria",
    "- Tests pass",
    "",
    "# Non-Goals",
    "- Out of scope items",
    "",
    "# Artifact List",
    "- plan/spec.md",
    "",
  ].join("\n"));
  writeFile(workspaceRoot, `${artBase}/plan/acceptance.json`, JSON.stringify({
    criteria: ["Tests pass"],
    artifacts: ["plan/spec.md"],
    owner: "test",
    version: "1.0",
  }));
  writeFile(workspaceRoot, `${artBase}/plan/milestones.md`, "# Milestones\n- M1: test artifacts\n");
  writeFile(workspaceRoot, `${artBase}/handoff/pm_to_architect.json`, JSON.stringify({
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "test",
    artifacts: ["plan/spec.md"],
    acceptance: { criteria: ["test"] },
  }));

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
  assert.ok(!result.diagnostics.error_code, `expected no error_code, got: ${result.diagnostics.error_code}`);
}

async function testImplementationStepStillRequiresTargetPaths() {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.js", "const value = 1;\n");

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
