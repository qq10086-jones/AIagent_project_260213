import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  captureScopedSnapshot,
  ensureImplementationDelta,
  gatherGitSummary,
} from "../scoped_delta.js";

function writeFile(root, relPath, content) {
  const abs = path.join(root, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

async function main() {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "scoped-delta-"));
  const taskDir = path.join(workspaceRoot, "artifacts", "runs", "run-1", "task_1");
  fs.mkdirSync(taskDir, { recursive: true });

  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.js", "export const value = 1;\n");
  const baseline = captureScopedSnapshot(workspaceRoot, ["workspace/sandbox/crm_site"]);
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.js", "export const value = 2;\n");

  const summary = await gatherGitSummary(workspaceRoot, taskDir, baseline, ["workspace/sandbox/crm_site"]);
  assert.equal(summary.diffStats.files, 1);
  assert.deepEqual(summary.filesChanged, ["workspace/sandbox/crm_site/app.js"]);
  assert.ok(summary.diffPath);

  const fallbackWorkspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "scoped-delta-fallback-"));
  const fallbackTaskDir = path.join(fallbackWorkspaceRoot, "artifacts", "runs", "run-2", "task_2");
  fs.mkdirSync(fallbackTaskDir, { recursive: true });

  const emptyCurrent = {
    filesChanged: [],
    diffStats: { added: 0, deleted: 0, files: 0 },
  };
  const noDeltaBaseline = captureScopedSnapshot(fallbackWorkspaceRoot, ["workspace/sandbox/crm_site/server.js"]);
  const recovered = await ensureImplementationDelta({
    workspaceRoot: fallbackWorkspaceRoot,
    stepId: "impl_be",
    taskId: "task-empty",
    executionAdapterPacket: { target_paths: ["workspace/sandbox/crm_site/server.js"] },
    taskPrompt: "Implement backend.",
    taskDir: fallbackTaskDir,
    baselineSnapshot: noDeltaBaseline,
    current: emptyCurrent,
  });
  assert.equal(recovered.diffStats.files, 0);
  assert.deepEqual(recovered.filesChanged, []);
  assert.equal(
    fs.existsSync(path.join(fallbackWorkspaceRoot, "sandbox", "crm_site", "workflow_impl_be_stub_task-empty.js")),
    false,
  );

  // Dir-target case: stub goes to taskDir only, workspace remains clean.
  const dirFallbackWorkspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "scoped-delta-dir-fallback-"));
  const dirFallbackTaskDir = path.join(dirFallbackWorkspaceRoot, "artifacts", "runs", "run-3", "task_3");
  fs.mkdirSync(dirFallbackTaskDir, { recursive: true });

  const dirBaseline = captureScopedSnapshot(dirFallbackWorkspaceRoot, ["workspace/sandbox/crm_site"]);
  const recoveredDir = await ensureImplementationDelta({
    workspaceRoot: dirFallbackWorkspaceRoot,
    stepId: "impl_fe",
    taskId: "task-dir",
    executionAdapterPacket: { target_paths: ["workspace/sandbox/crm_site"] },
    taskPrompt: "Implement frontend.",
    taskDir: dirFallbackTaskDir,
    baselineSnapshot: dirBaseline,
    current: emptyCurrent,
  });
  // Workspace unchanged: stub is NOT promoted, filesChanged is empty.
  assert.equal(recoveredDir.diffStats.files, 0);
  assert.deepEqual(recoveredDir.filesChanged, []);
  // Stub written as audit artifact inside taskDir — NOT into workspace/sandbox.
  assert.equal(
    fs.existsSync(path.join(dirFallbackWorkspaceRoot, "sandbox", "crm_site", "workflow_impl_fe_stub_task-dir.js")),
    false,
    "stub must not appear in workspace sandbox",
  );
  assert.equal(
    fs.existsSync(path.join(dirFallbackTaskDir, "workflow_impl_fe_stub_task-dir.js")),
    true,
    "stub must be written to taskDir as audit artifact",
  );

  console.log("scoped_delta.test.js: all tests passed");
}

main().catch((err) => {
  console.error("scoped_delta.test.js: failed");
  console.error(err);
  process.exit(1);
});
