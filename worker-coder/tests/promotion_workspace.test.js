import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { promoteIsolatedChanges } from "../promotion_workspace.js";
import { captureScopedSnapshot } from "../scoped_delta.js";

function writeFile(root, relPath, content) {
  const abs = path.join(root, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "promotion-workspace-"));
  const taskDir = path.join(root, "artifacts", "runs", "run-1", "task_1");
  const isolatedRoot = path.join(taskDir, "isolated_workspace");
  fs.mkdirSync(taskDir, { recursive: true });

  writeFile(root, "workspace/sandbox/crm_site/app.js", "export const value = 1;\n");
  writeFile(isolatedRoot, "workspace/sandbox/crm_site/app.js", "export const value = 2;\n");

  const baselineSnapshot = captureScopedSnapshot(root, ["workspace/sandbox/crm_site/app.js"]);

  const shadow = promoteIsolatedChanges({
    workspaceRoot: root,
    isolatedWorkspaceRoot: isolatedRoot,
    taskDir,
    filesChanged: ["workspace/sandbox/crm_site/app.js"],
    allowedTargetPaths: ["workspace/sandbox/crm_site/app.js"],
    mode: "shadow",
    baselineSnapshot,
  });
  assert.equal(shadow.ok, true);
  assert.equal(shadow.applied, false);
  assert.equal(
    fs.readFileSync(path.join(root, "sandbox", "crm_site", "app.js"), "utf8"),
    "export const value = 1;\n",
  );

  const promoted = promoteIsolatedChanges({
    workspaceRoot: root,
    isolatedWorkspaceRoot: isolatedRoot,
    taskDir,
    filesChanged: ["workspace/sandbox/crm_site/app.js"],
    allowedTargetPaths: ["workspace/sandbox/crm_site/app.js"],
    mode: "promote",
    baselineSnapshot,
  });
  assert.equal(promoted.ok, true);
  assert.equal(promoted.applied, true);
  assert.equal(
    fs.readFileSync(path.join(root, "sandbox", "crm_site", "app.js"), "utf8"),
    "export const value = 2;\n",
  );

  const blocked = promoteIsolatedChanges({
    workspaceRoot: root,
    isolatedWorkspaceRoot: isolatedRoot,
    taskDir,
    filesChanged: ["workspace/sandbox/other/file.js"],
    allowedTargetPaths: ["workspace/sandbox/crm_site/app.js"],
    mode: "promote",
    baselineSnapshot,
  });
  assert.equal(blocked.ok, false);
  assert.equal(blocked.applied, false);
  assert.match(blocked.error, /outside scope/);

  // Test drift detection — conflicting drift (drifted file overlaps with changedFiles)
  writeFile(root, "workspace/sandbox/crm_site/app.js", "export const value = 3;\n"); // mutate host after baseline
  const drifted = promoteIsolatedChanges({
    workspaceRoot: root,
    isolatedWorkspaceRoot: isolatedRoot,
    taskDir,
    filesChanged: ["workspace/sandbox/crm_site/app.js"],
    allowedTargetPaths: ["workspace/sandbox/crm_site/app.js"],
    mode: "promote",
    baselineSnapshot,
  });
  assert.equal(drifted.ok, false);
  assert.equal(drifted.applied, false);
  assert.match(drifted.error, /PROMOTION_CONFLICT/);

  // Test additive drift — another workflow added a NEW file this task doesn't touch.
  // This should NOT block promotion (concurrent workflows writing different files is safe).
  const taskDir2 = path.join(root, "artifacts", "runs", "run-1", "task_2");
  const isolatedRoot2 = path.join(taskDir2, "isolated_workspace");
  fs.mkdirSync(taskDir2, { recursive: true });
  writeFile(root, "workspace/sandbox/crm_site/stub_from_other_workflow.js", "// other workflow\n");
  const baselineSnapshotOther = captureScopedSnapshot(root, ["workspace/sandbox/crm_site/other.js"]);
  writeFile(isolatedRoot2, "workspace/sandbox/crm_site/other.js", "export const x = 42;\n");
  // The workspace has stub_from_other_workflow.js (from another workflow), but this task
  // only touches other.js — that's additive drift, not a conflict.
  const additiveDrift = promoteIsolatedChanges({
    workspaceRoot: root,
    isolatedWorkspaceRoot: isolatedRoot2,
    taskDir: taskDir2,
    filesChanged: ["workspace/sandbox/crm_site/other.js"],
    allowedTargetPaths: ["workspace/sandbox/crm_site/other.js"],
    mode: "promote",
    baselineSnapshot: baselineSnapshotOther,
  });
  assert.equal(additiveDrift.ok, true, `additive drift should not block: ${additiveDrift.error}`);
  assert.equal(additiveDrift.applied, true);

  console.log("promotion_workspace.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("promotion_workspace.test.js: failed");
  console.error(err);
  process.exit(1);
}
