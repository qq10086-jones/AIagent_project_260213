import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { promoteIsolatedChanges } from "../promotion_workspace.js";

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

  writeFile(root, "sandbox/crm_site/app.js", "export const value = 1;\n");
  writeFile(isolatedRoot, "sandbox/crm_site/app.js", "export const value = 2;\n");

  const shadow = promoteIsolatedChanges({
    workspaceRoot: root,
    isolatedWorkspaceRoot: isolatedRoot,
    taskDir,
    filesChanged: ["sandbox/crm_site/app.js"],
    allowedTargetPaths: ["sandbox/crm_site/app.js"],
    mode: "shadow",
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
    filesChanged: ["sandbox/crm_site/app.js"],
    allowedTargetPaths: ["sandbox/crm_site/app.js"],
    mode: "promote",
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
    filesChanged: ["sandbox/other/file.js"],
    allowedTargetPaths: ["sandbox/crm_site/app.js"],
    mode: "promote",
  });
  assert.equal(blocked.ok, false);
  assert.equal(blocked.applied, false);
  assert.match(blocked.error, /outside scope/);

  console.log("promotion_workspace.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("promotion_workspace.test.js: failed");
  console.error(err);
  process.exit(1);
}
