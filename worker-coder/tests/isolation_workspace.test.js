import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  inferIsolationContextPaths,
  materializeIsolationWorkspace,
  resolveIsolationMode,
} from "../isolation_workspace.js";

function writeFile(root, relPath, content) {
  const abs = path.join(root, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function main() {
  assert.equal(resolveIsolationMode("scaffold"), "scaffold");
  assert.equal(resolveIsolationMode("shadow"), "shadow");
  assert.equal(resolveIsolationMode("promote"), "promote");
  assert.equal(resolveIsolationMode(""), "disabled");
  assert.deepEqual(
    inferIsolationContextPaths(["sandbox/crm_site/app.js"]),
    ["sandbox/crm_site"],
  );

  const root = fs.mkdtempSync(path.join(os.tmpdir(), "isolation-workspace-"));
  const taskDir = path.join(root, "artifacts", "runs", "run-1", "task_1");
  fs.mkdirSync(taskDir, { recursive: true });
  writeFile(root, "sandbox/crm_site/app.js", "export const app = 1;\n");
  writeFile(root, "sandbox/crm_site/components/Card.js", "export const Card = 1;\n");

  const disabled = materializeIsolationWorkspace({
    workspaceRoot: root,
    taskDir,
    targetPaths: ["sandbox/crm_site/app.js"],
    mode: "disabled",
  });
  assert.equal(disabled.enabled, false);
  assert.ok(disabled.artifacts.execution_mode);
  assert.equal(disabled.artifacts.baseline_manifest, null);

  const scaffolded = materializeIsolationWorkspace({
    workspaceRoot: root,
    taskDir,
    targetPaths: ["sandbox/crm_site/app.js", "sandbox/crm_site/components"],
    mode: "scaffold",
  });
  assert.equal(scaffolded.enabled, true);
  assert.ok(scaffolded.artifacts.execution_mode);
  assert.ok(scaffolded.artifacts.baseline_manifest);
  assert.ok(scaffolded.artifacts.isolated_workspace_manifest);
  assert.equal(
    fs.existsSync(path.join(scaffolded.isolatedWorkspaceRoot, "sandbox", "crm_site", "app.js")),
    true,
  );
  assert.equal(
    fs.existsSync(path.join(scaffolded.isolatedWorkspaceRoot, "sandbox", "crm_site", "components", "Card.js")),
    true,
  );

  const baselineManifest = JSON.parse(fs.readFileSync(scaffolded.artifacts.baseline_manifest, "utf8"));
  const isolatedManifest = JSON.parse(fs.readFileSync(scaffolded.artifacts.isolated_workspace_manifest, "utf8"));
  assert.equal(baselineManifest.entries.some((item) => item.path === "sandbox/crm_site/app.js"), true);
  assert.equal(isolatedManifest.entries.some((item) => item.path === "sandbox/crm_site/components/Card.js"), true);

  console.log("isolation_workspace.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("isolation_workspace.test.js: failed");
  console.error(err);
  process.exit(1);
}
