import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { clampInt, runStaticChecks } from "../static_checks.js";

function writeFile(root, relPath, content) {
  const abs = path.join(root, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

async function main() {
  assert.equal(clampInt("3.9", 1, 5, 2), 3);
  assert.equal(clampInt("bad", 1, 5, 2), 2);
  assert.equal(clampInt(9, 1, 5, 2), 5);

  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "static-checks-"));
  const taskDir = path.join(workspaceRoot, "artifacts", "runs", "run-1", "task_1");
  fs.mkdirSync(taskDir, { recursive: true });

  const empty = await runStaticChecks({
    workspaceRoot,
    filesChanged: [],
    taskDir,
  });
  assert.equal(empty.checked, false);
  assert.equal(empty.ok, true);

  writeFile(workspaceRoot, "workspace/sandbox/data/bad.json", "{ bad json");
  const badJson = await runStaticChecks({
    workspaceRoot,
    filesChanged: ["workspace/sandbox/data/bad.json"],
    taskDir,
  });
  assert.equal(badJson.checked, true);
  assert.equal(badJson.ok, false);
  assert.match(badJson.error, /json parse failed/);
  assert.ok(badJson.logPath);

  writeFile(workspaceRoot, "workspace/sandbox/data/node_modules/pkg/tsconfig.json", "{ bad json with comments }");
  const ignoredNodeModules = await runStaticChecks({
    workspaceRoot,
    filesChanged: ["workspace/sandbox/data/node_modules/pkg/tsconfig.json"],
    taskDir,
  });
  assert.equal(ignoredNodeModules.checked, true);
  assert.equal(ignoredNodeModules.ok, true);
  assert.deepEqual(ignoredNodeModules.records, []);

  console.log("static_checks.test.js: all tests passed");
}

main().catch((err) => {
  console.error("static_checks.test.js: failed");
  console.error(err);
  process.exit(1);
});
