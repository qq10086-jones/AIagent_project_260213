import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { CodingService } from "../coding_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "worker-coder-isolation-shadow-"));
}

function writeFile(workspaceRoot, relPath, content) {
  const abs = path.join(workspaceRoot, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

async function main() {
  const previousMode = process.env.CODER_ISOLATION_MODE;
  process.env.CODER_ISOLATION_MODE = "shadow";

  try {
    const workspaceRoot = makeWorkspace();
    writeFile(workspaceRoot, "sandbox/crm_site/app.js", "export const value = 1;\n");
    writeFile(
      workspaceRoot,
      "sandbox/crm_site/package.json",
      JSON.stringify({
        name: "crm-site-shadow-test",
        scripts: {
          lint: "node --check app.js",
        },
      }, null, 2),
    );

    const result = await CodingService.delegateTask({
      workspaceRoot,
      task_prompt: "update frontend file",
      step_id: "impl_fe",
      target_paths: ["sandbox/crm_site/app.js"],
      provider: "opencode",
      run_id: "run-shadow",
      task_id: "task-shadow",
      opencode_command: [
        "mock-inline-autofix",
        "sandbox/crm_site/app.js",
        "{{task_prompt}}",
      ],
      verification_command: "node --check sandbox/crm_site/app.js",
    });

    assert.equal(result.ok, false, JSON.stringify(result));
    assert.equal(result.diagnostics.isolation.enabled, true);
    assert.equal(result.diagnostics.execution_workspace_root.endsWith(path.join("task_task-shadow", "isolated_workspace")), true);
    assert.equal(result.diagnostics.error_code, "E_STATIC_CHECK_FAILED");
    assert.equal(
      fs.readFileSync(path.join(workspaceRoot, "sandbox", "crm_site", "app.js"), "utf8"),
      "export const value = 1;\n",
    );
    assert.ok(result.artifacts.isolation_execution_mode);
    assert.ok(result.artifacts.isolation_baseline_manifest);
    assert.ok(result.artifacts.isolation_workspace_manifest);

    console.log("isolation_delegate_shadow.test.js: all tests passed");
  } finally {
    if (previousMode === undefined) delete process.env.CODER_ISOLATION_MODE;
    else process.env.CODER_ISOLATION_MODE = previousMode;
  }
}

main().catch((err) => {
  console.error("isolation_delegate_shadow.test.js: failed");
  console.error(err);
  process.exit(1);
});
