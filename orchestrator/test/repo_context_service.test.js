import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { createRepoContextService } from "../src/domain/repo_context_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-repo-context-"));
}

function writeFile(workspaceRoot, rel, content) {
  const abs = path.join(workspaceRoot, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

test("repo context service builds repo map and scoped context packet", async () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "package.json", JSON.stringify({ name: "crm-app", scripts: { test: "node --test" } }, null, 2));
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.js", "export function boot() { return true; }\n");
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/server.js", "function startServer() { return 'ok'; }\nmodule.exports = { startServer };\n");
  writeFile(workspaceRoot, "workspace/sandbox/crm_site/app.test.js", "test('boot', () => {})\n");

  const service = createRepoContextService({ workspaceRoot });
  const repoMap = await service.buildRepoMap({
    targetPaths: ["workspace/sandbox/crm_site/"],
    recentChangedFiles: ["workspace/sandbox/crm_site/server.js"],
  });
  const packet = await service.buildContextPacket({
    role: "backend",
    stepId: "impl_be",
    targetPaths: ["workspace/sandbox/crm_site/"],
    recentChangedFiles: ["workspace/sandbox/crm_site/server.js"],
    memoryHints: ["previous test failed at app.test.js"],
  });

  assert.equal(repoMap.target_paths[0], "workspace/sandbox/crm_site/");
  assert.match(repoMap.candidate_files.join("\n"), /sandbox\/crm_site\/app\.js/);
  assert.match(repoMap.entrypoints.join("\n"), /sandbox\/crm_site\/app\.js|sandbox\/crm_site\/server\.js/);
  assert.match(repoMap.related_tests.join("\n"), /sandbox\/crm_site\/app\.test\.js/);
  assert.equal(packet.step_id, "impl_be");
  assert.equal(packet.role, "backend");
  assert.deepEqual(packet.memory_hints, ["previous test failed at app.test.js"]);
  assert.match(JSON.stringify(packet.toolchain_facts), /crm-app/);
});
