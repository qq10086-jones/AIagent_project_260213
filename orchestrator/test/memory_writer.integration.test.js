import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { getPriorADRs, getTaskHistory } from "../src/domain/memory_reader.js";
import { persistWorkflowMemory } from "../src/domain/memory_writer.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-memory-writer-"));
}

test("persistWorkflowMemory appends task history and copies adr markdown for later reads", () => {
  const workspaceRoot = makeWorkspace();
  const memoryRoot = path.join(workspaceRoot, "memory");
  const releaseRoot = path.join(workspaceRoot, "artifacts", "release", "run-1");
  const runtimeRoot = path.join(workspaceRoot, "artifacts", "runs", "run-1");
  fs.mkdirSync(path.join(releaseRoot, "plan", "adr"), { recursive: true });
  fs.mkdirSync(path.join(runtimeRoot, "memory"), { recursive: true });
  fs.writeFileSync(
    path.join(releaseRoot, "plan", "adr", "ADR-001.md"),
    "# Use Postgres\n\nStatus: accepted\n"
  );
  fs.writeFileSync(
    path.join(runtimeRoot, "memory", "coding_failures.jsonl"),
    `${JSON.stringify({ failure_id: "cf-1", error_code: "E_TEST" })}\n`,
  );
  fs.writeFileSync(
    path.join(runtimeRoot, "memory", "coding_failure_latest.json"),
    JSON.stringify({ failure_id: "cf-1", error_code: "E_TEST" }, null, 2),
  );

  const previousMemoryRoot = process.env.MEMORY_ROOT;
  process.env.MEMORY_ROOT = memoryRoot;
  try {
    const result = persistWorkflowMemory({
      run: {
        workflow_run_id: "wf-1",
        run_id: "run-1",
        workflow_id: "coding_team_v0",
        project_type: "webapp_crm",
      },
      releaseRoot,
      runtimeRoot,
    });

    assert.equal(result.ok, true);
    assert.equal(result.copied_adr_paths.length, 1);
    assert.equal(result.coding_failure_memory.copied, true);

    const history = getTaskHistory("run-1", 5);
    assert.equal(history.length, 1);
    assert.equal(history[0].status, "succeeded");
    assert.equal(history[0].workflow_run_id, "wf-1");

    const adrs = getPriorADRs("run-1");
    assert.equal(adrs.length, 1);
    assert.equal(adrs[0].adr_id, "ADR-001");
    assert.equal(adrs[0].title, "Use Postgres");
    assert.equal(fs.existsSync(path.join(memoryRoot, "run-1", "coding_failures", "coding_failures.jsonl")), true);
  } finally {
    if (previousMemoryRoot === undefined) delete process.env.MEMORY_ROOT;
    else process.env.MEMORY_ROOT = previousMemoryRoot;
  }
});
