import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  buildArtifactTemplate,
  ensureExpectedArtifacts,
  isQaReportValid,
  loadAcceptanceIds,
  markdownHasHeadings,
} from "../artifact_scaffold.js";

function main() {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "artifact-scaffold-"));
  const artifactRoot = "artifacts/release/run-1";
  const rootAbs = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(rootAbs, "plan"), { recursive: true });
  fs.writeFileSync(path.join(rootAbs, "plan", "acceptance.json"), JSON.stringify({
    criteria: ["one", "two"],
    artifacts: ["plan/spec.md"],
    owner: "pm",
    version: "v1",
  }, null, 2));

  const scaffold = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["impl/be_notes.md", "handoff/be_to_fe.json", "release/release_notes.md"],
    stepId: "impl_be",
    taskPrompt: "Implement backend changes.",
  });
  assert.equal(scaffold.checked, true);
  assert.deepEqual(scaffold.failed, []);
  assert.ok(scaffold.created.includes("impl/be_notes.md"));
  assert.ok(fs.existsSync(path.join(rootAbs, "handoff", "be_to_fe.json")));

  const handoff = JSON.parse(fs.readFileSync(path.join(rootAbs, "handoff", "be_to_fe.json"), "utf8"));
  assert.equal(handoff.from_step, "impl_be");
  assert.equal(handoff.api_contracts[0].name, "List Customers");

  const qaReport = buildArtifactTemplate({
    relPath: "verify/qa_report.json",
    rootAbs,
    stepId: "qa_verify",
    taskPrompt: "Run QA.",
  });
  assert.equal(isQaReportValid(qaReport, rootAbs), true);
  assert.deepEqual(loadAcceptanceIds(rootAbs), ["A1", "A2"]);
  assert.equal(markdownHasHeadings("# Smoke Report\n## Executed Checks\n## Result Summary\n", ["smoke report", "executed checks", "result summary"]), true);

  fs.mkdirSync(path.join(rootAbs, "qa"), { recursive: true });
  fs.writeFileSync(path.join(rootAbs, "qa", "smoke_report.md"), "auto-generated to satisfy workflow artifact contract\n");
  const repaired = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["qa/smoke_report.md"],
    stepId: "qa_verify",
    taskPrompt: "Run QA.",
  });
  assert.ok(repaired.repaired.includes("qa/smoke_report.md"));

  console.log("artifact_scaffold.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("artifact_scaffold.test.js: failed");
  console.error(err);
  process.exit(1);
}
