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
    expectedArtifacts: ["impl/be_notes.md", "impl/be_changes/package.json", "handoff/be_to_fe.json", "release/release_notes.md", "release/README.md", "release/start.sh", "plan/interfaces.md", "impl/fe_changes/public/app.js"],
    stepId: "impl_be",
    taskPrompt: "Implement backend changes.",
  });
  assert.equal(scaffold.checked, true);
  assert.deepEqual(scaffold.failed, []);
  assert.ok(scaffold.created.includes("impl/be_notes.md"));
  assert.ok(scaffold.created.includes("impl/be_changes/package.json"));
  assert.ok(scaffold.created.includes("plan/interfaces.md"));
  assert.ok(scaffold.created.includes("impl/fe_changes/public/app.js"));
  assert.ok(fs.existsSync(path.join(rootAbs, "handoff", "be_to_fe.json")));
  const packageJson = JSON.parse(fs.readFileSync(path.join(rootAbs, "impl", "be_changes", "package.json"), "utf8"));
  assert.equal(packageJson.main, "server.js");
  const readmeText = fs.readFileSync(path.join(rootAbs, "release", "README.md"), "utf8");
  assert.match(readmeText, /npm install/);
  const startScriptText = fs.readFileSync(path.join(rootAbs, "release", "start.sh"), "utf8");
  assert.match(startScriptText, /node server\.js/);
  const interfacesText = fs.readFileSync(path.join(rootAbs, "plan", "interfaces.md"), "utf8");
  assert.match(interfacesText, /## GET \/api\/customers/);
  const genericWorkplanText = buildArtifactTemplate({
    relPath: "plan/workplan.md",
    rootAbs,
    stepId: "arch_design",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.match(genericWorkplanText, /## BE Tasks/);
  assert.match(genericWorkplanText, /## FE Tasks/);
  assert.match(genericWorkplanText, /\|\s*verify:/);
  const crmServerText = buildArtifactTemplate({
    relPath: "impl/be_changes/server.js",
    rootAbs,
    stepId: "impl_be",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.match(crmServerText, /app\.get\('\/api\/customers'/);
  assert.match(crmServerText, /express\.static\(publicDir\)/);
  assert.match(crmServerText, /process\.env\.PORT/);

  const handoff = JSON.parse(fs.readFileSync(path.join(rootAbs, "handoff", "be_to_fe.json"), "utf8"));
  assert.equal(handoff.from_step, "impl_be");
  assert.equal(handoff.api_contracts[0].name, "List Customers");

  const archHandoff = JSON.parse(buildArtifactTemplate({
    relPath: "handoff/architect_to_impl.json",
    rootAbs,
    stepId: "arch_design",
    taskPrompt: "Design architecture.",
  }));
  assert.equal(archHandoff.decisions[0].adr_id, "ADR-001");
  assert.equal(archHandoff.parallelization.fe_safe_parallel, true);

  const qaReport = buildArtifactTemplate({
    relPath: "verify/qa_report.json",
    rootAbs,
    stepId: "qa_verify",
    taskPrompt: "Run QA.",
  });
  assert.equal(isQaReportValid(qaReport, rootAbs), false);
  const realQaReport = JSON.stringify({
    overall_status: "pass",
    checks: [{ id: "check-1", status: "pass", evidence: "impl/fe_changes/app.js contains addTask function" }],
    verified_artifacts: ["impl/fe_changes/app.js"],
  });
  assert.equal(isQaReportValid(realQaReport, rootAbs), true);
  assert.deepEqual(loadAcceptanceIds(rootAbs), ["A1", "A2"]);
  assert.equal(markdownHasHeadings("# Smoke Report\n## Executed Checks\n## Result Summary\n", ["smoke report", "executed checks", "result summary"]), true);

  fs.mkdirSync(path.join(rootAbs, "qa"), { recursive: true });
  fs.writeFileSync(path.join(rootAbs, "qa", "smoke_report.md"), "auto-generated to satisfy workflow artifact contract\n");
  fs.writeFileSync(path.join(rootAbs, "plan", "interfaces.md"), "# plan/interfaces.md\n\nScaffold note: baseline content generated for workflow continuity.\n");
  const repaired = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["qa/smoke_report.md", "plan/interfaces.md"],
    stepId: "arch_design",
    taskPrompt: "Design architecture.",
  });
  assert.ok(repaired.repaired.includes("qa/smoke_report.md"));
  assert.ok(repaired.repaired.includes("plan/interfaces.md"));
  const repairedInterfacesText = fs.readFileSync(path.join(rootAbs, "plan", "interfaces.md"), "utf8");
  assert.match(repairedInterfacesText, /## POST \/api\/customers/);
  fs.mkdirSync(path.join(rootAbs, "impl", "be_changes"), { recursive: true });
  fs.writeFileSync(path.join(rootAbs, "impl", "be_changes", "server.js"), [
    "import express from 'express';",
    "const app = express();",
    "app.get('*', (_req, res) => res.status(404).send('Not Found'));",
    "app.listen(process.env.PORT || 3000);",
  ].join("\n"));
  const repairedCrmBackend = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["impl/be_changes/server.js"],
    stepId: "impl_be",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.ok(repairedCrmBackend.repaired.includes("impl/be_changes/server.js"));
  const repairedServerText = fs.readFileSync(path.join(rootAbs, "impl", "be_changes", "server.js"), "utf8");
  assert.match(repairedServerText, /app\.get\('\/api\/customers'/);
  assert.doesNotMatch(repairedServerText, /app\.get\('\*'/);

  const staticArchHandoff = JSON.parse(buildArtifactTemplate({
    relPath: "handoff/architect_to_impl.json",
    rootAbs,
    stepId: "arch_design",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: single_file_html\nGoal: Build a landing page",
  }));
  assert.deepEqual(staticArchHandoff.interfaces, ["GET /", "GET /styles.css", "GET /app.js", "Event: faq.toggle"]);

  const staticInterfacesText = buildArtifactTemplate({
    relPath: "plan/interfaces.md",
    rootAbs,
    stepId: "arch_design",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: single_file_html\nGoal: Build a landing page",
  });
  assert.match(staticInterfacesText, /## GET \//);
  assert.match(staticInterfacesText, /## Event: faq\.toggle/);
  const staticWorkplanText = buildArtifactTemplate({
    relPath: "plan/workplan.md",
    rootAbs,
    stepId: "arch_design",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: single_file_html\nGoal: Build a landing page",
  });
  assert.match(staticWorkplanText, /## BE Tasks/);
  assert.match(staticWorkplanText, /## FE Tasks/);
  assert.match(staticWorkplanText, /\|\s*verify:/);

  console.log("artifact_scaffold.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("artifact_scaffold.test.js: failed");
  console.error(err);
  process.exit(1);
}
