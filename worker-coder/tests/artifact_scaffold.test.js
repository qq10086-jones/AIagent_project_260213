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
  assert.match(crmServerText, /const express = require\('express'\);/);
  assert.match(crmServerText, /app\.get\('\/api\/customers'/);
  assert.match(crmServerText, /express\.static\(publicDir\)/);
  assert.match(crmServerText, /process\.env\.PORT/);
  assert.match(crmServerText, /app\.delete\('\/api\/customers\/:id'/);
  assert.match(crmServerText, /app\.get\('\/api\/activity'/);
  assert.doesNotMatch(crmServerText, /app\.get\('\/health'/);
  const crmSpecText = buildArtifactTemplate({
    relPath: "plan/spec.md",
    rootAbs,
    stepId: "pm_spec",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.doesNotMatch(crmSpecText, /search and filter/i);
  const crmPackageJson = JSON.parse(buildArtifactTemplate({
    relPath: "impl/be_changes/package.json",
    rootAbs,
    stepId: "impl_be",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  }));
  assert.equal(crmPackageJson.main, "server.js");
  assert.equal(crmPackageJson.type, "module");
  assert.deepEqual(Object.keys(crmPackageJson.dependencies).sort(), ["cors", "express"]);

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
  fs.writeFileSync(path.join(rootAbs, "impl", "be_changes", "server.js"), [
    "const express = require('express');",
    "const path = require('path');",
    "const app = express();",
    "app.use(express.json());",
    "app.use(express.static(path.join(__dirname, 'public')));",
    "const customers = [{ id: 'c1', name: 'Ada', email: 'ada@example.com' }];",
    "app.get('/api/customers', (_req, res) => res.json({ customers }));",
    "app.get('/api/customers/:id', (req, res) => res.json(customers[0]));",
    "app.post('/api/customers', (_req, res) => res.status(201).json(customers[0]));",
    "app.put('/api/customers/:id', (_req, res) => res.json(customers[0]));",
    "app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));",
    "app.listen(process.env.PORT || 3000);",
  ].join("\n"));
  const preservedCrmBackend = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["impl/be_changes/server.js"],
    stepId: "impl_be",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.deepEqual(preservedCrmBackend.repaired, []);
  const preservedServerText = fs.readFileSync(path.join(rootAbs, "impl", "be_changes", "server.js"), "utf8");
  assert.match(preservedServerText, /const express = require\('express'\);/);
  fs.writeFileSync(path.join(rootAbs, "impl", "be_changes", "server.js"), [
    "import express from 'express';",
    "import cors from 'cors';",
    "const app = express();",
    "app.use(cors());",
    "app.listen(process.env.PORT || 3000);",
  ].join("\n"));
  fs.writeFileSync(path.join(rootAbs, "impl", "be_changes", "package.json"), JSON.stringify({
    name: "crm-live-validation",
    version: "1.0.0",
    main: "server.js",
    dependencies: { express: "^4.21.2" },
  }, null, 2));
  const repairedPackageManifest = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["impl/be_changes/package.json"],
    stepId: "impl_be",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.ok(repairedPackageManifest.repaired.includes("impl/be_changes/package.json"));
  const repairedPackageJson = JSON.parse(fs.readFileSync(path.join(rootAbs, "impl", "be_changes", "package.json"), "utf8"));
  assert.equal(repairedPackageJson.type, "module");
  assert.equal(repairedPackageJson.main, "server.js");
  assert.equal(repairedPackageJson.dependencies.express, "^4.18.2");
  assert.equal(repairedPackageJson.dependencies.cors, "^2.8.5");
  fs.mkdirSync(path.join(rootAbs, "handoff"), { recursive: true });
  fs.writeFileSync(path.join(rootAbs, "handoff", "impl_to_qa.json"), JSON.stringify({
    from_step: "impl_fe",
    to_step: "qa",
    fe_changes_path: "impl/fe_changes",
    be_changes_path: "impl/be_changes",
    run_instructions: {
      backend: "node server.js",
    },
  }, null, 2));
  const repairedImplToQa = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["handoff/impl_to_qa.json"],
    stepId: "impl_fe",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app",
  });
  assert.ok(repairedImplToQa.repaired.includes("handoff/impl_to_qa.json"));
  const repairedImplToQaJson = JSON.parse(fs.readFileSync(path.join(rootAbs, "handoff", "impl_to_qa.json"), "utf8"));
  assert.deepEqual(repairedImplToQaJson.from_steps, ["impl_be", "impl_fe"]);
  assert.equal(repairedImplToQaJson.to_step, "qa_verify");
  assert.equal(typeof repairedImplToQaJson.run_instructions, "string");
  assert.equal(Array.isArray(repairedImplToQaJson.known_limitations), true);
  fs.writeFileSync(path.join(rootAbs, "plan", "interfaces.md"), [
    "# Interfaces",
    "",
    "## GET /api/customers",
    "",
    "## DELETE /api/customers/:id",
    "",
    "## GET /health",
  ].join("\n"));
  fs.writeFileSync(path.join(rootAbs, "plan", "workplan.json"), JSON.stringify({
    be_tasks: [
      { id: "T-BE-1", description: "List customers", verify: "GET /api/customers returns data" },
      { id: "T-BE-2", description: "Get customer detail", verify: "GET /api/customers/:id returns data" },
      { id: "T-BE-3", description: "Create customer", verify: "POST /api/customers returns 201" },
      { id: "T-BE-4", description: "Update customer", verify: "PUT /api/customers/:id returns 200" },
      { id: "T-BE-5", description: "Delete customer", verify: "DELETE /api/customers/:id returns 204" },
      { id: "T-BE-6", description: "Health endpoint", verify: "GET /health returns 200" },
    ],
    fe_tasks: [
      { id: "T-FE-1", description: "Customer list", verify: "list renders" },
      { id: "T-FE-2", description: "Detail view", verify: "detail renders" },
      { id: "T-FE-3", description: "Create form", verify: "create works" },
      { id: "T-FE-4", description: "Edit form", verify: "edit works" },
      { id: "T-FE-5", description: "Search and filter", verify: "search works" },
      { id: "T-FE-6", description: "Responsive mobile layout", verify: "mobile works" },
    ],
  }, null, 2));
  const repairedMinimalArch = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["plan/interfaces.md", "plan/workplan.json"],
    stepId: "arch_design",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app. Keep changes reviewable.",
  });
  assert.ok(repairedMinimalArch.repaired.includes("plan/interfaces.md"));
  assert.ok(repairedMinimalArch.repaired.includes("plan/workplan.json"));
  const repairedMinimalInterfaces = fs.readFileSync(path.join(rootAbs, "plan", "interfaces.md"), "utf8");
  assert.doesNotMatch(repairedMinimalInterfaces, /DELETE \/api\/customers\/:id/);
  assert.doesNotMatch(repairedMinimalInterfaces, /GET \/health/);
  const repairedMinimalWorkplan = JSON.parse(fs.readFileSync(path.join(rootAbs, "plan", "workplan.json"), "utf8"));
  assert.ok(repairedMinimalWorkplan.be_tasks.length <= 5);
  assert.ok(repairedMinimalWorkplan.fe_tasks.length <= 5);
  fs.writeFileSync(path.join(rootAbs, "handoff", "be_to_fe.json"), JSON.stringify({
    api_contracts: {
      "GET /api/customers": {
        description: "Returns summary list of customers",
        response: { type: "array" },
      },
    },
    shared_types: {
      Customer: {
        id: "string",
        name: "string",
      },
    },
    scope_constraints: ["Same-origin only"],
  }, null, 2));
  const repairedBeToFe = ensureExpectedArtifacts({
    workspaceRoot,
    artifactRoot,
    expectedArtifacts: ["handoff/be_to_fe.json"],
    stepId: "impl_be",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: webapp_crm\nGoal: Build a minimal CRM web app. Keep changes reviewable.",
  });
  assert.ok(repairedBeToFe.repaired.includes("handoff/be_to_fe.json"));
  const repairedBeToFeJson = JSON.parse(fs.readFileSync(path.join(rootAbs, "handoff", "be_to_fe.json"), "utf8"));
  assert.equal(repairedBeToFeJson.from_step, "impl_be");
  assert.equal(repairedBeToFeJson.to_step, "impl_fe");
  assert.equal(Array.isArray(repairedBeToFeJson.api_contracts), true);
  assert.equal(Array.isArray(repairedBeToFeJson.shared_types), true);
  assert.equal(repairedBeToFeJson.api_contracts[0].path, "/api/customers");
  assert.equal(repairedBeToFeJson.shared_types[0].name, "Customer");

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
  const staticWorkplanJson = JSON.parse(buildArtifactTemplate({
    relPath: "plan/workplan.json",
    rootAbs,
    stepId: "arch_design",
    taskPrompt: "Workflow: coding_team_v0\nProject Type: single_file_html\nGoal: Build a landing page",
  }));
  assert.equal(Array.isArray(staticWorkplanJson.be_tasks), true);
  assert.equal(Array.isArray(staticWorkplanJson.fe_tasks), true);
  assert.equal(staticWorkplanJson.be_tasks[0].id, "T-BE-1");
  assert.equal(staticWorkplanJson.fe_tasks[0].id, "T-FE-1");

  console.log("artifact_scaffold.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("artifact_scaffold.test.js: failed");
  console.error(err);
  process.exit(1);
}
