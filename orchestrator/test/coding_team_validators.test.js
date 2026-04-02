import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { validateArchitectOutput } from "../src/coding_team_validators.js";

function writeJson(absPath, value) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, JSON.stringify(value, null, 2));
}

function writeText(absPath, value) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, value);
}

test("validateArchitectOutput accepts crm workplan aligned with node backend contract", () => {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "arch-validator-ok-"));
  const artifactRoot = "runtime/artifacts/release/run-ok";
  const rootAbs = path.join(workspaceRoot, artifactRoot);
  writeText(path.join(rootAbs, "plan", "spec.md"), [
    "# Scope",
    "",
    "- Build a minimal CRM web app with customer list, detail page, and add/edit form.",
    "- Keep implementation reviewable.",
  ].join("\n"));
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: [
      "Core customer list is visible with stable navigation.",
      "Detail view loads from a selected record.",
      "Create/edit form supports basic validation.",
    ],
  });
  writeText(path.join(rootAbs, "plan", "arch.md"), [
    "# Module Breakdown",
    "",
    "## Modules",
    "- frontend app",
    "- backend api",
    "",
    "## Interfaces",
    "- GET /api/customers",
    "",
    "## Dependency Choices",
    "- Node.js + Express backend",
    "",
    "## Risk Notes",
    "- schema drift",
  ].join("\n"));
  writeText(path.join(rootAbs, "plan", "interfaces.md"), [
    "# Interfaces",
    "",
    "## GET /api/customers",
    "- request body: none",
    "- response body: Customer[]",
    "- auth requirement: none",
  ].join("\n"));
  writeText(path.join(rootAbs, "plan", "workplan.md"), "## BE Tasks\n- [ ] T-BE-1: Implement Express backend | verify: GET /api/customers returns 200\n");
  writeJson(path.join(rootAbs, "plan", "workplan.json"), {
    be_tasks: [
      { id: "T-BE-1", description: "Create Express server and package.json for the CRM slice", verify: "npm install succeeds and server starts" },
      { id: "T-BE-2", description: "Implement GET /api/customers and GET /api/customers/:id handlers", verify: "both endpoints return 200 for seeded customer data" },
      { id: "T-BE-3", description: "Implement POST /api/customers and PUT /api/customers/:id handlers with validation", verify: "create and edit flows persist valid customer data" },
    ],
    fe_tasks: [
      { id: "T-FE-1", description: "Render customer list with stable selection navigation", verify: "customer list renders and selected record opens detail view" },
      { id: "T-FE-2", description: "Implement create/edit form bound to same-origin API endpoints", verify: "form submits successfully for create and edit flows" },
    ],
  });
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ title: "schema drift", mitigation: "typed handoffs", level: "medium" }],
    decision_log: ["Use Node.js + Express for the runtime-aligned CRM backend"],
  });

  const result = validateArchitectOutput({ workspaceRoot, artifactRoot });
  assert.equal(result.ok, true);
});

test("validateArchitectOutput rejects crm workplan that drifts into python stack and auth scope", () => {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "arch-validator-bad-"));
  const artifactRoot = "runtime/artifacts/release/run-bad";
  const rootAbs = path.join(workspaceRoot, artifactRoot);
  writeText(path.join(rootAbs, "plan", "spec.md"), [
    "# Scope",
    "",
    "- Build a minimal CRM web app with customer list, detail page, and add/edit form.",
    "- Keep implementation reviewable.",
  ].join("\n"));
  writeJson(path.join(rootAbs, "plan", "acceptance.json"), {
    criteria: [
      "Core customer list is visible with stable navigation.",
      "Detail view loads from a selected record.",
      "Create/edit form supports basic validation.",
    ],
  });
  writeText(path.join(rootAbs, "plan", "arch.md"), [
    "# Module Breakdown",
    "",
    "## Modules",
    "- frontend app",
    "- backend api",
    "",
    "## Interfaces",
    "- GET /api/customers",
    "",
    "## Dependency Choices",
    "- Flask backend",
    "",
    "## Risk Notes",
    "- auth drift",
  ].join("\n"));
  writeText(path.join(rootAbs, "plan", "interfaces.md"), [
    "# Interfaces",
    "",
    "## GET /api/customers",
    "- request body: none",
    "- response body: Customer[]",
    "- auth requirement: none",
  ].join("\n"));
  writeText(path.join(rootAbs, "plan", "workplan.md"), "## BE Tasks\n- [ ] T-BE-1: Create Flask app | verify: flask run works\n");
  writeJson(path.join(rootAbs, "plan", "workplan.json"), {
    be_tasks: [
      { id: "T-BE-1", description: "Create Flask app skeleton", verify: "flask run starts" },
      { id: "T-BE-2", description: "Add API key auth middleware", verify: "requests without X-API-Key return 401" },
      { id: "T-BE-3", description: "Implement DELETE /api/customers/:id endpoint", verify: "returns 204" },
      { id: "T-BE-4", description: "Add pagination to GET /api/customers", verify: "response includes page metadata" },
      { id: "T-BE-5", description: "Seed demo data loader", verify: "seed script populates customers" },
      { id: "T-BE-6", description: "Refactor backend into services", verify: "service layer starts cleanly" },
    ],
    fe_tasks: [
      { id: "T-FE-1", description: "Render CRM list from same-origin API", verify: "customer list renders" },
      { id: "T-FE-2", description: "Add mobile responsive layout", verify: "usable at 375px width" },
      { id: "T-FE-3", description: "Add delete button with confirmation", verify: "delete removes row" },
      { id: "T-FE-4", description: "Add pagination controls", verify: "next page loads" },
      { id: "T-FE-5", description: "Add search filter bar", verify: "filtered results update" },
      { id: "T-FE-6", description: "Refine empty states", verify: "empty state shown when no rows" },
    ],
  });
  writeJson(path.join(rootAbs, "risk", "risk_report.json"), {
    risks: [{ title: "auth drift", mitigation: "limit scope", level: "medium" }],
    decision_log: ["Use Flask and API key auth"],
  });

  const result = validateArchitectOutput({ workspaceRoot, artifactRoot });
  assert.equal(result.ok, false);
  assert.match(String(result.detail || ""), /python_runtime_conflicts_with_node_backend_contract/);
  assert.match(String(result.detail || ""), /unexpected_auth_scope_for_minimal_crm/);
  assert.match(String(result.detail || ""), /delete_scope_not_present_in_interfaces/);
  assert.match(String(result.detail || ""), /pagination_not_requested_by_acceptance/);
  assert.match(String(result.detail || ""), /responsive_scope_not_requested/);
  assert.match(String(result.detail || ""), /be_task_count_exceeds_minimal_scope/);
  assert.match(String(result.detail || ""), /fe_task_count_exceeds_minimal_scope/);
});
