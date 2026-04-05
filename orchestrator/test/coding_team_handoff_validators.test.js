import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { validateCodingTeamHandoff } from "../src/coding_team_handoff_validators.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-handoff-"));
}

test("be_to_fe typed handoff accepts empty api_contracts/shared_types arrays when fields exist", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-static";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "impl", "be_changes"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "handoff"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_changes", "server.js"), "export default {};\n");
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_notes.md"), "# Backend Notes\n");
  fs.writeFileSync(
    path.join(releaseRoot, "handoff", "be_to_fe.json"),
    JSON.stringify({
      from_step: "impl_be",
      to_step: "impl_fe",
      be_changes_path: "impl/be_changes",
      api_contracts: [],
      shared_types: [],
      scope_constraints: ["Static site only."],
    }, null, 2)
  );

  const result = validateCodingTeamHandoff({
    workspaceRoot,
    artifactRoot,
    handoff: {
      from_step: "impl_be",
      to_steps: ["impl_fe"],
      required_artifacts: ["impl/be_changes/server.js", "impl/be_notes.md", "handoff/be_to_fe.json"],
      required_sections: ["api_contracts", "shared_types", "scope_constraints"],
      typed_handoff: {
        file: "handoff/be_to_fe.json",
        required_fields: ["from_step", "to_step", "be_changes_path", "api_contracts", "shared_types", "scope_constraints"],
      },
    },
  });

  assert.equal(result.ok, true);
  assert.equal(result.typed_schema_checked, "coding_team_be_to_fe_handoff.schema.json");
});

test("architect_to_impl typed handoff rejects object-rich fields that drift from schema", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-arch-invalid";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "plan"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "risk"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "handoff"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "plan", "arch.md"), "# Architecture\n");
  fs.writeFileSync(path.join(releaseRoot, "plan", "interfaces.md"), "## GET /api/customers\n");
  fs.writeFileSync(path.join(releaseRoot, "plan", "workplan.md"), "## BE Tasks\n- [ ] T-BE-1: Do thing | verify: works\n\n## FE Tasks\n- [ ] T-FE-1: Do thing | verify: works\n");
  fs.writeFileSync(
    path.join(releaseRoot, "handoff", "architect_to_impl.json"),
    JSON.stringify({
      from_step: "arch_design",
      to_steps: ["impl_be", "impl_fe"],
      modules: [{ name: "backend" }],
      interfaces: [{ method: "GET", path: "/api/customers" }],
      decisions: [{ id: "ADR-1", title: "Use Express" }],
      risks: [{ id: "R-1", title: "schema drift" }],
      workplan: {
        be_tasks: [{ id: "T-BE-1", description: "Do thing", verify: "works" }],
        fe_tasks: [{ id: "T-FE-1", description: "Do thing", verify: "works" }],
      },
    }, null, 2)
  );
  fs.writeFileSync(
    path.join(releaseRoot, "risk", "risk_report.json"),
    JSON.stringify({ risks: ["schema drift"] }, null, 2)
  );

  const result = validateCodingTeamHandoff({
    workspaceRoot,
    artifactRoot,
    handoff: {
      from_step: "arch_design",
      to_steps: ["impl_be", "impl_fe"],
      required_artifacts: ["plan/arch.md", "plan/interfaces.md", "risk/risk_report.json", "plan/workplan.md", "handoff/architect_to_impl.json"],
      required_sections: [],
      typed_handoff: {
        file: "handoff/architect_to_impl.json",
        required_fields: ["from_step", "to_steps", "modules", "interfaces", "decisions", "risks", "workplan"],
      },
    },
  });

  assert.equal(result.ok, false);
  assert.equal(result.code, "HANDOFF_TYPED_SCHEMA_INVALID");
  assert.match(result.detail, /\$\.modules\[0\] expected string, got object/);
  assert.match(result.detail, /\$\.decisions\[0\]\.adr_id missing/);
});

test("be_to_fe typed handoff accepts api_contracts without name field (LLM tolerance)", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-be-no-name";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "impl", "be_changes"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "handoff"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_changes", "server.js"), "export default {};\n");
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_notes.md"), "# Notes\n## API Contracts\n## Shared Types\n## Scope Constraints\n- none\n");
  fs.writeFileSync(
    path.join(releaseRoot, "handoff", "be_to_fe.json"),
    JSON.stringify({
      from_step: "impl_be",
      to_step: "impl_fe",
      be_changes_path: "impl/be_changes",
      api_contracts: [
        { method: "GET", path: "/api/tasks", responseBody: "Task[]", statusCodes: [200] },
        { method: "POST", path: "/api/tasks", requestBody: "{ title: string }", responseBody: "Task", statusCodes: [201, 400] },
      ],
      shared_types: [{ name: "Task", fields: [{ name: "id", type: "string" }] }],
      scope_constraints: ["In-memory storage only"],
    }, null, 2)
  );

  const result = validateCodingTeamHandoff({
    workspaceRoot,
    artifactRoot,
    handoff: {
      from_step: "impl_be",
      to_steps: ["impl_fe"],
      required_artifacts: ["impl/be_changes/server.js", "impl/be_notes.md", "handoff/be_to_fe.json"],
      required_sections: ["api_contracts", "shared_types", "scope_constraints"],
      typed_handoff: {
        file: "handoff/be_to_fe.json",
        required_fields: ["from_step", "to_step", "be_changes_path", "api_contracts", "shared_types", "scope_constraints"],
      },
    },
  });

  assert.equal(result.ok, true, `expected ok but got: ${result.detail}`);
  assert.equal(result.typed_schema_checked, "coding_team_be_to_fe_handoff.schema.json");
});
