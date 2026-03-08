import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { validateImplementationDelta } from "../src/domain/workflow_step_validator.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-frontend-step-"));
}

test("impl_fe validation passes with non-empty fe_changes, notes, and upstream be handoff", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-fe-1";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "impl", "fe_changes"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "handoff"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "impl", "fe_changes", "app.js"), "export const ui = true;\n");
  fs.writeFileSync(path.join(releaseRoot, "impl", "fe_notes.md"), "# FE Notes\n");
  fs.writeFileSync(
    path.join(releaseRoot, "handoff", "be_to_fe.json"),
    JSON.stringify({
      from_step: "impl_be",
      to_step: "impl_fe",
      be_changes_path: "impl/be_changes",
      api_contracts: [{ name: "List Customers", method: "GET", path: "/api/customers" }],
      shared_types: [{ name: "Customer" }],
      scope_constraints: ["Auth not implemented"],
    }, null, 2)
  );

  const result = validateImplementationDelta({
    run: { workflow_id: "coding_team_v0" },
    stepId: "impl_fe",
    output: {},
    payload: { artifact_root: artifactRoot },
    workspaceRoot,
  });

  assert.equal(result.ok, true);
  assert.equal(result.fe_changes_count, 1);
});

test("impl_fe validation fails when upstream be handoff is missing", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-fe-2";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "impl", "fe_changes"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "impl", "fe_changes", "app.js"), "export const ui = true;\n");
  fs.writeFileSync(path.join(releaseRoot, "impl", "fe_notes.md"), "# FE Notes\n");

  const result = validateImplementationDelta({
    run: { workflow_id: "coding_team_v0" },
    stepId: "impl_fe",
    output: {},
    payload: { artifact_root: artifactRoot },
    workspaceRoot,
  });

  assert.equal(result.ok, false);
  assert.equal(result.code, "STEP_IMPL_FE_HANDOFF_MISSING");
});
