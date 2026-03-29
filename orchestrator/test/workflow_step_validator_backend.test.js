import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { validateImplementationDelta } from "../src/domain/workflow_step_validator.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-backend-step-"));
}

test("impl_be validation passes with non-empty be_changes and required notes/handoff", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-1";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "impl", "be_changes"), { recursive: true });
  fs.mkdirSync(path.join(releaseRoot, "handoff"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_changes", "server.js"), "export const ok = true;\n");
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_changes", "package.json"), JSON.stringify({ name: "test-app", version: "1.0.0", main: "server.js" }, null, 2));
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_notes.md"), "# Notes\n");
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
    stepId: "impl_be",
    output: {},
    payload: { artifact_root: artifactRoot },
    workspaceRoot,
  });

  assert.equal(result.ok, true);
  assert.equal(result.be_changes_count, 2);
});

test("impl_be validation fails when be_changes directory is missing", () => {
  const workspaceRoot = makeWorkspace();
  const artifactRoot = "artifacts/release/run-2";
  const releaseRoot = path.join(workspaceRoot, artifactRoot);
  fs.mkdirSync(path.join(releaseRoot, "impl"), { recursive: true });
  fs.writeFileSync(path.join(releaseRoot, "impl", "be_notes.md"), "# Notes\n");

  const result = validateImplementationDelta({
    run: { workflow_id: "coding_team_v0" },
    stepId: "impl_be",
    output: {},
    payload: { artifact_root: artifactRoot },
    workspaceRoot,
  });

  assert.equal(result.ok, false);
  assert.equal(result.code, "STEP_IMPL_BE_ARTIFACTS_MISSING");
});
