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
