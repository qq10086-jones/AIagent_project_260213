import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import { loadRegistryOrThrow } from "../src/registry.js";

const baseRegistryPath = new URL("../../configs/registry/capability_registry.json", import.meta.url);
const baseRegistry = JSON.parse(fs.readFileSync(baseRegistryPath, "utf8"));

test("loadRegistryOrThrow accepts workflow steps with valid depends_on references", () => {
  const registry = structuredClone(baseRegistry);
  registry.workflows.coding_team_v0.steps[2].depends_on = ["arch_design"];
  registry.workflows.coding_team_v0.steps[3].depends_on = ["impl_be"];

  const tmpPath = path.join(os.tmpdir(), `registry-valid-${Date.now()}.json`);
  fs.writeFileSync(tmpPath, JSON.stringify(registry), "utf8");

  assert.doesNotThrow(() => loadRegistryOrThrow(tmpPath));
  fs.unlinkSync(tmpPath);
});

test("loadRegistryOrThrow rejects unknown depends_on references", () => {
  const registry = structuredClone(baseRegistry);
  registry.workflows.coding_team_v0.steps[3].depends_on = ["missing_step"];

  const tmpPath = path.join(os.tmpdir(), `registry-invalid-${Date.now()}.json`);
  fs.writeFileSync(tmpPath, JSON.stringify(registry), "utf8");

  assert.throws(
    () => loadRegistryOrThrow(tmpPath),
    /depends_on unknown step 'missing_step'/,
  );
  fs.unlinkSync(tmpPath);
});
