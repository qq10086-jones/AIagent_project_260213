import assert from "node:assert/strict";

import {
  buildRequiredConfigFiles,
  runConfigPreflight,
  assertConfigPreflight,
} from "../src/config_preflight.js";

function main() {
  const files = buildRequiredConfigFiles({
    runtimeConfigPath: "/app/configs/runtime/runtime_defaults.json",
    workspaceRoot: "/app",
  });
  assert.equal(files.length, 4);
  assert.equal(files[0].id, "runtime_defaults");
  assert.equal(files[1].id, "llm_providers");
  assert.equal(files[2].id, "llm_role_policy");
  assert.equal(files[3].id, "context_budget_policy");

  const okResult = runConfigPreflight({
    runtimeConfigPath: "/app/configs/runtime/runtime_defaults.json",
    workspaceRoot: "/app",
    existsSync: () => true,
  });
  assert.equal(okResult.ok, true);
  assert.equal(okResult.missing.length, 0);

  const failResult = runConfigPreflight({
    runtimeConfigPath: "/app/configs/runtime/runtime_defaults.json",
    workspaceRoot: "/app",
    existsSync: (filePath) =>
      !String(filePath).includes("llm_providers.json") &&
      !String(filePath).includes("context_budget_policy.json"),
  });
  assert.equal(failResult.ok, false);
  assert.equal(failResult.error_code, "CONFIG_PREFLIGHT_FAILED");
  assert.equal(failResult.missing.length, 2);
  assert.equal(failResult.missing[0].id, "llm_providers");
  assert.equal(failResult.missing[1].id, "context_budget_policy");

  assert.throws(
    () => assertConfigPreflight({
      runtimeConfigPath: "/app/configs/runtime/runtime_defaults.json",
      workspaceRoot: "/app",
      existsSync: () => false,
    }),
    /CONFIG_PREFLIGHT_FAILED/,
  );

  console.log("config_preflight.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("config_preflight.test.js: failed");
  console.error(err);
  process.exit(1);
}
