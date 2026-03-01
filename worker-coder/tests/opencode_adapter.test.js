import assert from "assert";
import { buildOpenCodeInvocation, runOpenCodeTask } from "../adapters/opencode_adapter.js";

function shouldSkipSpawn(result) {
  return String(result?.error || "").includes("spawn EPERM");
}

async function testBuildInvocation() {
  const inv = buildOpenCodeInvocation({
    taskPrompt: "fix bug",
    model: "minimax-m2.5",
    opencodeCommand: ["opencode", "run", "fix bug"],
  });
  assert.strictEqual(inv.command, "opencode");
  assert.deepStrictEqual(inv.args, ["run", "fix bug"]);
  assert.strictEqual(inv.commandSource, "payload.opencode_command");
}

async function testRunOpenCodeTaskWithMockCommand() {
  const cmd = [
    process.execPath,
    "-e",
    "console.log('mock opencode ok')",
  ];
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "implement calculator",
    model: "mock-model",
    opencodeCommand: cmd,
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode adapter spawn test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, true, `expected ok=true, got ${JSON.stringify(result)}`);
  assert.strictEqual(result.provider_used, "opencode");
  assert.ok(String(result.stdout || "").includes("mock opencode ok"));
}

async function testRunOpenCodeTaskMissingPrompt() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "",
    opencodeCommand: [process.execPath, "-e", "console.log('x')"],
  });
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_INVALID_INPUT");
}

async function testRunOpenCodeTaskTimeout() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "timeout",
    opencodeCommand: [process.execPath, "-e", "setTimeout(() => {}, 5000)"],
    maxRuntimeS: 1,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode timeout test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_TIMEOUT");
}

async function testRunOpenCodeTaskApplyFailed() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "apply patch",
    opencodeCommand: [process.execPath, "-e", "console.error('apply failed'); process.exit(2)"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode apply-failed test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_APPLY_FAILED");
}

async function testRunOpenCodeTaskProviderUnavailable() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "provider unavailable",
    opencodeCommand: ["opencode_command_not_found_12345", "run", "x"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode provider-unavailable test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_PROVIDER_UNAVAILABLE");
}

async function main() {
  await testBuildInvocation();
  await testRunOpenCodeTaskWithMockCommand();
  await testRunOpenCodeTaskMissingPrompt();
  await testRunOpenCodeTaskTimeout();
  await testRunOpenCodeTaskApplyFailed();
  await testRunOpenCodeTaskProviderUnavailable();
  console.log("opencode_adapter.test.js: all tests passed");
}

main().catch((err) => {
  console.error("opencode_adapter.test.js: failed");
  console.error(err);
  process.exit(1);
});
