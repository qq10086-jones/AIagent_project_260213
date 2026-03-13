import assert from "assert";
import { buildOpenCodeInvocation, runOpenCodeTask } from "../adapters/opencode_adapter.js";

function shouldSkipSpawn(result) {
  return String(result?.error || "").includes("spawn EPERM");
}

async function testBuildInvocation() {
  const inv = buildOpenCodeInvocation({
    taskPrompt: "fix bug",
    model: "qwen3-coder-plus-2025-07-22",
    opencodeCommand: ["opencode", "run", "fix bug", "--model", "{{model}}"],
  });
  assert.strictEqual(inv.command, "opencode");
  assert.deepStrictEqual(inv.args, ["run", "fix bug", "--model", "alibaba-coding-plan/qwen3-coder-plus"]);
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

async function testRunOpenCodeTaskAuthFailed() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "auth failed",
    opencodeCommand: [process.execPath, "-e", "console.error('invalid access token or token expired'); process.exit(1)"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode auth-failed test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
  assert.strictEqual(result.diagnostics.provider_error_class, "AUTH_FAILURE");
}

async function testRunOpenCodeTaskAuthFailedEvenWhenExitZero() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "auth failed with zero exit",
    opencodeCommand: [process.execPath, "-e", "console.error('Incorrect API key provided'); process.exit(0)"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode auth-failed-zero-exit test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
  assert.strictEqual(result.error, "OpenCode authentication failed");
}

async function testRunOpenCodeTaskRejectsDashScopeModelRef() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "invalid provider/model pairing",
    model: "dashscope/qwen-flash-2025-07-28",
    opencodeCommand: [process.execPath, "-e", "console.log('should not run')"],
    maxRuntimeS: 10,
  });
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_PROVIDER_CONFIG");
  assert.match(String(result.error || ""), /does not accept dashscope/i);
}

async function testRunOpenCodeTaskRequiresAlibabaCredential() {
  const prevAlibabaKey = process.env.ALIBABA_CODING_PLAN_API_KEY;
  delete process.env.ALIBABA_CODING_PLAN_API_KEY;
  try {
    const result = await runOpenCodeTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "missing alibaba credential",
      model: "alibaba-coding-plan/qwen3-coder-plus",
      opencodeCommand: [process.execPath, "-e", "console.log('should not run')"],
      maxRuntimeS: 10,
    });
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
    assert.match(String(result.error || ""), /ALIBABA_CODING_PLAN_API_KEY/i);
  } finally {
    if (prevAlibabaKey === undefined) delete process.env.ALIBABA_CODING_PLAN_API_KEY;
    else process.env.ALIBABA_CODING_PLAN_API_KEY = prevAlibabaKey;
  }
}

async function testRunOpenCodeTaskRequiresOpenCodeCredential() {
  const prevOpenCodeKey = process.env.OPENCODE_API_KEY;
  const prevOpenCodeZenKey = process.env.OPENCODE_ZEN_API_KEY;
  delete process.env.OPENCODE_API_KEY;
  delete process.env.OPENCODE_ZEN_API_KEY;
  try {
    const result = await runOpenCodeTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "missing opencode credential",
      model: "opencode-go/glm-5",
      opencodeCommand: [process.execPath, "-e", "console.log('should not run')"],
      maxRuntimeS: 10,
    });
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
    assert.match(String(result.error || ""), /OPENCODE_API_KEY|opencode auth login/i);
  } finally {
    if (prevOpenCodeKey === undefined) delete process.env.OPENCODE_API_KEY;
    else process.env.OPENCODE_API_KEY = prevOpenCodeKey;
    if (prevOpenCodeZenKey === undefined) delete process.env.OPENCODE_ZEN_API_KEY;
    else process.env.OPENCODE_ZEN_API_KEY = prevOpenCodeZenKey;
  }
}
async function main() {
  await testBuildInvocation();
  await testRunOpenCodeTaskWithMockCommand();
  await testRunOpenCodeTaskMissingPrompt();
  await testRunOpenCodeTaskTimeout();
  await testRunOpenCodeTaskApplyFailed();
  await testRunOpenCodeTaskProviderUnavailable();
  await testRunOpenCodeTaskAuthFailed();
  await testRunOpenCodeTaskAuthFailedEvenWhenExitZero();
  await testRunOpenCodeTaskRejectsDashScopeModelRef();
  await testRunOpenCodeTaskRequiresAlibabaCredential();
  await testRunOpenCodeTaskRequiresOpenCodeCredential();
  console.log("opencode_adapter.test.js: all tests passed");
}

main().catch((err) => {
  console.error("opencode_adapter.test.js: failed");
  console.error(err);
  process.exit(1);
});


