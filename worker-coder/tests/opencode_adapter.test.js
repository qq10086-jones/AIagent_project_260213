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
  assert.deepStrictEqual(inv.args, ["run", "fix bug", "--model", "dashscope/qwen-plus-2025-04-28"]);
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

async function testRunOpenCodeTaskRejectsAuthLikeStderrWhenExitZero() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "auth-like stderr with zero exit",
    opencodeCommand: [process.execPath, "-e", "console.error('Incorrect API key provided'); console.log('completed'); process.exit(0)"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode auth-like-stderr-zero-exit test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false, JSON.stringify(result));
  assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
  assert.strictEqual(result.diagnostics.provider_error_class, "AUTH_FAILURE");
  assert.ok(String(result.stdout || "").includes("completed"));
}

async function testRunOpenCodeTaskRejectsMiniMaxLoginFailMessage() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "minimax auth-like stderr with zero exit",
    opencodeCommand: [process.execPath, "-e", "console.error(\"Error: login fail: Please carry the API secret key in the 'Authorization' field of the request header\"); process.exit(0)"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode minimax-login-fail test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, false, JSON.stringify(result));
  assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
  assert.strictEqual(result.diagnostics.provider_error_class, "AUTH_FAILURE");
}

async function testRunOpenCodeTaskAllowsOllamaModelRef() {
  const result = await runOpenCodeTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "local ollama probe",
    model: "ollama/glm-4.7-flash:latest",
    opencodeCommand: [process.execPath, "-e", "console.log('ollama ok')"],
    maxRuntimeS: 10,
  });
  if (shouldSkipSpawn(result)) {
    console.log("opencode ollama-model test skipped due sandbox EPERM");
    return;
  }
  assert.strictEqual(result.ok, true, JSON.stringify(result));
  assert.strictEqual(result.model_used, "ollama/glm-4.7-flash:latest");
}

async function testRunOpenCodeTaskAcceptsQwenCredentialForDashScopeModel() {
  const prevQwenKey = process.env.QWEN_API_KEY;
  const prevDashKey = process.env.DASH_SCOPE_API_KEY;
  delete process.env.DASH_SCOPE_API_KEY;
  process.env.QWEN_API_KEY = "test-qwen-key";
  try {
    const result = await runOpenCodeTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "dashscope model with qwen credential",
      model: "dashscope/qwen-plus-2025-04-28",
      opencodeCommand: [process.execPath, "-e", "console.log('qwen credential ok')"],
      maxRuntimeS: 10,
    });
    if (shouldSkipSpawn(result)) {
      console.log("opencode qwen-credential test skipped due sandbox EPERM");
      return;
    }
    assert.strictEqual(result.ok, true, JSON.stringify(result));
  } finally {
    if (prevQwenKey === undefined) delete process.env.QWEN_API_KEY;
    else process.env.QWEN_API_KEY = prevQwenKey;
    if (prevDashKey === undefined) delete process.env.DASH_SCOPE_API_KEY;
    else process.env.DASH_SCOPE_API_KEY = prevDashKey;
  }
}

async function testRunOpenCodeTaskRequiresDashScopeCredential() {
  const prevAlibabaKey = process.env.ALIBABA_CODING_PLAN_API_KEY;
  const prevQwenKey = process.env.QWEN_API_KEY;
  const prevDashKey = process.env.DASH_SCOPE_API_KEY;
  delete process.env.ALIBABA_CODING_PLAN_API_KEY;
  delete process.env.QWEN_API_KEY;
  delete process.env.DASH_SCOPE_API_KEY;
  try {
    const result = await runOpenCodeTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "missing dashscope credential",
      model: "dashscope/qwen-plus-2025-04-28",
      opencodeCommand: [process.execPath, "-e", "console.log('should not run')"],
      maxRuntimeS: 10,
    });
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
    assert.match(String(result.error || ""), /DASHSCOPE_API_KEY|DASH_SCOPE_API_KEY|QWEN_API_KEY|ALIBABA_CODING_PLAN_API_KEY/i);
  } finally {
    if (prevAlibabaKey === undefined) delete process.env.ALIBABA_CODING_PLAN_API_KEY;
    else process.env.ALIBABA_CODING_PLAN_API_KEY = prevAlibabaKey;
    if (prevQwenKey === undefined) delete process.env.QWEN_API_KEY;
    else process.env.QWEN_API_KEY = prevQwenKey;
    if (prevDashKey === undefined) delete process.env.DASH_SCOPE_API_KEY;
    else process.env.DASH_SCOPE_API_KEY = prevDashKey;
  }
}

async function testRunOpenCodeTaskAcceptsDashScopeCredential() {
  const prevQwenKey = process.env.QWEN_API_KEY;
  const prevDashScopeKey = process.env.DASHSCOPE_API_KEY;
  const prevDashKey = process.env.DASH_SCOPE_API_KEY;
  delete process.env.QWEN_API_KEY;
  process.env.DASHSCOPE_API_KEY = "test-dashscope-key";
  try {
    const result = await runOpenCodeTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "dashscope model with dashscope credential",
      model: "dashscope/qwen-plus-2025-04-28",
      opencodeCommand: [process.execPath, "-e", "console.log('dashscope credential ok')"],
      maxRuntimeS: 10,
    });
    if (shouldSkipSpawn(result)) {
      console.log("opencode dashscope-credential test skipped due sandbox EPERM");
      return;
    }
    assert.strictEqual(result.ok, true, JSON.stringify(result));
  } finally {
    if (prevQwenKey === undefined) delete process.env.QWEN_API_KEY;
    else process.env.QWEN_API_KEY = prevQwenKey;
    if (prevDashScopeKey === undefined) delete process.env.DASHSCOPE_API_KEY;
    else process.env.DASHSCOPE_API_KEY = prevDashScopeKey;
    if (prevDashKey === undefined) delete process.env.DASH_SCOPE_API_KEY;
    else process.env.DASH_SCOPE_API_KEY = prevDashKey;
  }
}

async function testRunOpenCodeTaskRequiresOpenCodeCredential() {
  const prevOpenCodeKey = process.env.OPENCODE_API_KEY;
  const prevOpenCodeZenKey = process.env.OPENCODE_ZEN_API_KEY;
  const prevHome = process.env.HOME;
  const prevUserProfile = process.env.USERPROFILE;
  const prevHomeDrive = process.env.HOMEDRIVE;
  const prevHomePath = process.env.HOMEPATH;
  delete process.env.OPENCODE_API_KEY;
  delete process.env.OPENCODE_ZEN_API_KEY;
  process.env.HOME = "C:\\opencode-auth-missing-home";
  process.env.USERPROFILE = "C:\\opencode-auth-missing-home";
  process.env.HOMEDRIVE = "C:";
  process.env.HOMEPATH = "\\opencode-auth-missing-home";
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
    if (prevHome === undefined) delete process.env.HOME;
    else process.env.HOME = prevHome;
    if (prevUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = prevUserProfile;
    if (prevHomeDrive === undefined) delete process.env.HOMEDRIVE;
    else process.env.HOMEDRIVE = prevHomeDrive;
    if (prevHomePath === undefined) delete process.env.HOMEPATH;
    else process.env.HOMEPATH = prevHomePath;
  }
}
async function testRunOpenCodeTaskRequiresMiniMaxCredential() {
  const prevMiniMaxKey = process.env.MINIMAX_API_KEY;
  delete process.env.MINIMAX_API_KEY;
  try {
    const result = await runOpenCodeTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "missing minimax credential",
      model: "minimax/MiniMax-M2.5",
      opencodeCommand: [process.execPath, "-e", "console.log('should not run')"],
      maxRuntimeS: 10,
    });
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
    assert.match(String(result.error || ""), /MINIMAX_API_KEY/i);
  } finally {
    if (prevMiniMaxKey === undefined) delete process.env.MINIMAX_API_KEY;
    else process.env.MINIMAX_API_KEY = prevMiniMaxKey;
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
  await testRunOpenCodeTaskRejectsAuthLikeStderrWhenExitZero();
  await testRunOpenCodeTaskRejectsMiniMaxLoginFailMessage();
  await testRunOpenCodeTaskAllowsOllamaModelRef();
  await testRunOpenCodeTaskAcceptsQwenCredentialForDashScopeModel();
  await testRunOpenCodeTaskRequiresDashScopeCredential();
  await testRunOpenCodeTaskAcceptsDashScopeCredential();
  await testRunOpenCodeTaskRequiresOpenCodeCredential();
  await testRunOpenCodeTaskRequiresMiniMaxCredential();
  console.log("opencode_adapter.test.js: all tests passed");
}

main().catch((err) => {
  console.error("opencode_adapter.test.js: failed");
  console.error(err);
  process.exit(1);
});



