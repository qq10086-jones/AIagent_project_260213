import assert from "assert";
import path from "path";
import { fileURLToPath } from "url";
import { buildCodexInvocation, runCodexTask } from "../adapters/codex_adapter.js";

async function testBuildInvocation() {
  const inv = buildCodexInvocation({
    taskPrompt: "fix bug",
    model: "gpt-5-codex",
    codexCommand: ["codex", "exec", "fix bug"],
  });
  assert.strictEqual(inv.command, "codex");
  assert.deepStrictEqual(inv.args, ["exec", "fix bug"]);
  assert.strictEqual(inv.commandSource, "payload.codex_command");
}

async function testRunCodexTaskWithMockCommand() {
  const cmd = [
    process.execPath,
    "-e",
    "console.log('mock codex ok')",
  ];

  const prevKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || "test-key";
  try {
    const result = await runCodexTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "implement calculator",
      model: "mock-model",
      codexCommand: cmd,
      maxRuntimeS: 10,
    });

    assert.strictEqual(result.ok, true, `expected ok=true, got ${JSON.stringify(result)}`);
    assert.strictEqual(result.provider_used, "codex");
    assert.ok(String(result.stdout || "").includes("mock codex ok"));
  } catch (err) {
    if (String(err?.code || "") === "EPERM") {
      console.log("codex adapter spawn test skipped due sandbox EPERM");
      return;
    }
    throw err;
  } finally {
    if (prevKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = prevKey;
  }
}

async function testRunCodexTaskMissingPrompt() {
  const result = await runCodexTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "",
    codexCommand: [process.execPath, "-e", "console.log('x')"],
  });
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_INVALID_INPUT");
}

async function testRunCodexTaskMissingAuth() {
  const prevKey = process.env.OPENAI_API_KEY;
  delete process.env.OPENAI_API_KEY;
  const result = await runCodexTask({
    workspaceRoot: process.cwd(),
    taskPrompt: "do something",
    codexCommand: [process.execPath, "-e", "console.log('x')"],
  });
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.diagnostics.error_code, "E_PROVIDER_UNAVAILABLE");
  assert.ok(String(result.error || "").includes("Codex auth missing"));
  if (prevKey !== undefined) process.env.OPENAI_API_KEY = prevKey;
}

async function main() {
  await testBuildInvocation();
  await testRunCodexTaskWithMockCommand();
  await testRunCodexTaskMissingPrompt();
  await testRunCodexTaskMissingAuth();
  console.log("codex_adapter.test.js: all tests passed");
}

main().catch((err) => {
  console.error("codex_adapter.test.js: failed");
  console.error(err);
  process.exit(1);
});
