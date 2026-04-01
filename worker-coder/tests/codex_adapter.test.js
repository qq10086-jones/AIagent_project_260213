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

async function testRunCodexTaskEmitsRuntimeEvents() {
  const cmd = [
    process.execPath,
    "-e",
    "console.log('chunk one'); console.error('progress stderr');",
  ];

  const events = [];
  const prevKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = process.env.OPENAI_API_KEY || "test-key";
  try {
    const result = await runCodexTask({
      workspaceRoot: process.cwd(),
      taskPrompt: "emit runtime events",
      model: "mock-model",
      codexCommand: cmd,
      maxRuntimeS: 10,
      onRuntimeEvent: async (event) => {
        events.push(event);
      },
    });

    assert.strictEqual(result.ok, true, `expected ok=true, got ${JSON.stringify(result)}`);
    assert.ok(events.some((event) => event?.type === "tool_call" && event?.tag === "tool_call"));
    assert.ok(events.some((event) => event?.type === "text_delta" && event?.tag === "agent_message_chunk"));
    assert.ok(events.some((event) => event?.type === "status" && event?.tag === "tool_call_update"));
  } catch (err) {
    if (String(err?.code || "") === "EPERM") {
      console.log("codex adapter runtime-event test skipped due sandbox EPERM");
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
  assert.strictEqual(result.diagnostics.error_code, "E_AUTH_FAILED");
  assert.strictEqual(result.diagnostics.provider_error_class, "AUTH_FAILURE");
  assert.ok(String(result.error || "").includes("Codex auth missing"));
  if (prevKey !== undefined) process.env.OPENAI_API_KEY = prevKey;
}

async function main() {
  await testBuildInvocation();
  await testRunCodexTaskWithMockCommand();
  await testRunCodexTaskEmitsRuntimeEvents();
  await testRunCodexTaskMissingPrompt();
  await testRunCodexTaskMissingAuth();
  console.log("codex_adapter.test.js: all tests passed");
}

main().catch((err) => {
  console.error("codex_adapter.test.js: failed");
  console.error(err);
  process.exit(1);
});
