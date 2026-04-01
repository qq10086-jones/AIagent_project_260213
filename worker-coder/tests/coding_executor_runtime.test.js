import assert from "assert";
import { executeCodingAdapter } from "../coding_executor_runtime.js";

function shouldSkipSpawn(result) {
  return String(result?.error || "").includes("spawn EPERM");
}

function buildRequest(executionLane = "") {
  return {
    adapter_type: "coding_executor",
    provider: "auto",
    task_type: "coding_execution",
    payload: {
      task_prompt: "implement stable lane test",
      execution_lane: executionLane,
    },
  };
}

async function testStableLaneResolution() {
  const prevApiKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  try {
    const result = await executeCodingAdapter({
      workspaceRoot: process.cwd(),
      adapterRequest: buildRequest("stable_local_lane"),
      provider: "auto",
      model: null,
      runtimeCoderConfig: {
        execution_lane_default: "stable_local_lane",
        execution_lanes: {
          stable_local_lane: {
            provider: "opencode",
            model: "glm-4.7-flash:latest",
          },
        },
      },
      opencodeCommand: [process.execPath, "-e", "console.log('stable lane ok')"],
      allowProviderFallback: false,
      maxRuntimeS: 10,
    });
    if (shouldSkipSpawn(result)) {
      console.log("coding executor stable-lane test skipped due sandbox EPERM");
      return;
    }
    assert.strictEqual(result.ok, true, JSON.stringify(result));
    assert.strictEqual(result.execution_lane, "stable_local_lane");
    assert.strictEqual(result.provider_used, "opencode");
    assert.strictEqual(result.model_used, "glm-4.7-flash:latest");
    assert.strictEqual(result.diagnostics.model_provider, "opencode");
    assert.strictEqual(result.diagnostics.model_name, "glm-4.7-flash:latest");
    assert.strictEqual(result.diagnostics.fallback_taken, false);
  } finally {
    if (prevApiKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = prevApiKey;
  }
}

async function testNoSilentFallback() {
  const prevApiKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  try {
    const result = await executeCodingAdapter({
      workspaceRoot: process.cwd(),
      adapterRequest: buildRequest("stable_local_lane"),
      provider: "auto",
      model: null,
      runtimeCoderConfig: {
        execution_lane_default: "stable_local_lane",
        execution_lanes: {
          stable_local_lane: {
            provider: "opencode",
            model: "glm-4.7-flash:latest",
          },
        },
      },
      opencodeCommand: ["opencode_command_not_found_12345", "run", "x"],
      codexCommand: [process.execPath, "-e", "console.log('codex fallback should stay disabled')"],
      allowProviderFallback: false,
      maxRuntimeS: 10,
    });
    if (shouldSkipSpawn(result)) {
      console.log("coding executor no-fallback test skipped due sandbox EPERM");
      return;
    }
    assert.strictEqual(result.ok, false, JSON.stringify(result));
    assert.strictEqual(result.provider_used, "opencode");
    assert.strictEqual(result.diagnostics.fallback_taken, false);
    assert.strictEqual(result.diagnostics.fallback_target, null);
    assert.strictEqual(result.diagnostics.error_code, "E_PROVIDER_UNAVAILABLE");
  } finally {
    if (prevApiKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = prevApiKey;
  }
}

async function testExplicitFallback() {
  const prevApiKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  try {
    const result = await executeCodingAdapter({
      workspaceRoot: process.cwd(),
      adapterRequest: buildRequest("stable_local_lane"),
      provider: "auto",
      model: null,
      runtimeCoderConfig: {
        execution_lane_default: "stable_local_lane",
        execution_lanes: {
          stable_local_lane: {
            provider: "opencode",
            model: "glm-4.7-flash:latest",
          },
        },
      },
      opencodeCommand: ["opencode_command_not_found_12345", "run", "x"],
      codexCommand: [process.execPath, "-e", "console.log('explicit fallback ok')"],
      allowProviderFallback: true,
      maxRuntimeS: 10,
    });
    if (shouldSkipSpawn(result)) {
      console.log("coding executor explicit-fallback test skipped due sandbox EPERM");
      return;
    }
    assert.strictEqual(result.ok, true, JSON.stringify(result));
    assert.strictEqual(result.provider_used, "codex");
    assert.strictEqual(result.execution_lane, "stable_local_lane");
    assert.strictEqual(result.diagnostics.fallback_taken, true);
    assert.strictEqual(result.diagnostics.fallback_from, "opencode");
    assert.strictEqual(result.diagnostics.fallback_target, "codex");
  } finally {
    if (prevApiKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = prevApiKey;
  }
}

async function testRuntimeEventsPassThroughToProvider() {
  const prevApiKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  const events = [];
  try {
    const result = await executeCodingAdapter({
      workspaceRoot: process.cwd(),
      adapterRequest: buildRequest(""),
      provider: "codex",
      model: "mock-model",
      runtimeCoderConfig: {},
      codexCommand: [process.execPath, "-e", "console.log('runtime pass-through ok'); console.error('runtime progress');"],
      allowProviderFallback: false,
      maxRuntimeS: 10,
      onRuntimeEvent: async (event) => {
        events.push(event);
      },
    });
    assert.strictEqual(result.ok, true, JSON.stringify(result));
    assert.ok(events.some((event) => event?.type === "tool_call"));
    assert.ok(events.some((event) => event?.type === "text_delta"));
    assert.ok(events.some((event) => event?.type === "status"));
  } catch (err) {
    if (String(err?.code || "") === "EPERM" || String(err?.message || "").includes("spawn EPERM")) {
      console.log("coding executor runtime-event test skipped due sandbox EPERM");
      return;
    }
    throw err;
  } finally {
    if (prevApiKey === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = prevApiKey;
  }
}

async function main() {
  await testStableLaneResolution();
  await testNoSilentFallback();
  await testExplicitFallback();
  await testRuntimeEventsPassThroughToProvider();
  console.log("coding_executor_runtime.test.js: all tests passed");
}

main().catch((err) => {
  console.error("coding_executor_runtime.test.js: failed");
  console.error(err);
  process.exit(1);
});
