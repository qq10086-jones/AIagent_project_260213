import assert from "assert";
import { validateRuntimePreflight } from "../provider_preflight.js";

function testRejectsDashScopeOnOpenCodeLane() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "dashscope/qwen-flash-2025-07-28",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "dashscope/qwen-flash-2025-07-28",
        },
      },
    },
    env: {},
  });
  assert.equal(result.ok, false);
  assert.equal(result.issues[0].code, "MODEL_PROVIDER_MISMATCH");
}

function testRequiresAlibabaCredential() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "alibaba-coding-plan/qwen3-coder-plus",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "alibaba-coding-plan/qwen3-coder-plus",
        },
      },
    },
    env: {},
  });
  assert.equal(result.ok, false);
  assert.equal(result.issues[0].code, "ALIBABA_AUTH_MISSING");
}

function testPassesWhenAlibabaCredentialPresent() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "alibaba-coding-plan/qwen3-coder-plus",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "alibaba-coding-plan/qwen3-coder-plus",
        },
      },
    },
    env: {
      ALIBABA_CODING_PLAN_API_KEY: "test-key",
    },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function main() {
  testRejectsDashScopeOnOpenCodeLane();
  testRequiresAlibabaCredential();
  testPassesWhenAlibabaCredentialPresent();
  console.log("provider_preflight.test.js: all tests passed");
}

main();
