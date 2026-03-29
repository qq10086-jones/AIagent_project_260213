import assert from "assert";
import { validateRuntimePreflight } from "../provider_preflight.js";

function testPassesWhenDashScopeCredentialPresent() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "dashscope/qwen-plus-2025-04-28",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "dashscope/qwen-plus-2025-04-28",
        },
      },
    },
    env: {
      DASHSCOPE_API_KEY: "test-key",
    },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function testRequiresDashScopeCredential() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "dashscope/qwen-plus-2025-04-28",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "dashscope/qwen-plus-2025-04-28",
        },
      },
    },
    env: {},
  });
  assert.equal(result.ok, false);
  assert.equal(result.issues[0].code, "DASHSCOPE_AUTH_MISSING");
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

function testPassesWhenQwenCredentialPresentForAlibabaLane() {
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
      QWEN_API_KEY: "test-key",
    },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function testPassesForConfiguredLocalOllamaLane() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "ollama/glm-4.7-flash:latest",
    defaultExecutionLane: "stable_local_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_local_lane: {
          provider: "opencode",
          model: "ollama/glm-4.7-flash:latest",
        },
      },
    },
    env: {
      OLLAMA_BASE_URL: "http://host.docker.internal:11434",
    },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function testRequiresMiniMaxCredential() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "minimax/MiniMax-M2.5",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "minimax/MiniMax-M2.5",
        },
      },
    },
    env: {},
  });
  assert.equal(result.ok, false);
  assert.equal(result.issues[0].code, "MINIMAX_AUTH_MISSING");
}

function testPassesWhenMiniMaxCredentialPresent() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "minimax/MiniMax-M2.5",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "minimax/MiniMax-M2.5",
        },
      },
    },
    env: { MINIMAX_API_KEY: "test-key" },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function testPassesWhenMiniMaxCodingPlanCredentialPresent() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "minimax-coding-plan/MiniMax-M2.7",
    defaultExecutionLane: "stable_cloud_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "minimax-coding-plan/MiniMax-M2.7",
        },
      },
    },
    env: { MINIMAX_API_KEY: "test-key" },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function testCodexLaneRequiresOpenAIKey() {
  const result = validateRuntimePreflight({
    defaultProvider: "codex",
    defaultModel: "codex-mini-latest",
    defaultExecutionLane: "stable_codex_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_codex_lane: {
          provider: "codex",
          model: "codex-mini-latest",
        },
      },
    },
    env: {},
  });
  assert.equal(result.ok, false);
  assert.equal(result.issues[0].code, "CODEX_AUTH_MISSING");
}

function testCodexLanePassesWhenOpenAIKeyPresent() {
  const result = validateRuntimePreflight({
    defaultProvider: "codex",
    defaultModel: "codex-mini-latest",
    defaultExecutionLane: "stable_codex_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_codex_lane: {
          provider: "codex",
          model: "codex-mini-latest",
        },
      },
    },
    env: { OPENAI_API_KEY: "sk-test-key" },
  });
  assert.equal(result.ok, true);
  assert.equal(result.issues.length, 0);
}

function testFailsWhenDefaultExecutionLaneMissing() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "minimax-coding-plan/MiniMax-M2.7",
    defaultExecutionLane: "missing_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_cloud_lane: {
          provider: "opencode",
          model: "minimax-coding-plan/MiniMax-M2.7",
        },
      },
    },
    env: { MINIMAX_API_KEY: "test-key" },
  });
  assert.equal(result.ok, false);
  assert.equal(result.fatal, true);
  assert.equal(result.issues[0].code, "EXECUTION_LANE_UNKNOWN");
}

function testFailsWhenDefaultModelConflictsWithLaneModel() {
  const result = validateRuntimePreflight({
    defaultProvider: "opencode",
    defaultModel: "minimax-coding-plan/MiniMax-M2.7",
    defaultExecutionLane: "stable_dashscope_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_dashscope_lane: {
          provider: "opencode",
          model: "dashscope/qwen-plus-2025-04-28",
        },
      },
    },
    env: { DASHSCOPE_API_KEY: "test-key" },
  });
  assert.equal(result.ok, false);
  assert.equal(result.fatal, true);
  assert.equal(result.issues[0].code, "DEFAULT_MODEL_LANE_MISMATCH");
}

function testFailsWhenDefaultProviderConflictsWithLaneProvider() {
  const result = validateRuntimePreflight({
    defaultProvider: "codex",
    defaultModel: "codex-mini-latest",
    defaultExecutionLane: "stable_dashscope_lane",
    runtimeCoderConfig: {
      execution_lanes: {
        stable_dashscope_lane: {
          provider: "opencode",
          model: "dashscope/qwen-plus-2025-04-28",
        },
      },
    },
    env: { DASHSCOPE_API_KEY: "test-key", OPENAI_API_KEY: "sk-test-key" },
  });
  assert.equal(result.ok, false);
  assert.equal(result.fatal, true);
  assert.equal(result.issues[0].code, "DEFAULT_PROVIDER_LANE_MISMATCH");
}

function main() {
  testPassesWhenDashScopeCredentialPresent();
  testRequiresDashScopeCredential();
  testRequiresAlibabaCredential();
  testPassesWhenAlibabaCredentialPresent();
  testPassesWhenQwenCredentialPresentForAlibabaLane();
  testPassesForConfiguredLocalOllamaLane();
  testRequiresMiniMaxCredential();
  testPassesWhenMiniMaxCredentialPresent();
  testPassesWhenMiniMaxCodingPlanCredentialPresent();
  testCodexLaneRequiresOpenAIKey();
  testCodexLanePassesWhenOpenAIKeyPresent();
  testFailsWhenDefaultExecutionLaneMissing();
  testFailsWhenDefaultModelConflictsWithLaneModel();
  testFailsWhenDefaultProviderConflictsWithLaneProvider();
  console.log("provider_preflight.test.js: all tests passed");
}

main();
