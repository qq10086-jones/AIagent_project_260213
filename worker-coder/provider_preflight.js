import fs from "fs";
import os from "os";
import path from "path";

function hasOpenCodeAuth(env = process.env) {
  return Boolean(String(env.OPENCODE_API_KEY || env.OPENCODE_ZEN_API_KEY || "").trim())
    || fs.existsSync(path.join(os.homedir(), ".local", "share", "opencode", "auth.json"));
}

function normalizeProvider(value) {
  return String(value || "").trim().toLowerCase();
}

function normalizeModel(value) {
  return String(value || "").trim();
}

function validateCodexLane({ laneName, provider, env = process.env }) {
  const issues = [];
  if (normalizeProvider(provider) !== "codex") return issues;
  const hasApiKey = Boolean(String(env.OPENAI_API_KEY || "").trim());
  const hasAuthFile = fs.existsSync("/root/.codex/auth.json");
  if (!hasApiKey && !hasAuthFile) {
    issues.push({
      severity: "error",
      lane: laneName,
      code: "CODEX_AUTH_MISSING",
      message: `lane '${laneName}' requires OPENAI_API_KEY or /root/.codex/auth.json`,
    });
  }
  return issues;
}

function validateOpenCodeLane({ laneName, provider, model, env = process.env }) {
  const issues = [];
  const safeProvider = normalizeProvider(provider);
  const safeModel = normalizeModel(model);
  const modelProvider = normalizeProvider(safeModel.split("/")[0] || "");

  if (safeProvider !== "opencode") return issues;
  if (!safeModel) {
    issues.push({
      severity: "error",
      lane: laneName,
      code: "MODEL_MISSING",
      message: "opencode lane is missing a provider/model ref",
    });
    return issues;
  }
  if (modelProvider === "dashscope") {
    issues.push({
      severity: "error",
      lane: laneName,
      code: "MODEL_PROVIDER_MISMATCH",
      message: `opencode lane uses unsupported model ref '${safeModel}'`,
    });
    return issues;
  }
  if ((modelProvider === "opencode" || modelProvider === "opencode-go") && !hasOpenCodeAuth(env)) {
    issues.push({
      severity: "error",
      lane: laneName,
      code: "OPENCODE_AUTH_MISSING",
      message: `lane '${laneName}' requires OPENCODE_API_KEY/OPENCODE_ZEN_API_KEY or opencode auth login`,
    });
  }
  if ((modelProvider === "alibaba-coding-plan" || modelProvider === "alibaba-coding-plan-cn")
    && !String(env.ALIBABA_CODING_PLAN_API_KEY || "").trim()) {
    issues.push({
      severity: "error",
      lane: laneName,
      code: "ALIBABA_AUTH_MISSING",
      message: `lane '${laneName}' requires ALIBABA_CODING_PLAN_API_KEY`,
    });
  }
  return issues;
}

export function validateRuntimePreflight({
  defaultProvider = "",
  defaultModel = "",
  defaultExecutionLane = "",
  runtimeCoderConfig = {},
  env = process.env,
}) {
  const issues = [];
  const lanes = runtimeCoderConfig && typeof runtimeCoderConfig.execution_lanes === "object"
    ? runtimeCoderConfig.execution_lanes
    : {};

  if (defaultExecutionLane && lanes[defaultExecutionLane]) {
    const lane = lanes[defaultExecutionLane] || {};
    const laneProvider = normalizeProvider(lane.provider || defaultProvider);
    if (laneProvider === "codex") {
      issues.push(...validateCodexLane({ laneName: defaultExecutionLane, provider: laneProvider, env }));
    } else {
      issues.push(...validateOpenCodeLane({
        laneName: defaultExecutionLane,
        provider: laneProvider,
        model: lane.model || defaultModel,
        env,
      }));
    }
  } else {
    const laneProvider = normalizeProvider(defaultProvider);
    if (laneProvider === "codex") {
      issues.push(...validateCodexLane({ laneName: defaultExecutionLane || "default", provider: laneProvider, env }));
    } else {
      issues.push(...validateOpenCodeLane({
        laneName: defaultExecutionLane || "default",
        provider: laneProvider,
        model: defaultModel,
        env,
      }));
    }
  }

  return {
    ok: issues.length === 0,
    issues,
  };
}
