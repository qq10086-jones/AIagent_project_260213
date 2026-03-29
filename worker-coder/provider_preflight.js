import fs from "fs";
import os from "os";
import path from "path";

function hasOpenCodeAuth(env = process.env) {
  return Boolean(String(env.OPENCODE_API_KEY || env.OPENCODE_ZEN_API_KEY || "").trim())
    || fs.existsSync(path.join(os.homedir(), ".local", "share", "opencode", "auth.json"));
}

function hasMiniMaxKey(env = process.env) {
  if (String(env.MINIMAX_API_KEY || "").trim()) return true;
  const candidates = [
    env.OPENCODE_JSON_PATH,
    path.join(process.cwd(), "opencode.json"),
    "/app/opencode.json",
  ].filter(Boolean);
  for (const p of candidates) {
    try {
      const cfg = JSON.parse(fs.readFileSync(p, "utf8"));
      const key = String(
        cfg?.provider?.["minimax-coding-plan"]?.options?.apiKey ||
        cfg?.provider?.minimax?.options?.apiKey || ""
      ).trim();
      if (key && !key.startsWith("${")) return true;
    } catch {}
  }
  return false;
}

function normalizeProvider(value) {
  return String(value || "").trim().toLowerCase();
}

function normalizeModel(value) {
  return String(value || "").trim();
}

function buildIssue({ severity = "error", lane, code, message, fatal = false }) {
  return { severity, lane, code, message, fatal };
}

function validateCodexLane({ laneName, provider, env = process.env }) {
  const issues = [];
  if (normalizeProvider(provider) !== "codex") return issues;
  const hasApiKey = Boolean(String(env.OPENAI_API_KEY || "").trim());
  const hasAuthFile = fs.existsSync("/root/.codex/auth.json");
  if (!hasApiKey && !hasAuthFile) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "CODEX_AUTH_MISSING",
      message: `lane '${laneName}' requires OPENAI_API_KEY or /root/.codex/auth.json`,
    }));
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
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "MODEL_MISSING",
      message: "opencode lane is missing a provider/model ref",
      fatal: true,
    }));
    return issues;
  }
  if ((modelProvider === "opencode" || modelProvider === "opencode-go") && !hasOpenCodeAuth(env)) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "OPENCODE_AUTH_MISSING",
      message: `lane '${laneName}' requires OPENCODE_API_KEY/OPENCODE_ZEN_API_KEY or opencode auth login`,
    }));
  }
  if (modelProvider === "dashscope"
    && !String(env.DASHSCOPE_API_KEY || "").trim()
    && !String(env.DASH_SCOPE_API_KEY || "").trim()
    && !String(env.QWEN_API_KEY || "").trim()
    && !String(env.ALIBABA_CODING_PLAN_API_KEY || "").trim()) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "DASHSCOPE_AUTH_MISSING",
      message: `lane '${laneName}' requires DASHSCOPE_API_KEY, DASH_SCOPE_API_KEY, QWEN_API_KEY, or ALIBABA_CODING_PLAN_API_KEY`,
    }));
  }
  if ((modelProvider === "alibaba-coding-plan" || modelProvider === "alibaba-coding-plan-cn")
    && !String(env.ALIBABA_CODING_PLAN_API_KEY || "").trim()
    && !String(env.QWEN_API_KEY || "").trim()
    && !String(env.DASH_SCOPE_API_KEY || "").trim()) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "ALIBABA_AUTH_MISSING",
      message: `lane '${laneName}' requires ALIBABA_CODING_PLAN_API_KEY, QWEN_API_KEY, or DASH_SCOPE_API_KEY`,
    }));
  }
  if ((modelProvider === "minimax" || modelProvider.startsWith("minimax-")) && !hasMiniMaxKey(env)) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "MINIMAX_AUTH_MISSING",
      message: `lane '${laneName}' requires MINIMAX_API_KEY or provider.minimax.options.apiKey in opencode.json`,
    }));
  }
  return issues;
}

function validateLaneConsistency({ defaultProvider = "", defaultModel = "", defaultExecutionLane = "", lanes = {} }) {
  const issues = [];
  const laneName = String(defaultExecutionLane || "").trim();
  if (!laneName) return issues;
  const lane = lanes[laneName];
  if (!lane || typeof lane !== "object") {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "EXECUTION_LANE_UNKNOWN",
      message: `default execution lane '${laneName}' is not defined in runtime config`,
      fatal: true,
    }));
    return issues;
  }

  const laneProvider = normalizeProvider(lane.provider || "");
  const laneModel = normalizeModel(lane.model || "");
  const safeDefaultProvider = normalizeProvider(defaultProvider);
  const safeDefaultModel = normalizeModel(defaultModel);

  if (!laneProvider) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "EXECUTION_LANE_PROVIDER_MISSING",
      message: `default execution lane '${laneName}' is missing provider`,
      fatal: true,
    }));
  }
  if (!laneModel) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "EXECUTION_LANE_MODEL_MISSING",
      message: `default execution lane '${laneName}' is missing model`,
      fatal: true,
    }));
  }
  if (safeDefaultProvider && laneProvider && safeDefaultProvider !== laneProvider) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "DEFAULT_PROVIDER_LANE_MISMATCH",
      message: `default provider '${safeDefaultProvider}' conflicts with lane '${laneName}' provider '${laneProvider}'`,
      fatal: true,
    }));
  }
  if (safeDefaultModel && laneModel && safeDefaultModel !== laneModel) {
    issues.push(buildIssue({
      severity: "error",
      lane: laneName,
      code: "DEFAULT_MODEL_LANE_MISMATCH",
      message: `default model '${safeDefaultModel}' conflicts with lane '${laneName}' model '${laneModel}'`,
      fatal: true,
    }));
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

  issues.push(...validateLaneConsistency({
    defaultProvider,
    defaultModel,
    defaultExecutionLane,
    lanes,
  }));

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
    fatal: issues.some((issue) => Boolean(issue.fatal)),
  };
}
