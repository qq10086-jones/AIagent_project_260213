import fs from "fs";
import path from "path";

export function buildRequiredConfigFiles({
  runtimeConfigPath = "",
  workspaceRoot = process.cwd(),
  existsSync = fs.existsSync,
} = {}) {
  const root = path.resolve(workspaceRoot || process.cwd());
  const runtimeCandidates = [
    String(runtimeConfigPath || "").trim(),
    path.resolve(root, "configs/runtime/runtime_defaults.json"),
    path.resolve(root, "../configs/runtime/runtime_defaults.json"),
  ].filter(Boolean);
  const runtimeResolved = runtimeCandidates.find((item) => existsSync(item)) || runtimeCandidates[0];
  return [
    {
      id: "runtime_defaults",
      path: runtimeResolved,
      reason: "runtime defaults are required to resolve orchestrator and worker defaults",
    },
    {
      id: "llm_providers",
      path: path.resolve(root, "configs/llm_providers.json"),
      reason: "llm dispatcher provider registry is required at startup",
    },
    {
      id: "llm_role_policy",
      path: path.resolve(root, "configs/llm_role_policy.json"),
      reason: "llm dispatcher role policy is required at startup",
    },
    {
      id: "context_budget_policy",
      path: path.resolve(root, "configs/context_budget_policy.json"),
      reason: "context budget policy is required to finalize workflow step artifacts",
    },
  ];
}

export function runConfigPreflight({
  runtimeConfigPath = "",
  workspaceRoot = process.cwd(),
  existsSync = fs.existsSync,
} = {}) {
  const required = buildRequiredConfigFiles({ runtimeConfigPath, workspaceRoot, existsSync });
  const items = required.map((item) => ({
    ...item,
    exists: Boolean(existsSync(item.path)),
  }));
  const missing = items.filter((item) => !item.exists);
  return {
    ok: missing.length === 0,
    items,
    missing,
    error_code: missing.length > 0 ? "CONFIG_PREFLIGHT_FAILED" : null,
  };
}

export function assertConfigPreflight(options = {}) {
  const result = runConfigPreflight(options);
  if (!result.ok) {
    const detail = result.missing
      .map((item) => `${item.id}: ${item.path}`)
      .join("; ");
    const err = new Error(`CONFIG_PREFLIGHT_FAILED: missing required config files: ${detail}`);
    err.code = "CONFIG_PREFLIGHT_FAILED";
    err.details = result;
    throw err;
  }
  return result;
}
