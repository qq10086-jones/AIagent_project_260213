import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const LOCAL_POLICY_PATH = path.resolve(MODULE_DIR, "..", "..", "configs", "context_budget_policy.json");
const ROOT_POLICY_PATH = path.resolve(MODULE_DIR, "..", "..", "..", "configs", "context_budget_policy.json");

function isReadableJsonFile(filePath) {
  try {
    return fs.existsSync(filePath) && fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

const DEFAULT_POLICY_PATH = isReadableJsonFile(LOCAL_POLICY_PATH) ? LOCAL_POLICY_PATH : ROOT_POLICY_PATH;

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function getFileSizeSafe(filePath) {
  try {
    return Number(fs.statSync(filePath).size || 0);
  } catch {
    return 0;
  }
}

function mergeThresholds(policy, role) {
  const defaults = policy?.default_thresholds || {};
  const overrides = policy?.role_overrides?.[String(role || "")] || {};
  return { ...defaults, ...overrides };
}

function classifyBudget({ metrics, thresholds }) {
  const overflow =
    metrics.prompt_chars >= Number(thresholds.overflow_risk_prompt_chars || Number.MAX_SAFE_INTEGER) ||
    metrics.bytes_total >= Number(thresholds.overflow_risk_artifact_bytes || Number.MAX_SAFE_INTEGER) ||
    metrics.injected_context_bytes >= Number(thresholds.overflow_risk_injected_context_bytes || Number.MAX_SAFE_INTEGER);

  if (overflow) return "overflow_risk";

  const warning =
    metrics.prompt_chars >= Number(thresholds.warning_prompt_chars || Number.MAX_SAFE_INTEGER) ||
    metrics.bytes_total >= Number(thresholds.warning_artifact_bytes || Number.MAX_SAFE_INTEGER) ||
    metrics.injected_context_bytes >= Number(thresholds.warning_injected_context_bytes || Number.MAX_SAFE_INTEGER);

  return warning ? "warning" : "ok";
}

export function createContextBudgetService({ policyPath = DEFAULT_POLICY_PATH } = {}) {
  function loadPolicy() {
    if (!isReadableJsonFile(policyPath)) {
      const err = new Error(`context budget policy not found: ${policyPath}`);
      err.code = "CONTEXT_BUDGET_POLICY_MISSING";
      throw err;
    }
    return readJsonFile(policyPath);
  }

  function buildReport({
    stepId,
    role,
    prompt = "",
    injectedContext = "",
    artifactPaths = [],
    artifactBytes = [],
    runId = "",
    workflowRunId = "",
  }) {
    const policy = loadPolicy();
    const thresholds = mergeThresholds(policy, role);
    const thresholdSource = policy?.role_overrides?.[String(role || "")] ? `role_overrides.${role}` : "default_thresholds";
    const artifactByteList = [
      ...artifactPaths.map((item) => getFileSizeSafe(item)),
      ...artifactBytes.map((item) => Number(item || 0)).filter((item) => Number.isFinite(item) && item >= 0),
    ];
    const bytesTotal = artifactByteList.reduce((sum, item) => sum + item, 0);
    const largestArtifactBytes = artifactByteList.length > 0 ? Math.max(...artifactByteList) : 0;
    const metrics = {
      step_id: String(stepId || ""),
      artifact_count: artifactByteList.length,
      bytes_total: bytesTotal,
      largest_artifact_bytes: largestArtifactBytes,
      prompt_chars: String(prompt || "").length,
      injected_context_bytes: Buffer.byteLength(String(injectedContext || ""), "utf8"),
    };
    return {
      ...metrics,
      status: classifyBudget({ metrics, thresholds }),
      threshold_source: thresholdSource,
      thresholds,
      run_id: String(runId || ""),
      workflow_run_id: String(workflowRunId || ""),
    };
  }

  return {
    loadPolicy,
    buildReport,
  };
}
