import fs from "fs";
import path from "path";

/**
 * Nexus Policy Engine
 * Responsible for risk scoring and approval gate decisions.
 */

const CAPABILITY_REGISTRY_CANDIDATES = [
  path.join(process.cwd(), "configs", "registry", "capability_registry.json"),
  path.join(process.cwd(), "configs", "capability_registry.json"),
  path.join(process.cwd(), "..", "configs", "registry", "capability_registry.json"),
  path.join(process.cwd(), "..", "configs", "capability_registry.json"),
];

function resolveRegistryPath() {
  for (const p of CAPABILITY_REGISTRY_CANDIDATES) {
    if (fs.existsSync(p)) return p;
  }
  return CAPABILITY_REGISTRY_CANDIDATES[0];
}

function escapeRegex(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function pathLikePattern(pathPrefix) {
  const normalized = String(pathPrefix || "").trim().replace(/\\/g, "/");
  if (!normalized) return null;
  if (normalized === ".env*") {
    return /(^|[\s"'`(])(?:\.env(?:\.[A-Za-z0-9._-]+)?)(?=$|[\s"'`),/\\])/i;
  }
  const hasWildcard = normalized.endsWith("*");
  const clean = hasWildcard ? normalized.slice(0, -1) : normalized;
  if (!clean) return null;
  const escaped = escapeRegex(clean);
  const boundaryPrefix = "(^|[\\s\"'`(])";
  const boundarySuffix = "(?=$|[\\s\"'`),/\\\\])";
  if (clean.endsWith("/")) {
    return new RegExp(boundaryPrefix + escaped + "(?=\\S|$)", "i");
  }
  return new RegExp("(^|[\\s\"'`(/])" + escaped + boundarySuffix, "i");
}

export function analyzeTaskRisk(tool_name, payload) {
  const prompt = String(payload?.task_prompt || payload?.prompt || "");
  const reasons = [];

  // 1. Static Pattern Matching (Heuristics)
  const highRiskPatterns = [
    { re: /(?:\brm\s+-rf\b|\bgit\s+reset\s+--hard\b|\bdel\s+\/f\b|\bformat\s+(?:[a-z]:|\/[qsuvxy])|\bmkfs\b|\bdd\s+if=)/i, reason: "destructive_command" },
    { re: /\b(?:drop\s+table|truncate\s+table|alter\s+table)\b/i, reason: "db_destructive_operation" },
    { re: /(?:^|[\s"'`(/])(?:\.github\/|infra\/|deploy\/|k8s\/|Dockerfile)(?=$|[\s"'`),/\\])/i, reason: "sensitive_path_or_secret" },
    { re: /(^|[\s"'`(])(?:\.env(?:\.[A-Za-z0-9._-]+)?)(?=$|[\s"'`),/\\])/i, reason: "sensitive_path_or_secret" },
    { re: /\b(?:secret(?:s)?|api[_ -]?key|access[_ -]?key|private[_ -]?key|client[_ -]?secret)\b/i, reason: "sensitive_path_or_secret" },
    { re: /(生产|正式环境|线上|密钥|数据库迁移|删除表)/i, reason: "high_risk_intent" },
  ];

  for (const item of highRiskPatterns) {
    if (item.re.test(prompt)) reasons.push(item.reason);
  }

  // 2. Registry-based Path Checks
  try {
    const registryPath = resolveRegistryPath();
    const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"));
    const codingPolicy = registry.project_types?.coding_task?.policy;

    if (codingPolicy?.manual_approve_paths) {
      for (const pathPrefix of codingPolicy.manual_approve_paths) {
        const re = pathLikePattern(pathPrefix);
        if (re?.test(prompt)) {
          reasons.push(`registry_manual_path:${pathPrefix}`);
        }
      }
    }
  } catch (err) {
    console.warn(`[policy] Could not load capability registry: ${err.message}`);
  }

  const uniqueReasons = [...new Set(reasons)];
  const isHighRisk = uniqueReasons.length > 0;

  // coding.delegate tasks run in sandboxed containers; architecture docs naturally mention
  // "API key", "secret", etc.  Only require approval for truly destructive patterns.
  const isCodingTask = String(tool_name || "").startsWith("coding.");
  const onlySensitivePath = isCodingTask && uniqueReasons.length > 0 &&
    uniqueReasons.every((r) => r === "sensitive_path_or_secret");

  return {
    risk_level: isHighRisk ? (onlySensitivePath ? "medium" : "high") : (isCodingTask ? "medium" : "low"),
    requires_approval: isHighRisk && !onlySensitivePath,
    reasons: uniqueReasons,
  };
}
