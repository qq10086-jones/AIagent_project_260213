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

export function analyzeTaskRisk(tool_name, payload) {
  const prompt = String(payload?.task_prompt || payload?.prompt || "");
  const reasons = [];
  
  // 1. Static Pattern Matching (Heuristics)
  const highRiskPatterns = [
    { re: /\b(?:rm\s+-rf|git\s+reset\s+--hard|del\s+\/f|format\s+|mkfs|dd\s+if=)\b/i, reason: "destructive_command" },
    { re: /\b(?:drop\s+table|truncate\s+table|alter\s+table)\b/i, reason: "db_destructive_operation" },
    { re: /\b(?:\.github\/|infra\/|deploy\/|k8s\/|Dockerfile|\.env|secret|credentials?)\b/i, reason: "sensitive_path_or_secret" },
    { re: /(生产|正式环境|线上|密钥|数据库迁移|删除表)/i, reason: "high_risk_intent" }
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
        const cleanPrefix = pathPrefix.replace("*", "");
        if (prompt.includes(cleanPrefix)) {
          reasons.push(`registry_manual_path:${pathPrefix}`);
        }
      }
    }
  } catch (err) {
    console.warn(`[policy] Could not load capability registry: ${err.message}`);
  }

  const uniqueReasons = [...new Set(reasons)];
  const isHighRisk = uniqueReasons.length > 0;

  return {
    risk_level: isHighRisk ? "high" : (tool_name.startsWith("coding.") ? "medium" : "low"),
    requires_approval: isHighRisk,
    reasons: uniqueReasons
  };
}
