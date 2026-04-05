function truncate(value, maxLen = 240) {
  const safe = String(value || "").trim();
  if (safe.length <= maxLen) return safe;
  return `${safe.slice(0, maxLen)}...`;
}

// --- Advisory Agent: SafetyAuditor ---
// Evaluates whether the requested action poses a safety risk based on risk indicators.
const HIGH_RISK_REASONS = new Set([
  "destructive_command",
  "db_destructive_operation",
  "sensitive_path_or_secret",
  "high_risk_intent",
]);

export function auditSafety({ tool_name = "", reasons = [], requires_approval = false } = {}) {
  const hasCriticalReason = reasons.some((r) => HIGH_RISK_REASONS.has(String(r || "")));
  if (hasCriticalReason) {
    return {
      verdict: "deny",
      summary: `Critical risk indicators detected: ${reasons.join(", ")}`,
    };
  }
  if (requires_approval) {
    return {
      verdict: "review",
      summary: `Human approval recommended: ${reasons.join(", ") || "policy gate"}`,
    };
  }
  if (String(tool_name || "").startsWith("coding.")) {
    return {
      verdict: "allow",
      summary: "Coding task appears routine, but execution should remain audited.",
    };
  }
  return { verdict: "allow", summary: "No elevated safety risk detected." };
}

// --- Advisory Agent: ContextValidator ---
// Checks whether sufficient context (prompt / payload) is available to make a safe decision.
export function validateContext({ prompt = "" } = {}) {
  const hasPrompt = String(prompt || "").trim().length > 0;
  return {
    verdict: hasPrompt ? "sufficient" : "insufficient",
    summary: hasPrompt
      ? "Task prompt/context is present."
      : "Task prompt/context is missing; human review is safer.",
  };
}

// --- Advisory Agent: RiskScorer ---
// Assigns a numeric risk score (0-1) based on safety and context verdicts.
export function scoreRisk({ safety_verdict = "allow", context_verdict = "sufficient", reasons = [], tool_name = "" } = {}) {
  if (safety_verdict === "deny") return { score: 0.98 };
  if (safety_verdict === "review") return { score: 0.75 };
  if (context_verdict === "insufficient") return { score: 0.55 };
  if (String(tool_name || "").startsWith("coding.")) return { score: 0.45 };
  if (reasons.length > 0) return { score: 0.45 };
  return { score: 0.2 };
}

// --- Permission Council (composes the three agents) ---
export function evaluatePermissionAdvisory({ tool_name = "", payload = {}, risk = {} } = {}) {
  const startedAt = Date.now();
  const prompt = String(payload?.task_prompt || payload?.prompt || "").trim();
  const reasons = Array.isArray(risk?.reasons) ? risk.reasons : [];
  const requiresApproval = Boolean(risk?.requires_approval);

  const safety = auditSafety({ tool_name, reasons, requires_approval: requiresApproval });
  const context = validateContext({ prompt });
  const riskScore = scoreRisk({
    safety_verdict: safety.verdict,
    context_verdict: context.verdict,
    reasons,
    tool_name,
  });

  let council_advice = safety.verdict;
  let summary = safety.summary;

  // Context escalation: if safety says allow but context is insufficient, escalate to review
  if (council_advice === "allow" && context.verdict === "insufficient") {
    council_advice = "review";
    summary = context.summary;
  }

  return {
    council_advice,
    safety_verdict: safety.verdict,
    context_verdict: context.verdict,
    risk_level: String(risk?.risk_level || "low"),
    risk_score: riskScore.score,
    advisory_summary: truncate(summary),
    reasons,
    duration_ms: Math.max(1, Date.now() - startedAt),
  };
}
