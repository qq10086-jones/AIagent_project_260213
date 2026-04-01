function truncate(value, maxLen = 240) {
  const safe = String(value || "").trim();
  if (safe.length <= maxLen) return safe;
  return `${safe.slice(0, maxLen)}...`;
}

export function evaluatePermissionAdvisory({ tool_name = "", payload = {}, risk = {} } = {}) {
  const startedAt = Date.now();
  const prompt = String(payload?.task_prompt || payload?.prompt || "").trim();
  const reasons = Array.isArray(risk?.reasons) ? risk.reasons : [];
  const highRiskReasons = new Set(["destructive_command", "db_destructive_operation", "sensitive_path_or_secret", "high_risk_intent"]);
  const hasCriticalReason = reasons.some((reason) => highRiskReasons.has(String(reason || "")));
  const requiresApproval = Boolean(risk?.requires_approval);

  let council_advice = "allow";
  let safety_verdict = "allow";
  let context_verdict = prompt ? "sufficient" : "insufficient";
  let risk_score = 0.2;
  let summary = "No elevated permission risk detected.";

  if (hasCriticalReason) {
    council_advice = "deny";
    safety_verdict = "deny";
    risk_score = 0.98;
    summary = `Critical risk indicators detected: ${reasons.join(", ")}`;
  } else if (requiresApproval) {
    council_advice = "review";
    safety_verdict = "review";
    risk_score = 0.75;
    summary = `Human approval recommended: ${reasons.join(", ") || "policy gate"}`;
  } else if (String(tool_name || "").startsWith("coding.")) {
    council_advice = "allow";
    safety_verdict = "allow";
    risk_score = 0.45;
    summary = "Coding task appears routine, but execution should remain audited.";
  }

  if (!prompt) {
    context_verdict = "insufficient";
    if (council_advice === "allow") {
      council_advice = "review";
      safety_verdict = "review";
      risk_score = Math.max(risk_score, 0.55);
      summary = "Task prompt/context is missing; human review is safer.";
    }
  }

  return {
    council_advice,
    safety_verdict,
    context_verdict,
    risk_level: String(risk?.risk_level || "low"),
    risk_score,
    advisory_summary: truncate(summary),
    reasons,
    duration_ms: Math.max(1, Date.now() - startedAt),
  };
}
