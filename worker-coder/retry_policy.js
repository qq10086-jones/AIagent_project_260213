import { redactSensitiveText } from "./verification_runner.js";

export function shouldRetryAutoFix({ summary, attemptIndex, maxAttempts, sameErrorRepeatLimit, attemptRecords, errorCounts }) {
  if (!summary || summary.ok) return { retry: false, reason: "already_succeeded" };
  if (attemptIndex >= maxAttempts) return { retry: false, reason: "attempt_budget_exhausted" };
  const phase = String(summary?.diagnostics?.failed_phase || "");
  if (!["delegate", "static_check", "verification"].includes(phase)) {
    return { retry: false, reason: "phase_not_retryable" };
  }
  const key = `${String(summary?.diagnostics?.error_code || "")}|${String(summary?.error || "")}`;
  const sameErrorCount = Number(errorCounts.get(key) || 0) + 1;
  errorCounts.set(key, sameErrorCount);
  if (sameErrorCount >= sameErrorRepeatLimit) {
    return { retry: false, reason: "same_error_repeat_limit_reached" };
  }
  const previousAttempt = attemptRecords.length > 0 ? attemptRecords[attemptRecords.length - 1] : null;
  if (previousAttempt && previousAttempt.error_code === summary?.diagnostics?.error_code && previousAttempt.error === summary?.error) {
    return { retry: false, reason: "same_error_repeated_consecutively" };
  }
  return { retry: true, reason: "repair_retry_allowed" };
}

export function buildAttemptedFixes({ adapterResult = null, staticCheck = null, verification = null }) {
  const attempts = [];
  attempts.push("delegate_once");
  if (adapterResult?.diagnostics?.fallback_from) {
    attempts.push(`provider_fallback:${String(adapterResult.diagnostics.fallback_from)}->${String(adapterResult.provider_used || "")}`);
  }
  if (staticCheck?.checked) {
    attempts.push(`static_check:${staticCheck.ok ? "passed" : "failed"}`);
  }
  if (verification?.checked) {
    attempts.push(`verification:${verification.ok ? "passed" : "failed"}`);
  }
  return attempts;
}

export function buildFinalFailureSummary({
  summary,
  attemptRecords = [],
  maxAttempts,
  sameErrorRepeatLimit,
  wallClockTimeoutS,
  terminalReason,
  startedAt,
}) {
  const lastAttempt = attemptRecords.length > 0 ? attemptRecords[attemptRecords.length - 1] : null;
  return {
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    attempts_used: attemptRecords.length > 0 ? attemptRecords.length : 1,
    max_attempts: Number(maxAttempts || 1),
    same_error_repeat_limit: Number(sameErrorRepeatLimit || 1),
    wall_clock_timeout_s: Number(wallClockTimeoutS || 0),
    terminal_reason: String(terminalReason || "failed"),
    failed_phase: String(summary?.diagnostics?.failed_phase || "delegate"),
    error_code: String(summary?.diagnostics?.error_code || ""),
    error: redactSensitiveText(String(summary?.error || "")),
    command_used: summary?.command_used || null,
    verification_command: summary?.diagnostics?.verification?.command || null,
    last_attempt: lastAttempt,
  };
}
