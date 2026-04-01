function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function safeInt(value) {
  const num = Number(value || 0);
  return Number.isFinite(num) ? Math.max(0, Math.trunc(num)) : 0;
}

export function evaluateCouncilBaseline(summary = {}) {
  const alignmentRate = Number(summary?.rates?.alignment_rate || 0);
  const falseNegativeCount = safeInt(summary?.false_negative_count);
  const totalRecords = safeInt(summary?.total_records);
  const reviewedRecords = safeInt(summary?.reviewed_records);
  const hasEnoughData = totalRecords > 0 && reviewedRecords > 0;
  const meetsAlignment = alignmentRate >= 0.9;
  const meetsFalseNegative = falseNegativeCount === 0;
  const baselineReady = hasEnoughData && meetsAlignment && meetsFalseNegative;

  return {
    baseline_ready: baselineReady,
    has_enough_data: hasEnoughData,
    meets_alignment: meetsAlignment,
    meets_false_negative: meetsFalseNegative,
    threshold_alignment_rate: 0.9,
    threshold_false_negative_count: 0,
  };
}

export function renderPermissionAuditWeeklyReport(summary = {}) {
  const baseline = evaluateCouncilBaseline(summary);
  const windowDays = safeInt(summary?.window_days || 30) || 30;
  const riskLevel = String(summary?.risk_level || "all");
  const advice = summary?.advice_breakdown || {};
  const human = summary?.human_decision_breakdown || {};
  const rates = summary?.rates || {};

  const verdict = baseline.baseline_ready
    ? "READY_FOR_V4_REVIEW"
    : "ADVISORY_ONLY";

  const lines = [
    "# Permission Council Weekly Report",
    "",
    `- Window: ${windowDays} days`,
    `- Risk Level: ${riskLevel}`,
    `- Verdict: ${verdict}`,
    "",
    "## Volume",
    `- Total records: ${safeInt(summary?.total_records)}`,
    `- Reviewed records: ${safeInt(summary?.reviewed_records)} (${pct(rates.reviewed_rate)})`,
    `- Comparable decisions: ${safeInt(summary?.comparable_decision_count)}`,
    "",
    "## Council Advice Breakdown",
    `- Allow: ${safeInt(advice.allow)} (${pct(rates.advice_allow_rate)})`,
    `- Review: ${safeInt(advice.review)} (${pct(rates.advice_review_rate)})`,
    `- Deny: ${safeInt(advice.deny)} (${pct(rates.advice_deny_rate)})`,
    "",
    "## Human Decisions",
    `- Approved: ${safeInt(human.approved)}`,
    `- Rejected: ${safeInt(human.rejected)}`,
    `- Pending: ${safeInt(human.pending)}`,
    "",
    "## Quality Signals",
    `- Alignment rate: ${pct(rates.alignment_rate)} (target >= 90.0%)`,
    `- Override rate: ${pct(rates.override_rate)}`,
    `- False negatives: ${safeInt(summary?.false_negative_count)} (target = 0)`,
    `- Review escalations: ${safeInt(summary?.review_escalation_count)}`,
    "",
    "## Recommendation",
    baseline.baseline_ready
      ? "- Council advisory quality meets the v3.1 threshold. Prepare a scoped v4 expansion proposal for low-risk actions."
      : "- Keep Permission Council in advisory-only mode. Do not expand decision authority yet.",
  ];

  if (!baseline.has_enough_data) {
    lines.push("- Data volume is insufficient for a governance upgrade decision.");
  }
  if (!baseline.meets_alignment) {
    lines.push("- Alignment rate is below the required 90% threshold.");
  }
  if (!baseline.meets_false_negative) {
    lines.push("- False negatives are non-zero; this blocks any expansion of authority.");
  }

  return {
    verdict,
    baseline,
    markdown: `${lines.join("\n")}\n`,
  };
}
