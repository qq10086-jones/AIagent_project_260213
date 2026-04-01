import assert from "node:assert/strict";

import {
  evaluateCouncilBaseline,
  renderPermissionAuditWeeklyReport,
} from "../src/vnext/permission_audit_report.js";

function main() {
  const strongSummary = {
    window_days: 30,
    risk_level: "all",
    total_records: 24,
    reviewed_records: 24,
    comparable_decision_count: 20,
    aligned_decision_count: 19,
    override_count: 1,
    false_negative_count: 0,
    review_escalation_count: 6,
    advice_breakdown: { allow: 10, review: 6, deny: 8 },
    human_decision_breakdown: { approved: 12, rejected: 12, pending: 0 },
    rates: {
      advice_allow_rate: 10 / 24,
      advice_review_rate: 6 / 24,
      advice_deny_rate: 8 / 24,
      reviewed_rate: 1,
      alignment_rate: 19 / 20,
      override_rate: 1 / 20,
      false_negative_rate: 0,
    },
  };

  const strongBaseline = evaluateCouncilBaseline(strongSummary);
  assert.equal(strongBaseline.baseline_ready, true);
  assert.equal(strongBaseline.meets_alignment, true);
  assert.equal(strongBaseline.meets_false_negative, true);

  const strongReport = renderPermissionAuditWeeklyReport(strongSummary);
  assert.equal(strongReport.verdict, "READY_FOR_V4_REVIEW");
  assert.match(strongReport.markdown, /Alignment rate: 95\.0%/);
  assert.match(strongReport.markdown, /False negatives: 0/);
  assert.match(strongReport.markdown, /Prepare a scoped v4 expansion proposal/);

  const weakSummary = {
    ...strongSummary,
    aligned_decision_count: 7,
    comparable_decision_count: 10,
    false_negative_count: 2,
    rates: {
      ...strongSummary.rates,
      alignment_rate: 0.7,
      override_rate: 0.3,
      false_negative_rate: 2 / 24,
    },
  };
  const weakBaseline = evaluateCouncilBaseline(weakSummary);
  assert.equal(weakBaseline.baseline_ready, false);

  const weakReport = renderPermissionAuditWeeklyReport(weakSummary);
  assert.equal(weakReport.verdict, "ADVISORY_ONLY");
  assert.match(weakReport.markdown, /Keep Permission Council in advisory-only mode/);
  assert.match(weakReport.markdown, /Alignment rate is below the required 90% threshold/);
  assert.match(weakReport.markdown, /False negatives are non-zero/);

  console.log("permission_audit_report.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("permission_audit_report.test.js: failed");
  console.error(err);
  process.exit(1);
}
