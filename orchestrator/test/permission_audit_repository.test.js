import assert from "node:assert/strict";

import {
  getPermissionAuditSummary,
  listPermissionAuditRecords,
  updatePermissionAuditHumanDecision,
} from "../src/data/permission_audit_repository.js";

async function main() {
  const calls = [];
  const pool = {
    async query(sql, params) {
      calls.push([sql, params]);
      return { rows: [{ run_id: "run-1", council_advice: "review" }] };
    },
  };

  const rows = await listPermissionAuditRecords(pool, {
    run_id: "run-1",
    risk_level: "high",
    final_human_decision: "approved",
    limit: 10,
  });

  assert.equal(rows.length, 1);
  assert.match(String(calls.at(-1)?.[0] || ""), /permission_audit_log/);
  assert.deepEqual(calls.at(-1)?.[1], ["run-1", "high", "approved", 10]);

  const updateCalls = [];
  const updated = await updatePermissionAuditHumanDecision({
    query: async (sql, params) => {
      updateCalls.push([sql, params]);
      return { rowCount: 1 };
    },
  }, {
    task_id: "task-1",
    final_human_decision: "approved",
  });
  assert.equal(updated, 1);
  assert.match(String(updateCalls[0][0]), /UPDATE permission_audit_log/);
  assert.deepEqual(updateCalls[0][1], ["task-1", "approved"]);

  const summaryCalls = [];
  const summary = await getPermissionAuditSummary({
    query: async (sql, params) => {
      summaryCalls.push([sql, params]);
      return {
        rows: [{
          total_records: 10,
          reviewed_records: 8,
          advice_allow_count: 4,
          advice_review_count: 3,
          advice_deny_count: 3,
          human_approved_count: 5,
          human_rejected_count: 3,
          aligned_decision_count: 5,
          comparable_decision_count: 6,
          override_count: 1,
          false_negative_count: 1,
          review_escalation_count: 3,
        }],
      };
    },
  }, {
    days: 14,
    risk_level: "high",
  });
  assert.match(String(summaryCalls[0][0]), /COUNT\(\*\) FILTER/);
  assert.deepEqual(summaryCalls[0][1], [14, "high"]);
  assert.equal(summary.window_days, 14);
  assert.equal(summary.risk_level, "high");
  assert.equal(summary.total_records, 10);
  assert.equal(summary.reviewed_records, 8);
  assert.equal(summary.comparable_decision_count, 6);
  assert.equal(summary.aligned_decision_count, 5);
  assert.equal(summary.override_count, 1);
  assert.equal(summary.false_negative_count, 1);
  assert.deepEqual(summary.advice_breakdown, { allow: 4, review: 3, deny: 3 });
  assert.deepEqual(summary.human_decision_breakdown, { approved: 5, rejected: 3, pending: 2 });
  assert.equal(summary.rates.reviewed_rate, 0.8);
  assert.equal(summary.rates.alignment_rate, 0.8333);
  assert.equal(summary.rates.override_rate, 0.1667);
  assert.equal(summary.rates.false_negative_rate, 0.1);
  console.log("permission_audit_repository.test.js: all tests passed");
}

main().catch((err) => {
  console.error("permission_audit_repository.test.js: failed");
  console.error(err);
  process.exit(1);
});
