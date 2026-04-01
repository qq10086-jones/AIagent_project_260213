import assert from "node:assert/strict";

import { listExecutionAuditEventsByEvidenceId } from "../src/data/execution_audit_repository.js";

async function testRepositoryBuildsEvidenceQuery() {
  const calls = [];
  const pool = {
    async query(sql, params) {
      calls.push([sql, params]);
      return {
        rows: [
          {
            id: 1,
            run_id: "run-1",
            task_id: "task-1",
            worker_name: "worker-coder",
            event_type: "task_complete",
            payload_json: { worker_result: { metadata: { evidence_id: "abc123" } } },
            created_at: "2026-04-01T00:00:00.000Z",
          },
        ],
      };
    },
  };

  const rows = await listExecutionAuditEventsByEvidenceId(pool, "abc123");
  assert.equal(rows.length, 1);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][1][0], "abc123");
  assert.match(String(calls[0][0]), /execution_audit_log/);
  assert.match(String(calls[0][0]), /payload_json->'worker_result'->'metadata'->>'evidence_id'/);
}

async function testRepositoryRejectsBlankEvidenceId() {
  const pool = {
    async query() {
      throw new Error("should not query for blank evidence_id");
    },
  };
  const rows = await listExecutionAuditEventsByEvidenceId(pool, "   ");
  assert.deepEqual(rows, []);
}

async function main() {
  await testRepositoryBuildsEvidenceQuery();
  await testRepositoryRejectsBlankEvidenceId();
  console.log("execution_audit_repository.test.js: all tests passed");
}

main().catch((err) => {
  console.error("execution_audit_repository.test.js: failed");
  console.error(err);
  process.exit(1);
});
