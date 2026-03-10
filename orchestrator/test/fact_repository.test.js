import assert from "node:assert/strict";

import { findLatestFactForRun } from "../src/data/fact_repository.js";

async function main() {
  const calls = [];
  const pool = {
    async query(sql, params) {
      calls.push({ sql: String(sql), params });
      return {
        rows: [{
          fact_id: "fact-1",
          run_id: "run-1",
          agent_name: "coder",
          kind: "tool_result",
          payload_json: "{\"tool_name\":\"coding.delegate\"}",
          created_at: "2026-03-10T00:00:00.000Z",
        }],
      };
    },
  };

  const row = await findLatestFactForRun(pool, {
    run_id: "run-1",
    agent_name: "coder",
    tool_name: "coding.delegate",
  });

  assert.equal(row.fact_id, "fact-1");
  assert.equal(calls.length, 1);
  assert.match(calls[0].sql, /FROM fact_items/);
  assert.deepEqual(calls[0].params, ["run-1", "coder", "%\"coding.delegate\"%"]);

  console.log("fact_repository.test.js: all tests passed");
}

main().catch((err) => {
  console.error("fact_repository.test.js: failed");
  console.error(err);
  process.exit(1);
});
