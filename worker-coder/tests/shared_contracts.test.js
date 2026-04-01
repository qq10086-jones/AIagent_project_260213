import assert from "node:assert/strict";
import { CORE_TOOL_SCHEMAS, validateToolSchema } from "../../shared/contracts/tool_schema.js";
import { hashWorkerResultOutput, validateWorkerResult } from "../../shared/contracts/worker_result.js";
import { validateSingleAgentPayloadGuardrails, validateSingleAgentWorkerResultGuardrails } from "../../shared/contracts/single_agent_guardrails.js";

function testCoreToolSchemas() {
  for (const schema of Object.values(CORE_TOOL_SCHEMAS)) {
    const result = validateToolSchema(schema);
    assert.equal(result.ok, true, JSON.stringify(result));
  }
}

function testWorkerResultNormalization() {
  const result = validateWorkerResult({
    run_id: "run-123",
    worker_name: "worker-coder",
    status: "succeeded",
    ok: true,
    output: { summary: "done", files_changed: ["a.js"] },
    metadata: {
      duration_ms: 12,
      tool_calls: 3,
      permission_decisions: [{ tool_name: "write_file", recommendation: "allow" }],
      bounded_validation: [{ name: "shape_check", ok: true }],
      replay_tag: "rt-1",
      evidence_id: "ev-1",
    },
  });
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.value.metadata.output_hash.length, 64);
}

function testWorkerResultRequiresRunId() {
  const result = validateWorkerResult({
    worker_name: "worker-coder",
    ok: false,
    error: "boom",
  });
  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /run_id is required/);
}

function testHashStableAcrossKeyOrder() {
  const a = hashWorkerResultOutput({ b: 2, a: 1 });
  const b = hashWorkerResultOutput({ a: 1, b: 2 });
  assert.equal(a, b);
}

function testSingleAgentGuardrails() {
  assert.equal(validateSingleAgentPayloadGuardrails({
    task_envelope: { decision: "single_agent", evidence_id: "ev-1", replay_tag: "rp-1" },
    evidence_id: "ev-1",
    replay_tag: "rp-1",
  }).ok, true);

  assert.equal(validateSingleAgentPayloadGuardrails({
    task_envelope: { decision: "single_agent" },
  }).ok, false);

  assert.equal(validateSingleAgentWorkerResultGuardrails({
    metadata: {
      evidence_id: "ev-1",
      replay_tag: "rp-1",
      output_hash: "abc123",
      bounded_validation: [{ name: "shape", ok: true }],
    },
  }).ok, true);

  assert.equal(validateSingleAgentWorkerResultGuardrails({
    metadata: {
      evidence_id: "",
      replay_tag: "",
      output_hash: "",
      bounded_validation: [],
    },
  }).ok, false);
}

testCoreToolSchemas();
testWorkerResultNormalization();
testWorkerResultRequiresRunId();
testHashStableAcrossKeyOrder();
testSingleAgentGuardrails();
console.log("shared_contracts.test.js: all tests passed");
