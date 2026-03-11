import assert from "node:assert/strict";

import { createBrainGatewayHandlers } from "../src/vnext/brain_gateway.js";

function makeRes() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return this;
    },
  };
}

async function main() {
  const recordCalls = [];
  const handlers = createBrainGatewayHandlers({
    pool: {},
    async recordEvent(runId, eventName, payload) {
      recordCalls.push({ runId, eventName, payload });
    },
    async findLatestFactForRun(_pool, input) {
      if (input.run_id === "run-404") return null;
      return {
        fact_id: "fact-1",
        run_id: input.run_id,
        agent_name: input.agent_name,
        kind: "tool_result",
        created_at: "2026-03-11T00:00:00.000Z",
        payload_json: "{\"tool_name\":\"coding.delegate\",\"ok\":true}",
      };
    },
  });

  {
    const res = makeRes();
    await handlers.handleLatestFact({ query: {} }, res);
    assert.equal(res.statusCode, 400);
    assert.equal(res.body.ok, false);
  }

  {
    const res = makeRes();
    await handlers.handleLatestFact({ query: { run_id: "run-404", agent_name: "coder" } }, res);
    assert.equal(res.statusCode, 404);
    assert.equal(res.body.ok, false);
  }

  {
    const res = makeRes();
    await handlers.handleLatestFact({ query: { run_id: "run-1", agent_name: "coder", tool_name: "coding.delegate" } }, res);
    assert.equal(res.statusCode, 200);
    assert.equal(res.body.ok, true);
    assert.equal(res.body.fact.fact_id, "fact-1");
    assert.equal(res.body.fact.payload.tool_name, "coding.delegate");
  }

  {
    const res = makeRes();
    await handlers.handleRoutingDecision({ body: { payload: { routing_decision_source: "dynamic_routing_advisory_only" } } }, res);
    assert.equal(res.statusCode, 400);
    assert.equal(res.body.ok, false);
  }

  {
    const res = makeRes();
    await handlers.handleRoutingDecision({
      body: {
        workflow_run_id: "wf-1",
        event_name: "brain.routing.decision",
        payload: { routing_decision_source: "dynamic_routing_advisory_only", classifier_confidence_band: "high" },
      },
    }, res);
    assert.equal(res.statusCode, 200);
    assert.equal(res.body.ok, true);
    assert.equal(recordCalls.length, 1);
    assert.equal(recordCalls[0].runId, "wf-1");
    assert.equal(recordCalls[0].eventName, "brain.routing.decision");
    assert.equal(recordCalls[0].payload.workflow_run_id, "wf-1");
    assert.equal(recordCalls[0].payload.routing_decision_source, "dynamic_routing_advisory_only");
  }

  console.log("brain_gateway.integration.test.js: all tests passed");
}

main().catch((err) => {
  console.error("brain_gateway.integration.test.js: failed");
  console.error(err);
  process.exit(1);
});
