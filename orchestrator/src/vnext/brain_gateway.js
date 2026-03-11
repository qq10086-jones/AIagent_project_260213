function parsePayload(payloadJson) {
  try {
    return typeof payloadJson === "string" ? JSON.parse(payloadJson) : payloadJson;
  } catch {
    return { raw: String(payloadJson || "") };
  }
}

export function createBrainGatewayHandlers({
  pool,
  recordEvent,
  findLatestFactForRun,
}) {
  async function handleLatestFact(req, res) {
    try {
      const run_id = String(req.query?.run_id || "").trim();
      const agent_name = String(req.query?.agent_name || "").trim();
      const tool_name = String(req.query?.tool_name || "").trim();
      if (!run_id || !agent_name) {
        return res.status(400).json({ ok: false, error: "run_id and agent_name are required" });
      }
      const row = await findLatestFactForRun(pool, { run_id, agent_name, tool_name });
      if (!row) {
        return res.status(404).json({ ok: false, error: "fact not found" });
      }
      return res.json({
        ok: true,
        fact: {
          fact_id: row.fact_id,
          run_id: row.run_id,
          agent_name: row.agent_name,
          kind: row.kind,
          created_at: row.created_at,
          payload: parsePayload(row.payload_json),
        },
      });
    } catch (err) {
      return res.status(500).json({ ok: false, error: err.message || "brain fact query failed" });
    }
  }

  async function handleRoutingDecision(req, res) {
    try {
      const run_id = String(req.body?.run_id || "").trim();
      const workflow_run_id = String(req.body?.workflow_run_id || "").trim();
      const event_name = String(req.body?.event_name || "brain.routing.decision").trim();
      const payload = req.body?.payload && typeof req.body.payload === "object" ? req.body.payload : {};
      if (!run_id && !workflow_run_id) {
        return res.status(400).json({ ok: false, error: "run_id or workflow_run_id is required" });
      }
      await recordEvent(workflow_run_id || run_id, event_name, {
        run_id: run_id || null,
        workflow_run_id: workflow_run_id || null,
        ...payload,
      });
      return res.json({ ok: true, event_name });
    } catch (err) {
      return res.status(500).json({ ok: false, error: err.message || "brain routing decision ingest failed" });
    }
  }

  return {
    handleLatestFact,
    handleRoutingDecision,
  };
}
