function asString(value) {
  return String(value || "").trim();
}

export function isSingleAgentPayload(payload = {}) {
  return String(payload?.task_envelope?.decision || "").trim() === "single_agent";
}

export function validateSingleAgentPayloadGuardrails(payload = {}) {
  const errors = [];
  if (!isSingleAgentPayload(payload)) {
    return { ok: true, errors };
  }
  if (!asString(payload.evidence_id || payload?.task_envelope?.evidence_id)) {
    errors.push("evidence_id is required for single_agent");
  }
  if (!asString(payload.replay_tag || payload?.task_envelope?.replay_tag)) {
    errors.push("replay_tag is required for single_agent");
  }
  return { ok: errors.length === 0, errors };
}

export function validateSingleAgentWorkerResultGuardrails(workerResult = {}, { requireSingleAgent = true } = {}) {
  const errors = [];
  if (!requireSingleAgent) {
    return { ok: true, errors };
  }
  const metadata = workerResult && typeof workerResult === "object" && workerResult.metadata && typeof workerResult.metadata === "object"
    ? workerResult.metadata
    : {};
  if (!asString(metadata.evidence_id)) {
    errors.push("worker_result.metadata.evidence_id is required for single_agent");
  }
  if (!asString(metadata.replay_tag)) {
    errors.push("worker_result.metadata.replay_tag is required for single_agent");
  }
  if (!asString(metadata.output_hash)) {
    errors.push("worker_result.metadata.output_hash is required for single_agent");
  }
  if (!Array.isArray(metadata.bounded_validation) || metadata.bounded_validation.length === 0) {
    errors.push("worker_result.metadata.bounded_validation must contain at least one item for single_agent");
  }
  return { ok: errors.length === 0, errors };
}
