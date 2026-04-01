import crypto from "crypto";

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeString(value) {
  return String(value || "").trim();
}

function stableSerialize(value) {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableSerialize(item)).join(",")}]`;
  }
  if (isPlainObject(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableSerialize(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value ?? null);
}

export function hashWorkerResultOutput(value) {
  return crypto.createHash("sha256").update(stableSerialize(value)).digest("hex");
}

export function normalizeWorkerResult(value = {}) {
  const source = isPlainObject(value) ? value : {};
  const metadata = isPlainObject(source.metadata) ? source.metadata : {};
  const output = source.output ?? source.payload ?? null;
  const normalized = {
    run_id: normalizeString(source.run_id),
    worker_name: normalizeString(source.worker_name || source.agent_name || "unknown"),
    status: normalizeString(source.status || (source.ok ? "succeeded" : "failed")) || "failed",
    ok: Boolean(source.ok),
    output,
    error: source.error ? String(source.error) : null,
    metadata: {
      duration_ms: Number.isFinite(Number(metadata.duration_ms ?? source.duration_ms))
        ? Number(metadata.duration_ms ?? source.duration_ms)
        : 0,
      tool_calls: Number.isFinite(Number(metadata.tool_calls))
        ? Number(metadata.tool_calls)
        : 0,
      permission_decisions: Array.isArray(metadata.permission_decisions) ? metadata.permission_decisions : [],
      evidence_id: normalizeString(metadata.evidence_id),
      replay_tag: normalizeString(metadata.replay_tag),
      output_hash: normalizeString(metadata.output_hash) || hashWorkerResultOutput(output),
      bounded_validation: Array.isArray(metadata.bounded_validation) ? metadata.bounded_validation : [],
      created_at: normalizeString(metadata.created_at || new Date().toISOString()),
    },
  };
  return normalized;
}

export function validateWorkerResult(value) {
  const normalized = normalizeWorkerResult(value);
  const errors = [];
  if (!normalized.run_id) errors.push("run_id is required");
  if (!normalized.worker_name) errors.push("worker_name is required");
  if (!normalized.status) errors.push("status is required");
  if (!Number.isFinite(normalized.metadata.duration_ms) || normalized.metadata.duration_ms < 0) {
    errors.push("metadata.duration_ms must be a non-negative number");
  }
  if (!Number.isFinite(normalized.metadata.tool_calls) || normalized.metadata.tool_calls < 0) {
    errors.push("metadata.tool_calls must be a non-negative number");
  }
  if (!Array.isArray(normalized.metadata.permission_decisions)) {
    errors.push("metadata.permission_decisions must be an array");
  }
  if (!normalized.metadata.output_hash) {
    errors.push("metadata.output_hash is required");
  }
  if (!Array.isArray(normalized.metadata.bounded_validation)) {
    errors.push("metadata.bounded_validation must be an array");
  }
  return {
    ok: errors.length === 0,
    errors,
    value: normalized,
  };
}
