const RISK_LEVELS = new Set(["safe", "low", "medium", "high", "critical"]);

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeString(value) {
  return String(value || "").trim();
}

export function normalizeRiskProfile(value = {}) {
  const source = isPlainObject(value) ? value : {};
  const level = normalizeString(source.level || "medium").toLowerCase();
  return {
    level: RISK_LEVELS.has(level) ? level : "medium",
    reversible: Boolean(source.reversible),
    scope: normalizeString(source.scope || "workspace") || "workspace",
    requires_human_approval: Boolean(source.requires_human_approval),
  };
}

export function normalizeToolSchema(value = {}) {
  const source = isPlainObject(value) ? value : {};
  return {
    name: normalizeString(source.name),
    description: normalizeString(source.description),
    input_schema: isPlainObject(source.input_schema) ? source.input_schema : { type: "object", additionalProperties: true },
    risk_profile: normalizeRiskProfile(source.risk_profile),
  };
}

export function validateToolSchema(value) {
  const normalized = normalizeToolSchema(value);
  const errors = [];
  if (!normalized.name) errors.push("name is required");
  if (!normalized.description) errors.push("description is required");
  if (!isPlainObject(normalized.input_schema)) errors.push("input_schema must be an object");
  if (!RISK_LEVELS.has(normalized.risk_profile.level)) errors.push("risk_profile.level is invalid");
  return {
    ok: errors.length === 0,
    errors,
    value: normalized,
  };
}

export const CORE_TOOL_SCHEMAS = {
  read_file: normalizeToolSchema({
    name: "read_file",
    description: "Read file contents from the workspace.",
    input_schema: {
      type: "object",
      required: ["path"],
      properties: {
        path: { type: "string" },
      },
      additionalProperties: false,
    },
    risk_profile: {
      level: "safe",
      reversible: true,
      scope: "workspace",
      requires_human_approval: false,
    },
  }),
  write_file: normalizeToolSchema({
    name: "write_file",
    description: "Write or replace file contents in the workspace.",
    input_schema: {
      type: "object",
      required: ["path", "content"],
      properties: {
        path: { type: "string" },
        content: { type: "string" },
      },
      additionalProperties: false,
    },
    risk_profile: {
      level: "medium",
      reversible: true,
      scope: "workspace",
      requires_human_approval: true,
    },
  }),
  bash: normalizeToolSchema({
    name: "bash",
    description: "Execute a shell command in the workspace context.",
    input_schema: {
      type: "object",
      required: ["command"],
      properties: {
        command: { type: "string" },
        cwd: { type: "string" },
      },
      additionalProperties: false,
    },
    risk_profile: {
      level: "high",
      reversible: false,
      scope: "workspace",
      requires_human_approval: true,
    },
  }),
  quant_briefing: normalizeToolSchema({
    name: "quant_briefing",
    description: "Generate a quant briefing from the queue-driven quant pipeline.",
    input_schema: {
      type: "object",
      properties: {
        goal: { type: "string" },
        symbol: { type: "string" },
        market: { type: "string" },
        risk_profile: { type: "string" },
      },
      additionalProperties: true,
    },
    risk_profile: {
      level: "low",
      reversible: true,
      scope: "analysis",
      requires_human_approval: false,
    },
  }),
};
