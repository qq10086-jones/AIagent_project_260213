import crypto from "crypto";
import { v4 as uuidv4 } from "uuid";

export const TASK_INTENTS = ["chat", "coding", "quant", "docs", "research", "ops", "unknown"];
export const TASK_DECISIONS = ["direct_reply", "single_agent", "orchestrated_workflow", "human_review_required", "clarification_required"];

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function validateTaskEnvelope(envelope) {
  const errors = [];
  if (!isPlainObject(envelope)) {
    return { ok: false, errors: ["task envelope must be an object"] };
  }

  if (!String(envelope.task_id || "").trim()) errors.push("task_id is required");
  if (!String(envelope.source || "").trim()) errors.push("source is required");
  if (!String(envelope.raw_input || "").trim()) errors.push("raw_input is required");
  if (!TASK_INTENTS.includes(String(envelope.intent || ""))) errors.push(`intent must be one of: ${TASK_INTENTS.join(", ")}`);
  if (typeof envelope.requires_orchestration !== "boolean") errors.push("requires_orchestration must be boolean");
  if (!String(envelope.target_team || "").trim()) errors.push("target_team is required");
  if (!Array.isArray(envelope.expected_outputs)) errors.push("expected_outputs must be an array");
  if (!isPlainObject(envelope.constraints)) errors.push("constraints must be an object");
  if (!isPlainObject(envelope.context)) errors.push("context must be an object");
  if (envelope.decision && !TASK_DECISIONS.includes(String(envelope.decision || ""))) {
    errors.push(`decision must be one of: ${TASK_DECISIONS.join(", ")}`);
  }
  if (envelope.evidence_id !== undefined && !String(envelope.evidence_id || "").trim()) {
    errors.push("evidence_id must be non-empty when provided");
  }
  if (envelope.replay_tag !== undefined && !String(envelope.replay_tag || "").trim()) {
    errors.push("replay_tag must be non-empty when provided");
  }

  return { ok: errors.length === 0, errors };
}

function buildReplayTag({ raw_input = "", normalized_input = {}, intent = "", decision = "", execution_plan = {} }) {
  const canonical = JSON.stringify({
    raw_input: String(raw_input || ""),
    normalized_input: isPlainObject(normalized_input) ? normalized_input : {},
    intent: String(intent || ""),
    decision: String(decision || ""),
    execution_plan: isPlainObject(execution_plan) ? execution_plan : {},
  });
  return crypto.createHash("sha256").update(canonical).digest("hex");
}

export function createTaskEnvelope(input = {}) {
  const decision = input.decision ? String(input.decision) : undefined;
  const executionPlan = isPlainObject(input.execution_plan) ? input.execution_plan : undefined;
  const envelope = {
    task_id: String(input.task_id || uuidv4()),
    source: String(input.source || "api"),
    raw_input: String(input.raw_input || ""),
    normalized_input: isPlainObject(input.normalized_input) ? input.normalized_input : { text: String(input.raw_input || "") },
    intent: String(input.intent || "unknown"),
    sub_intent: String(input.sub_intent || "general"),
    requires_orchestration: Boolean(input.requires_orchestration),
    target_team: String(input.target_team || "brain"),
    expected_outputs: Array.isArray(input.expected_outputs) ? input.expected_outputs : [],
    constraints: isPlainObject(input.constraints) ? input.constraints : {},
    context: isPlainObject(input.context) ? input.context : {},
    decision,
    execution_plan: executionPlan,
    evidence_id:
      decision === "single_agent"
        ? String(input.evidence_id || uuidv4().replace(/-/g, ""))
        : (input.evidence_id ? String(input.evidence_id) : undefined),
    replay_tag:
      decision === "single_agent"
        ? String(input.replay_tag || buildReplayTag({
          raw_input: input.raw_input,
          normalized_input: input.normalized_input,
          intent: input.intent,
          decision,
          execution_plan: executionPlan,
        }))
        : (input.replay_tag ? String(input.replay_tag) : undefined),
  };

  const validation = validateTaskEnvelope(envelope);
  if (!validation.ok) {
    const err = new Error(`TASK_ENVELOPE_INVALID: ${validation.errors.join("; ")}`);
    err.code = "TASK_ENVELOPE_INVALID";
    err.details = validation.errors;
    throw err;
  }
  return envelope;
}
