function isObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function validateTaskEnvelopeShape(taskEnvelope) {
  if (!isObject(taskEnvelope)) return ["task_envelope must be an object"];
  const errors = [];
  if (!String(taskEnvelope.task_id || "").trim()) errors.push("task_envelope.task_id is required");
  if (!String(taskEnvelope.source || "").trim()) errors.push("task_envelope.source is required");
  if (!String(taskEnvelope.raw_input || "").trim()) errors.push("task_envelope.raw_input is required");
  if (!isObject(taskEnvelope.normalized_input)) errors.push("task_envelope.normalized_input must be an object");
  if (!String(taskEnvelope.intent || "").trim()) errors.push("task_envelope.intent is required");
  if (typeof taskEnvelope.requires_orchestration !== "boolean") errors.push("task_envelope.requires_orchestration must be boolean");
  if (!String(taskEnvelope.target_team || "").trim()) errors.push("task_envelope.target_team is required");
  if (!Array.isArray(taskEnvelope.expected_outputs)) errors.push("task_envelope.expected_outputs must be an array");
  if (!isObject(taskEnvelope.constraints)) errors.push("task_envelope.constraints must be an object");
  if (!isObject(taskEnvelope.context)) errors.push("task_envelope.context must be an object");
  return errors;
}

export function validateRouteContractResponseShape(payload) {
  const errors = [];
  if (!isObject(payload)) return { ok: false, errors: ["route response must be an object"] };
  if (payload.ok !== true) errors.push("ok must be true");
  if (!isObject(payload.normalized)) errors.push("normalized must be an object");
  if (!String(payload.decision || "").trim()) errors.push("decision is required");
  if (!isObject(payload.route)) errors.push("route must be an object");
  errors.push(...validateTaskEnvelopeShape(payload.task_envelope));
  return { ok: errors.length === 0, errors };
}

export function validateDispatchSuccessResponseShape(payload) {
  const errors = [];
  if (!isObject(payload)) return { ok: false, errors: ["dispatch success response must be an object"] };
  if (payload.ok !== true) errors.push("ok must be true");
  if (!String(payload.response_mode || "").trim()) errors.push("response_mode is required");
  if (!String(payload.run_id || "").trim()) errors.push("run_id is required");
  errors.push(...validateTaskEnvelopeShape(payload.task_envelope));
  return { ok: errors.length === 0, errors };
}

export function validateDispatchErrorResponseShape(payload) {
  const errors = [];
  if (!isObject(payload)) return { ok: false, errors: ["dispatch error response must be an object"] };
  if (payload.ok !== false) errors.push("ok must be false");
  if (payload.response_mode !== "final_completion_reply") errors.push("response_mode must be final_completion_reply");
  if (!String(payload.run_id || "").trim()) errors.push("run_id is required");
  if (!String(payload.error || "").trim()) errors.push("error is required");
  if (!String(payload.error_code || "").trim()) errors.push("error_code is required");
  if (!(payload.task_envelope === null || isObject(payload.task_envelope))) {
    errors.push("task_envelope must be object or null");
  }
  return { ok: errors.length === 0, errors };
}

export function assertRouteContractResponse(payload) {
  const checked = validateRouteContractResponseShape(payload);
  if (!checked.ok) {
    const err = new Error(`ROUTE_CONTRACT_INVALID: ${checked.errors.join("; ")}`);
    err.code = "ROUTE_CONTRACT_INVALID";
    throw err;
  }
  return payload;
}

export function assertDispatchSuccessResponse(payload) {
  const checked = validateDispatchSuccessResponseShape(payload);
  if (!checked.ok) {
    const err = new Error(`DISPATCH_SUCCESS_CONTRACT_INVALID: ${checked.errors.join("; ")}`);
    err.code = "DISPATCH_SUCCESS_CONTRACT_INVALID";
    throw err;
  }
  return payload;
}

export function assertDispatchErrorResponse(payload) {
  const checked = validateDispatchErrorResponseShape(payload);
  if (!checked.ok) {
    const err = new Error(`DISPATCH_ERROR_CONTRACT_INVALID: ${checked.errors.join("; ")}`);
    err.code = "DISPATCH_ERROR_CONTRACT_INVALID";
    throw err;
  }
  return payload;
}
