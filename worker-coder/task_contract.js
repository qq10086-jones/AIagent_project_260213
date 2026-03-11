import { loadWorkerCodingTaskClassesOrThrow } from "./worker_coding_task_classes.js";

const { taskClassSet: WORKER_CODING_TASK_CLASS_SET } = loadWorkerCodingTaskClassesOrThrow();

export function normalizeTaskClass(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return WORKER_CODING_TASK_CLASS_SET.has(normalized) ? normalized : null;
}

export function normalizeContextEnvelope(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    return null;
  }
  const maxFiles = Number(input.max_files);
  const maxTokens = Number(input.max_tokens);
  const dependencyDepth = Number(input.dependency_depth);
  const contextSource = String(input.context_source || "").trim().toLowerCase();
  return {
    max_files: Number.isFinite(maxFiles) ? Math.max(0, Math.trunc(maxFiles)) : null,
    max_tokens: Number.isFinite(maxTokens) ? Math.max(0, Math.trunc(maxTokens)) : null,
    dependency_depth: Number.isFinite(dependencyDepth) ? Math.max(0, Math.trunc(dependencyDepth)) : null,
    context_source: ["manual", "template", "automated"].includes(contextSource) ? contextSource : null,
  };
}

export function buildTaskContractMetadata({ taskClass, betaTemplateId, contextEnvelope }) {
  const normalizedTaskClass = normalizeTaskClass(taskClass);
  const normalizedEnvelope = normalizeContextEnvelope(contextEnvelope);
  return {
    task_class: normalizedTaskClass,
    beta_template_id: betaTemplateId ? String(betaTemplateId) : null,
    context_envelope: normalizedEnvelope,
  };
}

export function deriveFailureAttribution({ phase, errorCode }) {
  const normalizedPhase = String(phase || "").trim().toLowerCase();
  const normalizedCode = String(errorCode || "").trim().toUpperCase();

  if (normalizedPhase === "context_guard" || normalizedCode === "E_CONTEXT_ENVELOPE_EXCEEDED") {
    return "context_failure";
  }
  if (normalizedPhase === "verification") {
    return "verification_failure";
  }
  if (["provider_validation", "retry_budget"].includes(normalizedPhase)) {
    return "infrastructure_failure";
  }
  if (normalizedCode.includes("TIMEOUT") || normalizedCode === "E_PROVIDER_UNAVAILABLE" || normalizedCode === "E_DELEGATE_FAILED") {
    return "infrastructure_failure";
  }
  return "coding_logic_failure";
}
