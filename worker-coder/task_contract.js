import { encode } from "gpt-tokenizer";
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

// --- Token counting (Phase 0, P0-3) ---

export function countTokens(text) {
  return encode(String(text || "")).length;
}

export function validateContextTokenBudget({ contextText, maxTokens }) {
  if (!maxTokens || maxTokens <= 0) return { ok: true, usage: 0, max: 0, overflow: 0 };
  const usage = countTokens(contextText);
  return {
    ok: usage <= maxTokens,
    usage,
    max: maxTokens,
    overflow: Math.max(0, usage - maxTokens),
  };
}

export function truncateContextForTokenBudget({ contextPacket, repoMap, maxTokens }) {
  if (!maxTokens || maxTokens <= 0) return { contextPacket, repoMap, truncated: false, tokenUsage: 0 };
  const measure = () => countTokens(JSON.stringify(contextPacket) + JSON.stringify(repoMap));
  let usage = measure();
  if (usage <= maxTokens) return { contextPacket, repoMap, truncated: false, tokenUsage: usage };

  // 按优先级逐层删减
  const cp = contextPacket && typeof contextPacket === "object" ? { ...contextPacket } : {};
  const rm = repoMap && typeof repoMap === "object" ? { ...repoMap } : {};

  // 1. memory_hints
  if (Array.isArray(cp.memory_hints) && cp.memory_hints.length > 0) {
    cp.memory_hints = [];
    usage = countTokens(JSON.stringify(cp) + JSON.stringify(rm));
    if (usage <= maxTokens) return { contextPacket: cp, repoMap: rm, truncated: true, tokenUsage: usage };
  }

  // 2. symbol_hints (in repoMap)
  if (Array.isArray(rm.symbol_hints) && rm.symbol_hints.length > 0) {
    rm.symbol_hints = [];
    usage = countTokens(JSON.stringify(cp) + JSON.stringify(rm));
    if (usage <= maxTokens) return { contextPacket: cp, repoMap: rm, truncated: true, tokenUsage: usage };
  }

  // 3. toolchain_facts (in repoMap)
  if (Array.isArray(rm.toolchain_facts) && rm.toolchain_facts.length > 0) {
    rm.toolchain_facts = [];
    usage = countTokens(JSON.stringify(cp) + JSON.stringify(rm));
    if (usage <= maxTokens) return { contextPacket: cp, repoMap: rm, truncated: true, tokenUsage: usage };
  }

  // 4. candidate_files 截断到一半
  if (Array.isArray(rm.candidate_files) && rm.candidate_files.length > 2) {
    rm.candidate_files = rm.candidate_files.slice(0, Math.max(2, Math.floor(rm.candidate_files.length / 2)));
    usage = countTokens(JSON.stringify(cp) + JSON.stringify(rm));
  }

  return { contextPacket: cp, repoMap: rm, truncated: true, tokenUsage: usage };
}
