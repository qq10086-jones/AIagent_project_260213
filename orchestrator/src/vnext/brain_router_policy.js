/**
 * brain_router_policy.js
 *
 * WS-13-02 / WS-16-06:
 * Deterministic routing policy layer on top of analyzer output.
 */

const QUANT_TRADE_KEYWORDS = [
  "buy",
  "sell",
  "trade",
  "execute order",
  "market order",
  "broker",
];

const EXECUTION_CUE_RE = /\b(build|implement|create|fix|develop|deploy|refactor|feature|app|api|backend|frontend|workflow|system)\b/i;
const CLARIFICATION_PROMPT = "检测到这可能是一个执行类任务。是否要交给 Coding Team 继续处理？回复 /yes 可按编码工作流继续。";

function logRule(ruleId, override, decision) {
  console.info(`[brain_router_policy] rule=${ruleId} override=${override} final=${decision}`);
}

function applyRoutingPolicy(rawInput, llmResult, heuristicIntent = null) {
  const input = String(rawInput || "");

  if (input.trimStart().toLowerCase().startsWith("/coder")) {
    const result = {
      applied_rule: "P-01",
      decision: "orchestrated_workflow",
      intent: "coding",
      override: true,
    };
    logRule(result.applied_rule, result.override, result.decision);
    return result;
  }

  const tokens = input.trim().split(/\s+/).filter(Boolean);
  const isChinese = /[\u4e00-\u9fa5]/.test(input);
  const effectiveLength = isChinese ? input.trim().length : tokens.length;
  
  if (effectiveLength < 3) {
    const result = {
      applied_rule: "P-02",
      decision: "direct_reply",
      intent: "chat",
      override: true,
    };
    logRule(result.applied_rule, result.override, result.decision);
    return result;
  }

  if (llmResult === undefined) {
    if (heuristicIntent === "unknown" && EXECUTION_CUE_RE.test(input)) {
      const result = {
        applied_rule: "P-06",
        decision: "clarification_required",
        intent: "unknown",
        override: true,
        clarification_prompt: CLARIFICATION_PROMPT,
      };
      logRule(result.applied_rule, result.override, result.decision);
      return result;
    }
    if (heuristicIntent === "unknown") {
      const result = {
        applied_rule: "P-04",
        decision: "direct_reply",
        intent: "chat",
        override: true,
      };
      logRule(result.applied_rule, result.override, result.decision);
      return result;
    }
    logRule(null, false, "pass-through");
    return { applied_rule: null, decision: null, intent: null, override: false };
  }

  if (llmResult === null) {
    const result = {
      applied_rule: "P-05",
      decision: "direct_reply",
      intent: "chat",
      override: true,
      error_note: "LLM_CLASSIFICATION_FAILED",
    };
    logRule(result.applied_rule, result.override, result.decision);
    return result;
  }

  if (llmResult.intent === "quant") {
    const lowerInput = input.toLowerCase();
    const hasTradeKeyword = QUANT_TRADE_KEYWORDS.some((kw) => lowerInput.includes(kw));
    if (hasTradeKeyword) {
      const result = {
        applied_rule: "P-03",
        decision: "human_review_required",
        intent: "quant",
        override: true,
      };
      logRule(result.applied_rule, result.override, result.decision);
      return result;
    }
  }

  const intentIsUnknown = llmResult.intent === "unknown";
  const confidenceLow = typeof llmResult.confidence === "number" && llmResult.confidence < 0.4;
  if (intentIsUnknown || confidenceLow) {
    const result = {
      applied_rule: "P-04",
      decision: "direct_reply",
      intent: "chat",
      override: true,
    };
    logRule(result.applied_rule, result.override, result.decision);
    return result;
  }

  const result = {
    applied_rule: null,
    decision: llmResult.decision || "direct_reply",
    intent: llmResult.intent || "chat",
    override: false,
  };
  logRule(result.applied_rule, result.override, result.decision);
  return result;
}

export { applyRoutingPolicy, CLARIFICATION_PROMPT };
