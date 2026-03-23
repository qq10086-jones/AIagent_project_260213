import { createTaskEnvelope } from "./task_envelope.js";
import { applyRoutingPolicy } from "./brain_router_policy.js";
import { classifyTask } from "./brain_router_classifier.js";
import { resolveCodingProjectType, resolveCodingWorkflowId } from "./coding_project_type.js";

const CODING_RE_EN = /\b(build|implement|feature|fix|bug|patch|refactor|frontend|backend|full[- ]?stack|api|database|repo|code|coding|pm|architect|ui|qa)\b/i;
const CODING_RE_ZH = /(测试|修复|开发|功能|需求|前端|后端|架构|设计文档|网站|做个|制作|系统|平台|网页|小程序)/;
const QUANT_RE_EN = /\b(quant|ticker|stock|portfolio|market|trading|alpha|strategy|backtest)\b/i;
const QUANT_RE_ZH = /(日股|美股|选股|量化|股票|仓位)/;
const DOCS_RE_EN = /\b(doc|docs|design|prd|spec|proposal|milestone|roadmap)\b/i;
const DOCS_RE_ZH = /(文档|方案|规划|拆解|需求文档)/;
const RESEARCH_RE_EN = /\b(research|compare|comparison|benchmark|investigate|analysis memo)\b/i;
const RESEARCH_RE_ZH = /(调研|对比|评估)/;
const OPS_RE_EN = /\b(deploy|docker|k8s|infra|ops|operation|monitor|log|restart)\b/i;
const OPS_RE_ZH = /(部署|运维|日志)/;
const SIMPLE_CODING_RE_EN = /\b(fix|bug|patch|small|minor)\b/i;
const SIMPLE_CODING_RE_ZH = /(单个|简单|小改|修补)/;
const COMPLEX_CUE_RE_EN = /\b(system|platform|workflow|multi[- ]?agent|full project|mvp|end[- ]?to[- ]?end|architecture|orchestrat)\b/i;
const COMPLEX_CUE_RE_ZH = /(多角色|全栈|项目|系统|工作流)/;

function countWords(text) {
  return String(text || "").trim().split(/\s+/).filter(Boolean).length;
}

function inferComplexity(text) {
  const normalized = String(text || "");
  if (COMPLEX_CUE_RE_EN.test(normalized) || COMPLEX_CUE_RE_ZH.test(normalized) || normalized.length > 220 || countWords(normalized) > 35) {
    return "complex";
  }
  if (SIMPLE_CODING_RE_EN.test(normalized) || SIMPLE_CODING_RE_ZH.test(normalized) || normalized.length < 80) {
    return "simple";
  }
  return "medium";
}

function inferIntentFromText(text) {
  const normalized = String(text || "");
  if (CODING_RE_EN.test(normalized) || CODING_RE_ZH.test(normalized)) return "coding";
  if (QUANT_RE_EN.test(normalized) || QUANT_RE_ZH.test(normalized)) return "quant";
  if (DOCS_RE_EN.test(normalized) || DOCS_RE_ZH.test(normalized)) return "docs";
  if (RESEARCH_RE_EN.test(normalized) || RESEARCH_RE_ZH.test(normalized)) return "research";
  if (OPS_RE_EN.test(normalized) || OPS_RE_ZH.test(normalized)) return "ops";
  return "unknown";
}

function normalizeLegacyIntent(parsed = {}) {
  const toolName = String(parsed.tool_name || "").trim();
  if (!toolName) return null;
  if (toolName.startsWith("quant.")) return { intent: "quant", tool_name: toolName };
  if (toolName.startsWith("news.") || toolName.startsWith("github.") || toolName.startsWith("web.")) {
    return { intent: "research", tool_name: toolName };
  }
  if (toolName.startsWith("coding.")) return { intent: "coding", tool_name: toolName };
  if (toolName.startsWith("openclaw.")) return { intent: "ops", tool_name: toolName };
  return null;
}

function normalizeAnalyzerIntent(parsed = {}) {
  const intent = String(parsed.intent || "").trim().toLowerCase();
  const modeSuggested = String(parsed.mode_suggested || "").trim().toLowerCase();
  const requiresTools = Boolean(parsed.requires_tools);

  if (!requiresTools && modeSuggested === "chat") return "chat";
  if (intent === "chat" || intent === "qa") return "chat";
  if (intent === "quant" || intent === "quant_research") return "quant";
  if (intent === "research" || intent === "web_search") return "research";
  if (intent === "docs") return "docs";
  if (intent === "coding") return "coding";
  if (intent === "ops") return "ops";
  return null;
}

function resolveTargetTeam(intent) {
  if (intent === "coding") return "coding_team";
  if (intent === "quant") return "quant_team";
  if (intent === "docs" || intent === "research") return "document_team";
  if (intent === "ops") return "ops_team";
  if (intent === "chat") return "brain";
  return "brain";
}

function resolveExpectedOutputs(intent, decision) {
  if (decision === "direct_reply") return ["direct_reply"];
  if (decision === "clarification_required") return ["clarification_prompt"];
  if (intent === "coding" && decision === "orchestrated_workflow") {
    return ["design_doc", "task_breakdown", "repo_changes", "tests", "qa_summary"];
  }
  if (intent === "coding") return ["repo_changes", "execution_log"];
  if (intent === "quant") return ["analysis_report"];
  if (intent === "docs") return ["design_doc", "task_breakdown"];
  if (intent === "research") return ["research_brief", "recommendation_memo"];
  if (intent === "ops") return ["run_log", "execution_summary"];
  return ["direct_reply"];
}

function resolveExecutionPlan({ intent, complexity, legacyIntent, rawInput, registry }) {
  if (intent === "chat") {
    return { mode: "direct_reply" };
  }

  if (intent === "coding") {
    const workflowId = resolveCodingWorkflowId(registry);
    const workflowProjectType = resolveCodingProjectType(rawInput || legacyIntent?.raw_input || "", registry);
    if (complexity === "simple") {
      return {
        mode: "single_agent",
        tool_name: "coding.delegate",
        project_type: workflowProjectType,
      };
    }
    return {
      mode: "orchestrated_workflow",
      workflow_id: workflowId,
      project_type: workflowProjectType,
    };
  }

  if (intent === "quant") {
    return {
      mode: "single_agent",
      tool_name: legacyIntent?.tool_name || "quant.deep_analysis",
      project_type: "quant_execution",
    };
  }

  if (intent === "docs" || intent === "research") {
    return {
      mode: "single_agent",
      tool_name: "coding.delegate",
      project_type: resolveCodingProjectType("", registry),
    };
  }

  if (intent === "ops") {
    return {
      mode: "single_agent",
      tool_name: "openclaw.run",
      project_type: "coding_task",
    };
  }

  return { mode: "human_review_required" };
}

function decisionFromPlan(plan) {
  if (plan.mode === "direct_reply") return "direct_reply";
  if (plan.mode === "clarification_required") return "clarification_required";
  if (plan.mode === "single_agent") return "single_agent";
  if (plan.mode === "orchestrated_workflow") return "orchestrated_workflow";
  return "human_review_required";
}

export function routeTaskRequest({
  source,
  raw_input,
  normalized_input,
  context = {},
  analyzerResult = undefined,
  registry,
}) {
  const heuristicIntent = inferIntentFromText(raw_input);
  const legacyIntent = normalizeLegacyIntent(analyzerResult || {});
  if (legacyIntent) legacyIntent.raw_input = raw_input;
  const analyzerIntent = normalizeAnalyzerIntent(analyzerResult || {});
  const intent = legacyIntent?.intent || analyzerIntent || heuristicIntent;
  const complexity = inferComplexity(raw_input);
  let executionPlan = resolveExecutionPlan({ intent, complexity, legacyIntent, rawInput: raw_input, registry });
  let decision = decisionFromPlan(executionPlan);
  let finalIntent = intent;

  const normalizedAnalyzerResult = analyzerResult
    ? {
        ...analyzerResult,
        intent: analyzerIntent || String(analyzerResult.intent || "").trim().toLowerCase() || "unknown",
      }
    : analyzerResult;

  // Apply deterministic policy override layer (WS-13)
  // analyzerResult=undefined → heuristic-only (no LLM called), skip LLM-dependent rules
  // analyzerResult=null → LLM was called and failed (P-05 applies)
  const policyResult = applyRoutingPolicy(raw_input, normalizedAnalyzerResult, intent);
  if (policyResult.override) {
    finalIntent = policyResult.intent;
    if (policyResult.decision === "orchestrated_workflow") {
      const wfId = resolveCodingWorkflowId(registry);
      executionPlan = {
        mode: "orchestrated_workflow",
        workflow_id: wfId,
        project_type: resolveCodingProjectType(raw_input, registry),
      };
    } else if (policyResult.decision === "direct_reply") {
      executionPlan = { mode: "direct_reply" };
    } else if (policyResult.decision === "clarification_required") {
      executionPlan = {
        mode: "clarification_required",
        clarification_prompt: policyResult.clarification_prompt,
      };
    } else {
      executionPlan = { mode: policyResult.decision };
    }
    decision = decisionFromPlan(executionPlan);
  }

  const classifierResult = classifyTask(raw_input);

  const requiresOrchestration = decision === "orchestrated_workflow";
  const envelope = createTaskEnvelope({
    source,
    raw_input,
    normalized_input,
    intent: finalIntent,
    sub_intent: complexity === "simple" ? "simple_request" : "project_workflow",
    requires_orchestration: requiresOrchestration,
    target_team: resolveTargetTeam(finalIntent),
    expected_outputs: resolveExpectedOutputs(finalIntent, decision),
    constraints: {
      local_only: true,
      approval_mode: "manual",
      risk_level: finalIntent === "ops" ? "high" : (finalIntent === "coding" ? "medium" : "low"),
      complexity,
    },
    context,
    decision,
    execution_plan: executionPlan,
  });

  envelope.classifier_result = classifierResult;

  return {
    decision,
    route: {
      intent: finalIntent,
      complexity,
      target_team: envelope.target_team,
      requires_orchestration: envelope.requires_orchestration,
      expected_outputs: envelope.expected_outputs,
      execution_plan: executionPlan,
      classifier_result: classifierResult,
    },
    task_envelope: envelope,
    clarification_prompt: executionPlan.clarification_prompt,
  };
}
