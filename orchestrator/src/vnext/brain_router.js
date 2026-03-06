import { createTaskEnvelope } from "./task_envelope.js";

const CODING_RE = /\b(build|implement|feature|fix|bug|patch|refactor|frontend|backend|full[- ]?stack|api|database|repo|code|coding|pm|architect|ui|qa|测试|修复|开发|功能|需求|前端|后端|架构|设计文档)\b/i;
const QUANT_RE = /\b(quant|ticker|stock|portfolio|market|trading|alpha|strategy|backtest|日股|美股|选股|量化|股票|仓位)\b/i;
const DOCS_RE = /\b(doc|docs|design|prd|spec|proposal|milestone|roadmap|文档|方案|规划|拆解|需求文档)\b/i;
const RESEARCH_RE = /\b(research|compare|comparison|benchmark|investigate|analysis memo|调研|对比|评估)\b/i;
const OPS_RE = /\b(deploy|docker|k8s|infra|ops|operation|monitor|log|restart|部署|运维|日志)\b/i;
const SIMPLE_CODING_RE = /\b(fix|bug|patch|small|minor|单个|简单|小改|修补)\b/i;
const COMPLEX_CUE_RE = /\b(system|platform|workflow|multi[- ]?agent|full project|mvp|end[- ]?to[- ]?end|architecture|orchestrat|多角色|全栈|项目|系统|工作流)\b/i;

function countWords(text) {
  return String(text || "").trim().split(/\s+/).filter(Boolean).length;
}

function inferComplexity(text) {
  const normalized = String(text || "");
  if (COMPLEX_CUE_RE.test(normalized) || normalized.length > 220 || countWords(normalized) > 35) {
    return "complex";
  }
  if (SIMPLE_CODING_RE.test(normalized) || normalized.length < 80) {
    return "simple";
  }
  return "medium";
}

function inferIntentFromText(text) {
  const normalized = String(text || "");
  if (CODING_RE.test(normalized)) return "coding";
  if (QUANT_RE.test(normalized)) return "quant";
  if (DOCS_RE.test(normalized)) return "docs";
  if (RESEARCH_RE.test(normalized)) return "research";
  if (OPS_RE.test(normalized)) return "ops";
  return "chat";
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

function resolveExecutionPlan({ intent, complexity, legacyIntent, registry }) {
  if (intent === "chat") {
    return { mode: "direct_reply" };
  }

  if (intent === "coding") {
    const workflowId = registry.project_types?.coding_task?.default_workflow || "coding_team_v0";
    const workflowProjectType = registry.workflows?.[workflowId]?.project_type || "webapp_crm";
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
      project_type: registry.workflows?.coding_team_v0?.project_type || "webapp_crm",
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
  if (plan.mode === "single_agent") return "single_agent";
  if (plan.mode === "orchestrated_workflow") return "orchestrated_workflow";
  return "human_review_required";
}

export function routeTaskRequest({
  source,
  raw_input,
  normalized_input,
  context = {},
  analyzerResult = null,
  registry,
}) {
  const heuristicIntent = inferIntentFromText(raw_input);
  const legacyIntent = normalizeLegacyIntent(analyzerResult || {});
  const intent = legacyIntent?.intent || heuristicIntent;
  const complexity = inferComplexity(raw_input);
  const executionPlan = resolveExecutionPlan({ intent, complexity, legacyIntent, registry });
  const decision = decisionFromPlan(executionPlan);
  const requiresOrchestration = decision === "orchestrated_workflow";
  const envelope = createTaskEnvelope({
    source,
    raw_input,
    normalized_input,
    intent,
    sub_intent: complexity === "simple" ? "simple_request" : "project_workflow",
    requires_orchestration: requiresOrchestration,
    target_team: resolveTargetTeam(intent),
    expected_outputs: resolveExpectedOutputs(intent, decision),
    constraints: {
      local_only: true,
      approval_mode: "manual",
      risk_level: intent === "ops" ? "high" : (intent === "coding" ? "medium" : "low"),
      complexity,
    },
    context,
    decision,
    execution_plan: executionPlan,
  });

  return {
    decision,
    route: {
      intent,
      complexity,
      target_team: envelope.target_team,
      requires_orchestration: envelope.requires_orchestration,
      expected_outputs: envelope.expected_outputs,
      execution_plan: executionPlan,
    },
    task_envelope: envelope,
  };
}
