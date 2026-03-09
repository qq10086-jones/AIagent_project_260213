/**
 * brain_router_classifier.js
 *
 * WS-28-02: Brain Router Classification v1
 * Evaluates a task request and produces a routing decision recommendation based
 * on the M7 Classification Taxonomy.
 */

const HIGH_RISK_KEYWORDS = ["stripe", "payment", "auth", "2fa", "security", "core db", "migration", "production data"];
const ARCH_KEYWORDS = ["architecture", "refactor", "graphql", "migrate", "subsystem", "framework", "dynamic routing"];
const DUAL_BRANCH_KEYWORDS = ["api", "endpoint", "database", "schema", "model", "controller", "fullstack", "both backend and frontend"];
const UI_KEYWORDS = ["button", "color", "css", "html", "react", "vue", "frontend", "ui", "ux", "page", "view"];
const BE_KEYWORDS = ["worker", "job", "queue", "cache", "redis", "query", "sql"];
const INFRA_KEYWORDS = ["docker", "k8s", "ci/cd", "pipeline", "deploy", "nginx", "proxy"];

function calculateConfidence(text) {
  const wordCount = String(text || "").trim().split(/\s+/).filter(Boolean).length;
  if (wordCount > 10) return 0.9;
  if (wordCount > 5) return 0.7;
  if (wordCount > 2) return 0.4;
  return 0.1;
}

function getConfidenceBand(confidence) {
  if (confidence >= 0.8) return "high";
  if (confidence >= 0.5) return "medium";
  return "low";
}

function identifyDomainLead(text) {
  const normalized = text.toLowerCase();
  if (ARCH_KEYWORDS.some(k => normalized.includes(k))) return "architecture";
  if (INFRA_KEYWORDS.some(k => normalized.includes(k))) return "infra";
  
  const hasUI = UI_KEYWORDS.some(k => normalized.includes(k));
  const hasBE = BE_KEYWORDS.some(k => normalized.includes(k)) || DUAL_BRANCH_KEYWORDS.some(k => normalized.includes(k));

  if (hasUI && hasBE) return "fullstack";
  if (hasUI) return "fe_led";
  if (hasBE) return "be_led";
  return "fullstack"; // Default when ambiguous but within a coding context
}

function classifyWorkShape(text) {
  const normalized = text.toLowerCase();
  
  if (HIGH_RISK_KEYWORDS.some(k => normalized.includes(k))) {
    return "high_risk_release_sensitive";
  }
  if (ARCH_KEYWORDS.some(k => normalized.includes(k))) {
    return "architectural_orchestration_required";
  }
  
  const domainLead = identifyDomainLead(text);
  if (domainLead === "fullstack") {
    return "dual_branch_parallel_candidate";
  }
  
  return "single_branch_safe";
}

function determineModelTier(workShape) {
  if (workShape === "architectural_orchestration_required" || workShape === "high_risk_release_sensitive") {
    return "deep_reasoning";
  }
  if (workShape === "single_branch_safe") {
    return "fast_low_cost";
  }
  return "balanced_default";
}

function determineExecutionDecision(workShape, confidenceBand) {
  // Safe Fallback (Section 14.4)
  if (confidenceBand === "low" || confidenceBand === "unavailable") {
    return {
      parallel_candidate: false,
      final_execution_decision: "forced_sequential",
      routing_decision_source: confidenceBand === "unavailable" ? "classifier_unavailable_fallback" : "classifier_low_confidence_fallback",
      deny_or_degrade_reason: "Confidence below safe threshold"
    };
  }

  if (workShape === "dual_branch_parallel_candidate") {
    return {
      parallel_candidate: true,
      final_execution_decision: "gated_parallel_allowed",
      routing_decision_source: "classifier_recommended_parallel",
      deny_or_degrade_reason: null
    };
  }

  return {
    parallel_candidate: false,
    final_execution_decision: "forced_sequential",
    routing_decision_source: "classifier_recommended_sequential",
    deny_or_degrade_reason: "Work shape does not require parallel execution"
  };
}

export function classifyTask(taskInput, classifierStatus = "healthy") {
  const version = "1.0.0";
  const featureSnapshotRef = `snap-${Date.now()}`;

  // Handle unavailable degradation (WS-29-01 / Section 5.4)
  if (classifierStatus !== "healthy") {
    return {
      classifier_version: version,
      feature_snapshot_ref: featureSnapshotRef,
      work_shape: "single_branch_safe",
      domain_lead: "fullstack",
      confidence: 0,
      confidence_band: "unavailable",
      parallel_candidate: false,
      model_tier: "balanced_default",
      required_contracts: [],
      safety_override_result: "allowed",
      final_execution_decision: "forced_sequential",
      routing_decision_source: "classifier_unavailable_fallback",
      deny_or_degrade_reason: `Classifier status is ${classifierStatus}`
    };
  }

  const confidence = calculateConfidence(taskInput);
  const confidenceBand = getConfidenceBand(confidence);
  
  // Apply ambiguity fallback
  let workShape = classifyWorkShape(taskInput);
  let domainLead = identifyDomainLead(taskInput);
  let modelTier = determineModelTier(workShape);
  
  if (confidenceBand === "low") {
    workShape = "single_branch_safe";
    modelTier = "balanced_default";
  }

  const decision = determineExecutionDecision(workShape, confidenceBand);

  return {
    classifier_version: version,
    feature_snapshot_ref: featureSnapshotRef,
    work_shape: workShape,
    domain_lead: domainLead,
    confidence: confidence,
    confidence_band: confidenceBand,
    parallel_candidate: decision.parallel_candidate,
    model_tier: modelTier,
    required_contracts: ["coding_team_v0"], // Default for M7 limits
    safety_override_result: "allowed", // To be filled by higher layers if needed
    final_execution_decision: decision.final_execution_decision,
    routing_decision_source: decision.routing_decision_source,
    deny_or_degrade_reason: decision.deny_or_degrade_reason
  };
}
