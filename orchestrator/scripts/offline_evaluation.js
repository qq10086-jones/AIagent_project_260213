import { classifyTask } from "../src/vnext/brain_router_classifier.js";

const testCases = [
  { id: "R001", prompt: "Add a button to the login page", expected_shape: "single_branch_safe", expected_tier: "fast_low_cost" },
  { id: "R002", prompt: "Fix the bug in the payment gateway and update the database schema, this is critical.", expected_shape: "high_risk_release_sensitive", expected_tier: "deep_reasoning" },
  { id: "R003", prompt: "Add a new endpoint for fetching users and update the frontend dashboard to display them.", expected_shape: "dual_branch_parallel_candidate", expected_tier: "balanced_default" },
  { id: "R004", prompt: "Migrate the entire monolith to a new microservices architecture using GraphQL.", expected_shape: "architectural_orchestration_required", expected_tier: "deep_reasoning" },
  { id: "R005", prompt: "Fix", expected_shape: "single_branch_safe", expected_tier: "balanced_default" } // Ambiguous, low confidence
];

console.log("==========================================");
console.log("WS-28-04: Brain Router Offline Evaluation");
console.log("==========================================\n");

let highRiskMisroutes = 0;
let lowConfidenceCount = 0;
let parallelCandidateCorrect = 0;
let parallelCandidateTotal = 0;

for (const tc of testCases) {
  const result = classifyTask(tc.prompt);
  
  if (result.confidence_band === "low") {
    lowConfidenceCount++;
  }
  
  if (tc.expected_shape === "high_risk_release_sensitive" && result.parallel_candidate) {
    highRiskMisroutes++;
  }

  if (tc.expected_shape === "dual_branch_parallel_candidate") {
    parallelCandidateTotal++;
    if (result.work_shape === "dual_branch_parallel_candidate") {
      parallelCandidateCorrect++;
    }
  }

  console.log(`[Case ${tc.id}]`);
  console.log(`Prompt: "${tc.prompt}"`);
  console.log(`Classified Shape:  ${result.work_shape} (Expected: ${tc.expected_shape})`);
  console.log(`Confidence:        ${result.confidence_band} (${result.confidence})`);
  console.log(`Model Tier:        ${result.model_tier}`);
  console.log(`Decision:          ${result.final_execution_decision}`);
  console.log("------------------------------------------");
}

const highRiskMisrouteRate = (highRiskMisroutes / testCases.length) * 100;
const lowConfidenceRatio = (lowConfidenceCount / testCases.length) * 100;
const precision = parallelCandidateTotal > 0 ? (parallelCandidateCorrect / parallelCandidateTotal) * 100 : 100;

console.log("\n[Metrics against targets]");
console.log(`High-risk misroute rate: ${highRiskMisrouteRate}% (Target < 2%)`);
console.log(`Low-confidence fallback ratio: ${lowConfidenceRatio}% (Target 10%-40%)`);
console.log(`Parallel candidate precision: ${precision}% (Target > 85%)`);

console.log("\nIf these metrics are satisfactory, request Architect approval to proceed to WS-29 (Runtime Integration).");
