#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { createParallelRolloutGate } from "../src/domain/parallel_rollout_gate.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const ARTIFACT_DIR = path.join(ROOT, "artifacts/canary/m7_phase_a_advisory");

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function writeJson(p, v) {
  ensureDir(path.dirname(p));
  fs.writeFileSync(p, JSON.stringify(v, null, 2), "utf8");
}

function buildWorkspace({ rolloutOverride = {}, policyOverride = {}, cohortOverride = {} } = {}) {
  const dir = fs.mkdtempSync(path.join(ROOT, "artifacts/tmp_canary_m7_phase_a_"));
  const configDir = path.join(dir, "configs");
  fs.mkdirSync(configDir, { recursive: true });

  const rollout = {
    master_enabled: true,
    force_sequential: false,
    dynamic_routing_enabled: true,
    router_mode: "dynamic_routing_advisory",
    circuit_breaker: { activated: false },
    classifier_circuit_breaker: { activated: false },
    last_policy_change: new Date().toISOString(),
    ...rolloutOverride,
  };
  const policy = {
    allowed_workflow_types: ["coding_team_v0"],
    allowed_project_types: ["crm"],
    fe_safe_eligible_input_classes: ["fe_led"],
    ...policyOverride,
  };
  const cohort = {
    allowed_workflow_types: ["coding_team_v0"],
    allowed_project_types: ["crm"],
    allowed_input_classes: ["fe_led"],
    runtime_controls: {
      cohort_enabled: true,
      environment: "staging",
    },
    ...cohortOverride,
  };

  fs.writeFileSync(path.join(configDir, "production_parallel_rollout.json"), JSON.stringify(rollout));
  fs.writeFileSync(path.join(configDir, "parallel_exposure_policy.json"), JSON.stringify(policy));
  fs.writeFileSync(path.join(configDir, "m7_exposure_cohorts.json"), JSON.stringify(cohort));
  return dir;
}

function cleanup(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}

const FE_SAFE_WORKFLOW = {
  steps: [
    { id: "pm_spec" },
    { id: "arch_design" },
    { id: "impl_be" },
    { id: "impl_fe", fe_safe_input_classes: ["fe_led"] },
    { id: "qa_verify" },
    { id: "release_pack" },
  ],
};

const FE_SAFE_RUN = { workflow_id: "coding_team_v0", project_type: "crm", input_class: "fe_led" };
const OUTSIDE_COHORT_RUN = { workflow_id: "coding_team_v0", project_type: "webapp_crm", input_class: "fe_led" };

const results = [];

function check(label, fn) {
  try {
    const detail = fn();
    results.push({ label, status: "PASS", detail: detail || null });
    console.log(`PASS  ${label}`);
  } catch (err) {
    results.push({ label, status: "FAIL", error: String(err.message || err) });
    console.error(`FAIL  ${label}: ${err.message || err}`);
  }
}

check("Advisory-only mode keeps static allow even when classifier recommends sequential", () => {
  const dir = buildWorkspace();
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({
      run: FE_SAFE_RUN,
      workflow: FE_SAFE_WORKFLOW,
      classifier_result: {
        confidence_band: "high",
        final_execution_decision: "forced_sequential",
        model_tier: "deep_reasoning",
      },
    });
    if (result.effective_exposure_decision !== "gated_parallel_allowed") {
      throw new Error(`expected gated_parallel_allowed, got ${result.effective_exposure_decision}`);
    }
    if (result.routing_decision_source !== "dynamic_routing_advisory_only") {
      throw new Error(`expected advisory source, got ${result.routing_decision_source}`);
    }
    return `model_tier=${result.model_tier}`;
  } finally {
    cleanup(dir);
  }
});

check("Advisory-only mode still falls back safely on classifier unavailability", () => {
  const dir = buildWorkspace();
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({
      run: FE_SAFE_RUN,
      workflow: FE_SAFE_WORKFLOW,
      classifier_result: null,
    });
    if (result.effective_exposure_decision !== "gated_parallel_allowed") {
      throw new Error(`expected gated_parallel_allowed, got ${result.effective_exposure_decision}`);
    }
    if (result.routing_decision_source !== "classifier_unavailable_fallback") {
      throw new Error(`expected classifier_unavailable_fallback, got ${result.routing_decision_source}`);
    }
    return "fallback preserved";
  } finally {
    cleanup(dir);
  }
});

check("Dynamic routing stays disabled outside approved cohort even when static policy would allow it", () => {
  const dir = buildWorkspace({
    policyOverride: {
      allowed_project_types: ["crm", "webapp_crm"],
    },
  });
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({
      run: OUTSIDE_COHORT_RUN,
      workflow: FE_SAFE_WORKFLOW,
      classifier_result: {
        confidence_band: "high",
        final_execution_decision: "forced_sequential",
        model_tier: "fast_low_cost",
      },
    });
    if (result.effective_exposure_decision !== "gated_parallel_allowed") {
      throw new Error(`expected static allow to remain, got ${result.effective_exposure_decision}`);
    }
    if (result.routing_decision_source !== "dynamic_routing_disabled") {
      throw new Error(`expected dynamic_routing_disabled, got ${result.routing_decision_source}`);
    }
    return "cohort restriction enforced";
  } finally {
    cleanup(dir);
  }
});

check("Legacy dynamic mode is not required for Phase A advisory canary", () => {
  const dir = buildWorkspace({
    rolloutOverride: {
      router_mode: "dynamic_routing_advisory",
    },
  });
  try {
    const gate = createParallelRolloutGate({ workspaceRoot: dir });
    const result = gate.evaluate({
      run: FE_SAFE_RUN,
      workflow: FE_SAFE_WORKFLOW,
      classifier_result: {
        confidence_band: "high",
        final_execution_decision: "gated_parallel_allowed",
        model_tier: "balanced_default",
      },
    });
    if (result.routing_decision_source !== "dynamic_routing_advisory_only") {
      throw new Error(`unexpected source ${result.routing_decision_source}`);
    }
    return "new advisory router_mode active";
  } finally {
    cleanup(dir);
  }
});

const passed = results.filter((r) => r.status === "PASS").length;
const failed = results.filter((r) => r.status === "FAIL").length;
const artifact = {
  canary: "canary_m7_phase_a_advisory",
  milestone: "post_m8_m7_phase_a",
  timestamp: new Date().toISOString(),
  total: results.length,
  passed,
  failed,
  status: failed === 0 ? "PASS" : "FAIL",
  results,
};

ensureDir(ARTIFACT_DIR);
writeJson(path.join(ARTIFACT_DIR, "canary_m7_phase_a_advisory.json"), artifact);

console.log();
console.log(`M7 Phase A Advisory Canary: ${artifact.status}  (${passed}/${results.length})`);
console.log(`Artifact: orchestrator/artifacts/canary/m7_phase_a_advisory/canary_m7_phase_a_advisory.json`);

if (failed > 0) process.exit(1);
