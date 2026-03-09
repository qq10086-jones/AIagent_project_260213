#!/usr/bin/env node

import fs from "fs";
import path from "path";
import {
  resolveOrchestratorArtifactPath,
  resolveOrchestratorPath,
} from "./_paths.js";

function loadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function exists(filePath) {
  try {
    fs.accessSync(filePath, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function main() {
  const rolloutPath = resolveOrchestratorPath("configs", "production_parallel_rollout.json");
  const cohortPath = resolveOrchestratorPath("configs", "m7_exposure_cohorts.json");
  const policyPath = resolveOrchestratorPath("configs", "parallel_exposure_policy.json");
  const acceleratedReportPath = resolveOrchestratorPath("artifacts", "m6_trial", "accelerated_validation_report_30m.json");
  const liveRuntimeReportPath = resolveOrchestratorPath("artifacts", "canary", "live_vnext_runtime", "live_vnext_runtime_report.json");

  const rollout = loadJson(rolloutPath);
  const cohorts = loadJson(cohortPath);
  const policy = loadJson(policyPath);
  const acceleratedReport = loadJson(acceleratedReportPath);
  const liveRuntimeReport = loadJson(liveRuntimeReportPath);

  const issues = [];
  const readyChecks = {};

  readyChecks.rollout_master_enabled = rollout?.master_enabled === true;
  readyChecks.dynamic_routing_currently_disabled = rollout?.dynamic_routing_enabled === false;
  readyChecks.current_router_mode_static = rollout?.router_mode === "static_policy_only";
  readyChecks.cohort_file_present = Boolean(cohorts);
  readyChecks.cohort_disabled_pending_approval = cohorts?.runtime_controls?.cohort_enabled === false;
  readyChecks.cohort_matches_phase_a =
    JSON.stringify(cohorts?.allowed_workflow_types ?? []) === JSON.stringify(["coding_team_v0"]) &&
    JSON.stringify(cohorts?.allowed_project_types ?? []) === JSON.stringify(["crm"]) &&
    JSON.stringify(cohorts?.allowed_input_classes ?? []) === JSON.stringify(["fe_led"]);
  readyChecks.static_policy_superset_ok =
    (policy?.allowed_workflow_types ?? []).includes("coding_team_v0") &&
    (policy?.allowed_project_types ?? []).includes("crm") &&
    (policy?.fe_safe_eligible_input_classes ?? []).includes("fe_led");
  readyChecks.accelerated_validation_available = Boolean(acceleratedReport?.compressed_go_no_go);
  readyChecks.live_runtime_validation_available = liveRuntimeReport?.overall === "pass";

  if (!readyChecks.rollout_master_enabled) issues.push("production_parallel_rollout.json: master_enabled must remain true for Phase A");
  if (!readyChecks.dynamic_routing_currently_disabled) issues.push("production_parallel_rollout.json: dynamic_routing_enabled should still be false before approval");
  if (!readyChecks.current_router_mode_static) issues.push("production_parallel_rollout.json: router_mode should still be static_policy_only before approval");
  if (!readyChecks.cohort_file_present) issues.push("m7_exposure_cohorts.json missing");
  if (readyChecks.cohort_file_present && !readyChecks.cohort_disabled_pending_approval) issues.push("m7_exposure_cohorts.json: cohort_enabled should remain false before approval");
  if (readyChecks.cohort_file_present && !readyChecks.cohort_matches_phase_a) issues.push("m7_exposure_cohorts.json does not match Phase A narrow cohort");
  if (!readyChecks.static_policy_superset_ok) issues.push("parallel_exposure_policy.json does not cover the required Phase A cohort");
  if (!readyChecks.accelerated_validation_available) issues.push("accelerated validation report missing");
  if (!readyChecks.live_runtime_validation_available) issues.push("live runtime validation report missing or not pass");

  const report = {
    generated_at: new Date().toISOString(),
    mode: "phase_a_advisory_preflight",
    overall: issues.length === 0 ? "ready_for_review" : "not_ready",
    files: {
      rollout_path: rolloutPath,
      cohort_path: cohortPath,
      policy_path: policyPath,
      accelerated_report_path: acceleratedReportPath,
      live_runtime_report_path: liveRuntimeReportPath,
    },
    current_state: {
      master_enabled: rollout?.master_enabled ?? null,
      dynamic_routing_enabled: rollout?.dynamic_routing_enabled ?? null,
      router_mode: rollout?.router_mode ?? null,
      cohort_enabled: cohorts?.runtime_controls?.cohort_enabled ?? null,
      cohort_definition: {
        allowed_workflow_types: cohorts?.allowed_workflow_types ?? [],
        allowed_project_types: cohorts?.allowed_project_types ?? [],
        allowed_input_classes: cohorts?.allowed_input_classes ?? [],
      },
    },
    ready_checks: readyChecks,
    issues,
    recommended_phase_a_config_delta: {
      production_parallel_rollout: {
        dynamic_routing_enabled: true,
        router_mode: "dynamic_routing_advisory",
      },
      m7_exposure_cohorts: {
        runtime_controls: {
          cohort_enabled: true,
          environment: "production",
        },
      },
    },
    evidence_summary: {
      accelerated_validation_samples: acceleratedReport?.routing_samples ?? null,
      accelerated_forced_sequential_ratio: acceleratedReport?.forced_sequential_ratio ?? null,
      live_runtime_overall: liveRuntimeReport?.overall ?? null,
      accelerated_report_exists: exists(acceleratedReportPath),
      live_runtime_report_exists: exists(liveRuntimeReportPath),
    },
  };

  const outDir = resolveOrchestratorArtifactPath("m7_phase_a");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "phase_a_advisory_preflight.json");
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");

  console.log(`# M7 Phase A Advisory Preflight`);
  console.log(`- report: ${outPath.replace(/\\/g, "/")}`);
  console.log(`- overall: ${report.overall}`);
  if (issues.length > 0) {
    console.log(`- issues: ${issues.length}`);
    for (const issue of issues) console.log(`  - ${issue}`);
  } else {
    console.log(`- recommendation: ready_for_review`);
  }
}

main();
