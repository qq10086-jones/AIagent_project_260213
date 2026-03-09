#!/usr/bin/env node

import fs from "fs";
import path from "path";
import {
  resolveOrchestratorArtifactPath,
  resolveOrchestratorPath,
} from "./_paths.js";

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (!cur.startsWith("--")) continue;
    const key = cur.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const apply = args.apply === true;
  const approvedBy = String(args["approved-by"] || "PM+Architect");
  const approvalDate = String(args["approval-date"] || new Date().toISOString().slice(0, 10));

  const rolloutPath = resolveOrchestratorPath("configs", "production_parallel_rollout.json");
  const cohortPath = resolveOrchestratorPath("configs", "m7_exposure_cohorts.json");

  const rollout = loadJson(rolloutPath);
  const cohort = loadJson(cohortPath);

  const updatedRollout = {
    ...rollout,
    dynamic_routing_enabled: true,
    router_mode: "dynamic_routing_advisory",
    last_policy_change: new Date().toISOString(),
  };

  const updatedCohort = {
    ...cohort,
    runtime_controls: {
      ...(cohort.runtime_controls || {}),
      cohort_enabled: true,
      approved_by: approvedBy,
      approval_date: approvalDate,
      environment: "production",
      live_traffic: true,
    },
  };

  const outDir = resolveOrchestratorArtifactPath("m7_phase_a");
  fs.mkdirSync(outDir, { recursive: true });
  const reportPath = path.join(outDir, "phase_a_enablement_plan.json");

  const report = {
    generated_at: new Date().toISOString(),
    mode: apply ? "apply" : "dry_run",
    target: "m7_phase_a_advisory",
    files: {
      rollout_path: rolloutPath,
      cohort_path: cohortPath,
    },
    before: {
      rollout,
      cohort,
    },
    after: {
      rollout: updatedRollout,
      cohort: updatedCohort,
    },
  };

  if (apply) {
    writeJson(rolloutPath, updatedRollout);
    writeJson(cohortPath, updatedCohort);
  }

  writeJson(reportPath, report);

  console.log(`# Set M7 Phase A Advisory`);
  console.log(`- mode: ${apply ? "apply" : "dry_run"}`);
  console.log(`- report: ${reportPath.replace(/\\/g, "/")}`);
  console.log(`- router_mode -> ${updatedRollout.router_mode}`);
  console.log(`- dynamic_routing_enabled -> ${updatedRollout.dynamic_routing_enabled}`);
  console.log(`- cohort_enabled -> ${updatedCohort.runtime_controls?.cohort_enabled}`);
}

main();
