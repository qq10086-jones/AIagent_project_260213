#!/usr/bin/env node
/**
 * run_m7_dynamic_routing_trial.js
 *
 * WS-31-02: Limited Dynamic Routing Exposure Trial Runner.
 *
 * What this script does:
 *  1. Validates all prerequisites (cohort_enabled, master_enabled, dynamic_routing_enabled)
 *  2. Evaluates each replay corpus case through the three-layer routing gate
 *     (read-only — does NOT trigger real LLM calls; uses replay manifest as evidence base)
 *  3. Tracks classifier health via createClassifierHealthMonitor
 *  4. Evaluates circuit-breaker thresholds after each batch
 *  5. Optionally exercises the classifier unavailability drill (--drill-unavailable)
 *  6. Produces a structured trial result bundle (JSON artifact)
 *
 * Usage:
 *   node scripts/run_m7_dynamic_routing_trial.js \
 *     [--manifest path/to/manifest.json] \
 *     [--drill-unavailable]              \
 *     [--out path/to/trial_result.json]
 *
 * Prerequisites to run a live trial (not enforced here — checked at runtime):
 *   - configs/m7_exposure_cohorts.json:           cohort_enabled: true
 *   - configs/production_parallel_rollout.json:   master_enabled: true, dynamic_routing_enabled: true
 *   Both require explicit Architect sign-off before being set to true.
 *
 * In dry-run mode (prerequisites not met), the script evaluates replay cases
 * against the current static-policy gate and emits a preflight report.
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname }            from "path";
import { fileURLToPath }               from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = resolve(__dirname, "..");

function loadJson(filePath) {
  try { return JSON.parse(readFileSync(filePath, "utf8")); }
  catch { return null; }
}

// ── Prerequisite check ────────────────────────────────────────────────────────

function checkPrerequisites() {
  const rollout = loadJson(resolve(WORKSPACE_ROOT, "configs/production_parallel_rollout.json"));
  const cohorts = loadJson(resolve(WORKSPACE_ROOT, "configs/m7_exposure_cohorts.json"));
  const issues  = [];

  if (!rollout)                              issues.push("production_parallel_rollout.json not found");
  if (rollout && !rollout.master_enabled)    issues.push("production_parallel_rollout.json: master_enabled=false (requires Architect sign-off)");
  if (rollout && !rollout.dynamic_routing_enabled) issues.push("production_parallel_rollout.json: dynamic_routing_enabled=false");
  if (!cohorts)                              issues.push("configs/m7_exposure_cohorts.json not found");
  if (cohorts && !cohorts.runtime_controls?.cohort_enabled) issues.push("m7_exposure_cohorts.json: cohort_enabled=false (requires Architect sign-off)");

  return {
    ready:  issues.length === 0,
    issues,
    rollout,
    cohorts,
  };
}

// ── Gate evaluation (mirrors parallel_rollout_gate.js Layer-2 static path) ───

function evaluateStaticGate(replayCase, { cohorts }) {
  const allowed = cohorts?.included_workflow_classes?.includes(replayCase.workflow_type) &&
                  cohorts?.allowed_project_types?.includes(replayCase.project_type) &&
                  cohorts?.allowed_input_classes?.includes(replayCase.input_class);
  return allowed ? "gated_parallel_allowed" : "forced_sequential";
}

function evaluateDynamicGate(replayCase, { healthMonitor }) {
  if (healthMonitor.isUnavailable()) {
    return { decision: "forced_sequential", source: "classifier_unavailable_fallback" };
  }
  // Use fe_parallel_eligible_expected as classifier ground truth
  const decision = replayCase.fe_parallel_eligible_expected
    ? "gated_parallel_allowed"
    : "forced_sequential";
  return { decision, source: "classifier_recommended" };
}

// ── Circuit-breaker threshold evaluation ──────────────────────────────────────

function evaluateThresholds(results, cohorts) {
  const thresholds = cohorts?.rollback_trigger_thresholds ?? {};
  const total = results.length;
  if (total === 0) return { breached: false };

  const forcedSeq = results.filter((r) => r.dynamic_decision === "forced_sequential").length;
  const forcedSeqPct = (forcedSeq / total) * 100;
  const limit = thresholds.forced_sequential_spike_pct ?? 40;

  if (forcedSeqPct > limit) {
    return {
      breached: true,
      metric:   "forced_sequential_spike_pct",
      observed: forcedSeqPct,
      limit,
    };
  }
  return { breached: false };
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const drillUnavailable = Boolean(args["drill-unavailable"]);

  const { createClassifierHealthMonitor } = await import(
    "../src/domain/classifier_health_monitor.js"
  );

  const alerts = [];
  const healthMonitor = createClassifierHealthMonitor({
    windowSize:     50,
    alertThreshold: 0.90,
    alertFn:        (msg) => { alerts.push(msg); console.error(msg); },
    drillMode:      drillUnavailable,
  });

  // ── 1. Prerequisites ────────────────────────────────────────────────────────
  const prereq = checkPrerequisites();
  const mode   = prereq.ready ? "live_trial" : "dry_run_preflight";

  console.error(`[m7-trial] mode: ${mode}`);
  if (!prereq.ready) {
    console.error("[m7-trial] prerequisite issues:");
    for (const issue of prereq.issues) console.error(`  ✗ ${issue}`);
    console.error("[m7-trial] running preflight evaluation against static gate.");
  }

  // ── 2. Load replay manifest ─────────────────────────────────────────────────
  const manifestPath = args.manifest
    ? resolve(process.cwd(), args.manifest)
    : resolve(WORKSPACE_ROOT, "replay/manifests/m6_staging_replay_manifest.json");

  const manifest = loadJson(manifestPath);
  if (!manifest || !Array.isArray(manifest.corpus)) {
    console.error("[m7-trial] fatal: could not load replay manifest at", manifestPath);
    process.exit(1);
  }
  console.error(`[m7-trial] loaded manifest: ${manifest.manifest_id} (${manifest.corpus.length} cases)`);

  // ── 3. Evaluate cases ───────────────────────────────────────────────────────
  const results = [];
  for (const c of manifest.corpus) {
    // Simulate one classifier call per case
    const classifierSuccess = !drillUnavailable;
    healthMonitor.record(classifierSuccess);

    const staticDec = evaluateStaticGate(c, prereq);
    const { decision: dynDec, source: dynSource } = evaluateDynamicGate(c, { healthMonitor });

    results.push({
      replay_id:       c.replay_id,
      input_class:     c.input_class,
      project_type:    c.project_type,
      workflow_type:   c.workflow_type,
      expected_parallel: Boolean(c.fe_parallel_eligible_expected),
      static_decision:   staticDec,
      dynamic_decision:  dynDec,
      dynamic_source:    dynSource,
      agreement:         staticDec === dynDec,
    });
  }

  // ── 4. Circuit-breaker threshold evaluation ─────────────────────────────────
  const thresholdEval = evaluateThresholds(results, prereq.cohorts);
  const healthSnapshot = healthMonitor.snapshot();

  // ── 5. Assemble trial result bundle ────────────────────────────────────────
  const trialResult = {
    trial_id:         `m7-trial-${new Date().toISOString().replace(/[:.]/g, "-")}`,
    generated_at:     new Date().toISOString(),
    mode,
    manifest_id:      manifest.manifest_id,
    drill_unavailable: drillUnavailable,

    prerequisites: {
      ready:  prereq.ready,
      issues: prereq.issues,
    },

    summary: {
      total_cases:            results.length,
      static_gated_parallel:  results.filter((r) => r.static_decision  === "gated_parallel_allowed").length,
      dynamic_gated_parallel: results.filter((r) => r.dynamic_decision === "gated_parallel_allowed").length,
      agreement_count:        results.filter((r) => r.agreement).length,
      agreement_rate:         results.length > 0 ? results.filter((r) => r.agreement).length / results.length : null,
      forced_sequential_count: results.filter((r) => r.dynamic_decision === "forced_sequential").length,
    },

    classifier_health: healthSnapshot,
    circuit_breaker_evaluation: thresholdEval,
    alerts,
    cases: results,
  };

  const output = JSON.stringify(trialResult, null, 2);

  if (args.out) {
    const outPath = resolve(process.cwd(), args.out);
    writeFileSync(outPath, output, "utf8");
    console.error(`[m7-trial] result written to ${outPath}`);
  }

  process.stdout.write(output + "\n");

  // Non-zero exit if threshold breached — allows CI to detect it
  if (thresholdEval.breached) {
    console.error(`[m7-trial] THRESHOLD BREACHED: ${thresholdEval.metric} = ${thresholdEval.observed.toFixed(1)}% (limit: ${thresholdEval.limit}%)`);
    process.exit(2);
  }
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      const next = argv[i + 1];
      if (!next || next.startsWith("--")) {
        out[key] = true;
      } else {
        out[key] = next;
        i++;
      }
    }
  }
  return out;
}

main().catch((err) => {
  console.error("[m7-trial] fatal:", err.message);
  process.exit(1);
});
