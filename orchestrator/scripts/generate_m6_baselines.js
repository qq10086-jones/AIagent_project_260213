#!/usr/bin/env node
/**
 * WS-26-01 / WS-26-02 / WS-26-03
 * Generate M6 baseline metric files from staging replay artifacts.
 *
 * Outputs:
 *   metrics/baseline_context_budget.json
 *   metrics/diff_first_baseline.json
 *   metrics/patch_reliability.json
 *   metrics/parallel_eligibility.json
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const STAGING_DIR = path.join(ROOT, "artifacts/m6_staging_replay");
const METRICS_DIR = path.join(ROOT, "..", "metrics");

function ensureDir(p) { fs.mkdirSync(p, { recursive: true }); }
function writeJson(p, v) { ensureDir(path.dirname(p)); fs.writeFileSync(p, JSON.stringify(v, null, 2), "utf8"); }

// ── Load all sequential replay results ───────────────────────────────────────
// Use the sequential-mode results as the canonical baseline source.

function loadAllResults() {
  if (!fs.existsSync(STAGING_DIR)) return [];
  const runs = fs.readdirSync(STAGING_DIR).filter((d) => d.startsWith("replay-")).sort();
  const results = [];
  for (const run of runs) {
    const runDir = path.join(STAGING_DIR, run);
    for (const file of fs.readdirSync(runDir)) {
      if (!file.endsWith(".json") || file.startsWith("_")) continue;
      try {
        const data = JSON.parse(fs.readFileSync(path.join(runDir, file), "utf8"));
        // Only use sequential-mode results for baseline derivation
        if (data.execution_mode_requested === "sequential" || data.execution_mode_requested === undefined) {
          results.push(data);
        }
      } catch { /* skip malformed */ }
    }
  }
  return results;
}

// ── Statistics helpers ────────────────────────────────────────────────────────

function percentile(sorted, p) {
  if (sorted.length === 0) return null;
  const idx = Math.max(0, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[idx];
}

function stats(values) {
  if (values.length === 0) return { p50: null, p90: null, max: null, count: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  return {
    p50: percentile(sorted, 50),
    p90: percentile(sorted, 90),
    max: sorted[sorted.length - 1],
    count: values.length,
  };
}

// ── WS-26-01: Context Budget Baseline ────────────────────────────────────────

function generateContextBudgetBaseline(results) {
  const byRole = { pm: [], arch: [], be: [], fe: [], qa: [] };

  for (const r of results) {
    if (!r.context_budget) continue;
    if (r.context_budget.pm_pct != null) byRole.pm.push(r.context_budget.pm_pct);
    if (r.context_budget.arch_pct != null) byRole.arch.push(r.context_budget.arch_pct);
    if (r.context_budget.be_pct != null) byRole.be.push(r.context_budget.be_pct);
    if (r.context_budget.fe_pct != null) byRole.fe.push(r.context_budget.fe_pct);
    if (r.context_budget.qa_pct != null) byRole.qa.push(r.context_budget.qa_pct);
  }

  const distributions = {};
  const overflowRiskTails = [];
  for (const [role, values] of Object.entries(byRole)) {
    const s = stats(values);
    distributions[role] = s;
    if (s.p90 != null && s.p90 >= 85) {
      overflowRiskTails.push({ role, p90: s.p90, max: s.max, risk: "HIGH" });
    } else if (s.p90 != null && s.p90 >= 70) {
      overflowRiskTails.push({ role, p90: s.p90, max: s.max, risk: "MEDIUM" });
    }
  }

  return {
    generated_at: new Date().toISOString(),
    source: "m6_staging_replay (sequential mode)",
    total_samples: results.length,
    note: "Values derived from replay simulation. Re-derive from live LLM execution before GO_LIMITED_EXPOSURE.",
    distributions,
    overflow_risk_tails: overflowRiskTails.length > 0 ? overflowRiskTails : [{ note: "No high-risk tails detected in simulation baseline" }],
  };
}

// ── WS-26-02: Diff-first + Patch Reliability Baselines ───────────────────────

function generateDiffFirstBaseline(results) {
  const cleanRepo = results.filter((r) => !r.dirty_repo_case);
  const dirtyRepo = results.filter((r) => r.dirty_repo_case);

  function diffStats(subset) {
    const total = subset.length;
    const hits = subset.reduce((s, r) => s + (r.diff_first_hit || 0), 0);
    const fallbacks = subset.reduce((s, r) => s + (r.diff_first_fallback || 0), 0);
    const implSteps = total * 2; // impl_be + impl_fe
    const hitRate = implSteps > 0 ? hits / implSteps : null;
    const fallbackRate = implSteps > 0 ? fallbacks / implSteps : null;
    return { total_cases: total, impl_steps: implSteps, diff_first_hits: hits, diff_first_fallbacks: fallbacks, hit_rate: hitRate, fallback_rate: fallbackRate };
  }

  const cleanStats = diffStats(cleanRepo);
  const dirtyStats = diffStats(dirtyRepo);
  const overall = diffStats(results);

  return {
    generated_at: new Date().toISOString(),
    source: "m6_staging_replay (sequential mode)",
    note: "Values derived from replay simulation. Re-derive from live LLM execution before GO_LIMITED_EXPOSURE.",
    overall,
    clean_repo: cleanStats,
    dirty_repo: dirtyStats,
    go_nogo_threshold: {
      diff_first_hit_rate_required: 0.60,
      clean_repo_actual: cleanStats.hit_rate,
      assessment: cleanStats.hit_rate != null && cleanStats.hit_rate >= 0.60 ? "PASS" : "FAIL",
    },
  };
}

function generatePatchReliabilityBaseline(results) {
  const cleanRepo = results.filter((r) => !r.dirty_repo_case);
  const dirtyRepo = results.filter((r) => r.dirty_repo_case);

  function patchStats(subset) {
    const total = subset.length;
    const mismatches = subset.reduce((s, r) => s + (r.patch_mismatch_count || 0), 0);
    const implSteps = total * 2;
    const mismatchRate = implSteps > 0 ? mismatches / implSteps : null;
    return { total_cases: total, impl_steps: implSteps, patch_mismatches: mismatches, mismatch_rate: mismatchRate };
  }

  const cleanStats = patchStats(cleanRepo);
  const dirtyStats = patchStats(dirtyRepo);
  const overall = patchStats(results);

  return {
    generated_at: new Date().toISOString(),
    source: "m6_staging_replay (sequential mode)",
    note: "Values derived from replay simulation. Re-derive from live LLM execution before GO_LIMITED_EXPOSURE.",
    overall,
    clean_repo: cleanStats,
    dirty_repo: dirtyStats,
    go_nogo_threshold: {
      mismatch_rate_required_lte: 0.15,
      overall_actual: overall.mismatch_rate,
      assessment: overall.mismatch_rate != null && overall.mismatch_rate <= 0.15 ? "PASS" : "FAIL",
    },
  };
}

// ── WS-26-03: Parallel Eligibility Baseline ──────────────────────────────────

function generateParallelEligibilityBaseline(results) {
  const byInputClass = {};
  const denialReasons = {};

  for (const r of results) {
    const cls = r.input_class || "unknown";
    if (!byInputClass[cls]) byInputClass[cls] = { total: 0, eligible: 0, ineligible: 0 };
    byInputClass[cls].total++;

    const decision = r.gate_decision?.effective_exposure_decision;
    if (decision === "gated_parallel_allowed") {
      byInputClass[cls].eligible++;
    } else {
      byInputClass[cls].ineligible++;
      const reason = r.gate_decision?.deny_reason_code || r.gate_decision?.effective_exposure_decision_source || "unknown";
      denialReasons[reason] = (denialReasons[reason] || 0) + 1;
    }
  }

  const eligibilityRates = {};
  for (const [cls, counts] of Object.entries(byInputClass)) {
    eligibilityRates[cls] = {
      ...counts,
      eligibility_rate: counts.total > 0 ? counts.eligible / counts.total : null,
    };
  }

  const qualifyingClasses = Object.entries(eligibilityRates)
    .filter(([, v]) => v.eligibility_rate != null && v.eligibility_rate >= 0.80)
    .map(([cls, v]) => ({ input_class: cls, eligibility_rate: v.eligibility_rate }));

  return {
    generated_at: new Date().toISOString(),
    source: "m6_staging_replay (sequential mode — gate evaluated against rollout_master_disabled)",
    note: "All cases forced sequential because master_enabled=false. Eligibility rates reflect gate policy logic only. Re-run with master_enabled=true for live eligibility distribution.",
    by_input_class: eligibilityRates,
    denial_reasons_categorized: denialReasons,
    qualifying_input_classes_at_80pct: qualifyingClasses,
    go_nogo_threshold: {
      required: "at least 1 workflow type at >= 80% eligibility rate",
      note: "fe_led is declared FE-safe in policy; with master enabled, all fe_led cases would qualify. Assessment deferred to live run.",
      assessment: "DEFERRED",
    },
  };
}

// ── Main ──────────────────────────────────────────────────────────────────────

const results = loadAllResults();
console.log(`Loaded ${results.length} sequential replay results`);

const budgetBaseline = generateContextBudgetBaseline(results);
const diffFirstBaseline = generateDiffFirstBaseline(results);
const patchBaseline = generatePatchReliabilityBaseline(results);
const eligibilityBaseline = generateParallelEligibilityBaseline(results);

ensureDir(METRICS_DIR);
writeJson(path.join(METRICS_DIR, "baseline_context_budget.json"), budgetBaseline);
writeJson(path.join(METRICS_DIR, "diff_first_baseline.json"), diffFirstBaseline);
writeJson(path.join(METRICS_DIR, "patch_reliability.json"), patchBaseline);
writeJson(path.join(METRICS_DIR, "parallel_eligibility.json"), eligibilityBaseline);

console.log("Written:");
console.log("  metrics/baseline_context_budget.json");
console.log("  metrics/diff_first_baseline.json");
console.log("  metrics/patch_reliability.json");
console.log("  metrics/parallel_eligibility.json");

// Summary
console.log("\nContext budget p90 by role:");
for (const [role, s] of Object.entries(budgetBaseline.distributions)) {
  console.log(`  ${role}: p50=${s.p50}% p90=${s.p90}% max=${s.max}%`);
}
console.log(`\nDiff-first hit rate (clean-repo): ${(diffFirstBaseline.clean_repo.hit_rate * 100).toFixed(1)}%  ${diffFirstBaseline.go_nogo_threshold.assessment}`);
console.log(`Patch mismatch rate (overall):    ${(patchBaseline.overall.mismatch_rate * 100).toFixed(1)}%  ${patchBaseline.go_nogo_threshold.assessment}`);
console.log(`Parallel eligibility assessment:  ${eligibilityBaseline.go_nogo_threshold.assessment}`);
