/**
 * compare_fidelity_baseline.js
 *
 * P2-03 Real-Run Comparison Pack: compare current fidelity metrics
 * against the documented baseline to measure improvement delta.
 *
 * Usage:
 *   node orchestrator/scripts/compare_fidelity_baseline.js
 *   node orchestrator/scripts/compare_fidelity_baseline.js --since 2026-03-27
 *   node orchestrator/scripts/compare_fidelity_baseline.js --baseline /path/to/baseline.json
 */

import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import { ORCHESTRATOR_ROOT, resolveRepoPath } from "./_paths.js";
import { collectFidelityMetrics } from "./collect_fidelity_metrics.js";

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { baseline: null, since: null, workspace: null, outputPath: null };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--baseline" && args[i + 1]) opts.baseline = args[++i];
    if (args[i] === "--since" && args[i + 1]) opts.since = new Date(args[++i]);
    if (args[i] === "--workspace" && args[i + 1]) opts.workspace = args[++i];
    if (args[i] === "--output" && args[i + 1]) opts.outputPath = args[++i];
  }
  return opts;
}

function delta(current, baseline, key) {
  const cur = Number(current?.[key] ?? 0);
  const base = Number(baseline?.[key] ?? 0);
  return {
    baseline: base,
    current: cur,
    delta: Number((cur - base).toFixed(4)),
    direction: cur < base ? "improved" : cur > base ? "degraded" : "unchanged",
  };
}

export function compareFidelityBaseline({
  baselinePath = null,
  workspaceRoot = null,
  since = null,
} = {}) {
  const wsRoot = workspaceRoot
    ? path.resolve(workspaceRoot)
    : path.resolve(ORCHESTRATOR_ROOT, "..");

  const defaultBaselinePath = path.join(wsRoot, "artifacts", "baselines", "canary_baseline_report.json");
  const resolvedBaselinePath = baselinePath ? path.resolve(baselinePath) : defaultBaselinePath;
  const baseline = safeReadJson(resolvedBaselinePath);

  if (!baseline) {
    throw new Error(`Baseline not found at ${resolvedBaselinePath}. Run P0-06 first.`);
  }

  const current = collectFidelityMetrics({ workspaceRoot: wsRoot, since });

  // Derive baseline rates from baseline report structure
  const baselineRates = {
    false_positive_rate: Number(baseline?.totals?.false_positive_rate ?? 1.0),
    go_rate: baseline?.totals?.go_count != null && baseline?.totals?.samples != null
      ? Number(baseline.totals.go_count / baseline.totals.samples)
      : 1.0,
    preview_issue_rate: Number(baseline?.totals?.preview_issue_rate ?? 0),
    fidelity_pass_rate: 0, // baseline assumed 0 demo_usable
  };

  const runsWithReport = current.totals.runs_with_fidelity_report ?? 0;
  const hasComparableData = runsWithReport > 0;

  // Only compute deltas when there are fidelity reports to compare against
  const fpCurrentRate = hasComparableData ? current.rates.false_positive_estimate : null;
  const fidelityCurrentRate = hasComparableData ? current.rates.fidelity_pass_rate : null;

  const fpDeltaObj = hasComparableData
    ? delta({ value: fpCurrentRate }, { value: baselineRates.false_positive_rate }, "value")
    : { baseline: baselineRates.false_positive_rate, current: null, delta: null, direction: "no_data" };

  const fidelityDeltaObj = hasComparableData
    ? delta({ value: fidelityCurrentRate }, { value: baselineRates.fidelity_pass_rate }, "value")
    : { baseline: baselineRates.fidelity_pass_rate, current: null, delta: null, direction: "no_data" };

  const previewDeltaObj = delta(
    { value: current.rates.preview_mismatch_rate },
    { value: baselineRates.preview_issue_rate },
    "value"
  );

  let summary;
  if (!hasComparableData) {
    summary = `No fidelity-gated runs yet (0 of ${current.totals.runs_scanned} runs have product_fidelity_report.json). Baseline false-positive rate was ${(baselineRates.false_positive_rate * 100).toFixed(0)}%. Run fresh Discord canary scenarios with fidelity gate active to generate comparable data.`;
  } else if (fpDeltaObj.direction === "improved") {
    summary = `False-positive rate improved by ${(Math.abs(fpDeltaObj.delta) * 100).toFixed(1)}pp (${(baselineRates.false_positive_rate * 100).toFixed(0)}% → ${(fpCurrentRate * 100).toFixed(1)}%).`;
  } else if (fpDeltaObj.direction === "degraded") {
    summary = `False-positive rate increased by ${(fpDeltaObj.delta * 100).toFixed(1)}pp vs baseline. Investigation recommended.`;
  } else {
    summary = `False-positive rate unchanged from baseline (${(baselineRates.false_positive_rate * 100).toFixed(0)}%).`;
  }

  const comparison = {
    generated_at: new Date().toISOString(),
    baseline_path: resolvedBaselinePath,
    baseline_generated_at: String(baseline.generated_at || ""),
    baseline_type: String(baseline.baseline_type || ""),
    current_window: since ? since.toISOString() : "all_time",
    current_runs_scanned: current.totals.runs_scanned,
    current_runs_with_fidelity_report: runsWithReport,
    baseline_samples: Number(baseline?.totals?.samples ?? 0),
    has_comparable_data: hasComparableData,

    deltas: {
      false_positive_rate: fpDeltaObj,
      fidelity_pass_rate: fidelityDeltaObj,
      preview_mismatch_rate: previewDeltaObj,
    },

    current_breakdown: {
      go_rate: current.rates.go_rate,
      fidelity_pass_rate: fidelityCurrentRate,
      false_positive_estimate: fpCurrentRate,
      preview_mismatch_rate: current.rates.preview_mismatch_rate,
      fidelity_warning_issued: current.totals.fidelity_warning_issued,
      preview_warning_issued: current.totals.preview_warning_issued,
      perceptual_quality_distribution: current.rates.perceptual_quality_distribution,
    },

    summary,
    caveat: baseline.baseline_type === "historical_release_artifact_backfill"
      ? "Baseline is a backfill measurement. Replace with fresh live Discord canary runs for reliable comparison."
      : "",
  };

  return comparison;
}

export function main() {
  const opts = parseArgs();
  let result;
  try {
    result = compareFidelityBaseline({
      baselinePath: opts.baseline,
      workspaceRoot: opts.workspace ? path.resolve(opts.workspace) : null,
      since: opts.since,
    });
  } catch (err) {
    console.error(`[compare_fidelity_baseline] ${err.message}`);
    process.exit(1);
  }

  console.log("=== Fidelity Baseline Comparison ===");
  console.log(`generated_at: ${result.generated_at}`);
  console.log(`baseline: ${result.baseline_path} (${result.baseline_type})`);
  console.log(`current_runs_scanned: ${result.current_runs_scanned}`);
  console.log("");
  console.log("Summary:", result.summary);
  if (result.caveat) console.log("Caveat:", result.caveat);
  console.log("");
  console.log(`has_comparable_data: ${result.has_comparable_data}`);
  console.log(`current_runs_with_fidelity_report: ${result.current_runs_with_fidelity_report}`);
  console.log("");
  console.log("Deltas:");
  for (const [key, val] of Object.entries(result.deltas)) {
    if (val.direction === "no_data") {
      console.log(`  ${key}: baseline=${(val.baseline * 100).toFixed(1)}% current=n/a [no_data]`);
    } else {
      const sign = val.delta > 0 ? "+" : "";
      console.log(`  ${key}: baseline=${(val.baseline * 100).toFixed(1)}% current=${(val.current * 100).toFixed(1)}% delta=${sign}${(val.delta * 100).toFixed(1)}pp [${val.direction}]`);
    }
  }

  if (opts.outputPath) {
    fs.mkdirSync(path.dirname(opts.outputPath), { recursive: true });
    fs.writeFileSync(opts.outputPath, JSON.stringify(result, null, 2), "utf8");
    console.log(`\nReport written to: ${opts.outputPath}`);
  } else {
    console.log("\n" + JSON.stringify(result, null, 2));
  }

  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
