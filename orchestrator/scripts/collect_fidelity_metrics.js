/**
 * collect_fidelity_metrics.js
 *
 * P2-02 Observability: scan release artifacts and aggregate product fidelity,
 * preview validation, and GO/No-Go metrics for the observability dashboard.
 *
 * Usage:
 *   node orchestrator/scripts/collect_fidelity_metrics.js [--workspace /path]
 *   node orchestrator/scripts/collect_fidelity_metrics.js --since 2026-03-01
 */

import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";
import { ORCHESTRATOR_ROOT, resolveRepoPath } from "./_paths.js";

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { workspace: null, since: null };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--workspace" && args[i + 1]) opts.workspace = args[++i];
    if (args[i] === "--since" && args[i + 1]) opts.since = new Date(args[++i]);
  }
  return opts;
}

function scanReleaseRoots(releaseDir) {
  try {
    return fs.readdirSync(releaseDir)
      .map((name) => path.join(releaseDir, name))
      .filter((p) => fs.statSync(p).isDirectory());
  } catch {
    return [];
  }
}

function classifyFidelityFromGoNoGo(goNoGo) {
  if (!goNoGo) return "no_report";
  const pfWarn = goNoGo.product_fidelity_warning;
  if (!pfWarn) return "demo_usable_or_no_gate";
  return String(pfWarn.classification || "unknown");
}

export function collectFidelityMetrics({ workspaceRoot = ORCHESTRATOR_ROOT, since = null } = {}) {
  const releaseDir = path.join(workspaceRoot, "artifacts", "release");
  const roots = scanReleaseRoots(releaseDir);

  const totals = {
    runs_scanned: 0,
    go_count: 0,
    no_go_count: 0,
    no_verdict_count: 0,
    fidelity_demo_usable: 0,
    fidelity_shallow: 0,
    fidelity_preview_mismatch: 0,
    fidelity_domain_misaligned: 0,
    fidelity_visually_incomplete: 0,
    fidelity_no_report: 0,
    preview_matched: 0,
    preview_mismatch: 0,
    preview_missing: 0,
    preview_domain_pattern_mismatch: 0,
    preview_no_report: 0,
    perceptual_quality_low: 0,
    perceptual_quality_mid: 0,
    perceptual_quality_high: 0,
    perceptual_quality_no_report: 0,
    fidelity_warning_issued: 0,
    preview_warning_issued: 0,
    workflow_ids: {},
    project_types: {},
  };
  const samples = [];

  for (const root of roots) {
    const goNoGoPath = path.join(root, "qa", "go_no_go_result.json");
    const fidelityPath = path.join(root, "qa", "product_fidelity_report.json");
    const previewValPath = path.join(root, "qa", "preview_validation_report.json");

    const goNoGo = safeReadJson(goNoGoPath);
    if (!goNoGo) continue;

    const generatedAt = goNoGo.generated_at ? new Date(goNoGo.generated_at) : null;
    if (since && generatedAt && generatedAt < since) continue;

    totals.runs_scanned++;
    const verdict = String(goNoGo.verdict || "");
    if (verdict === "GO") totals.go_count++;
    else if (verdict === "NO_GO") totals.no_go_count++;
    else totals.no_verdict_count++;

    const wfId = String(goNoGo.workflow_id || "unknown");
    const ptId = String(goNoGo.project_type || "unknown");
    totals.workflow_ids[wfId] = (totals.workflow_ids[wfId] || 0) + 1;
    totals.project_types[ptId] = (totals.project_types[ptId] || 0) + 1;

    const fidelityReport = safeReadJson(fidelityPath);
    const previewReport = safeReadJson(previewValPath);

    // Fidelity classification
    if (fidelityReport) {
      const fc = String(fidelityReport.classification || "");
      if (fc === "demo_usable") totals.fidelity_demo_usable++;
      else if (fc === "artifact_complete_but_shallow") totals.fidelity_shallow++;
      else if (fc === "preview_mismatch") totals.fidelity_preview_mismatch++;
      else if (fc === "domain_misaligned") totals.fidelity_domain_misaligned++;
      else if (fc === "visually_incomplete") totals.fidelity_visually_incomplete++;
      else totals.fidelity_no_report++;

      if (fidelityReport.should_warn) totals.fidelity_warning_issued++;

      const pq = String(fidelityReport?.perceptual_quality?.score || "");
      if (pq === "low") totals.perceptual_quality_low++;
      else if (pq === "mid") totals.perceptual_quality_mid++;
      else if (pq === "high") totals.perceptual_quality_high++;
      else totals.perceptual_quality_no_report++;
    } else {
      // No product_fidelity_report.json — run predates fidelity gate
      // Count as "no_report" regardless of go_no_go verdict
      totals.fidelity_no_report++;
      totals.perceptual_quality_no_report++;
      // Still count fidelity warnings if they were embedded in the go_no_go
      if (goNoGo.product_fidelity_warning) totals.fidelity_warning_issued++;
    }

    // Preview classification
    if (previewReport) {
      const pc = String(previewReport.classification || "");
      if (pc === "preview_matched") totals.preview_matched++;
      else if (pc === "preview_mismatch") totals.preview_mismatch++;
      else if (pc === "preview_missing") totals.preview_missing++;
      else if (pc === "preview_domain_pattern_mismatch") totals.preview_domain_pattern_mismatch++;
      else totals.preview_no_report++;
      if (previewReport.should_warn) totals.preview_warning_issued++;
    } else {
      const pvWarn = goNoGo.preview_validation_warning;
      if (pvWarn) {
        const pc = String(pvWarn.classification || "");
        if (pc === "preview_mismatch") totals.preview_mismatch++;
        else totals.preview_no_report++;
        totals.preview_warning_issued++;
      } else {
        totals.preview_no_report++;
      }
    }

    samples.push({
      run_id: String(goNoGo.run_id || path.basename(root)),
      workflow_id: wfId,
      project_type: ptId,
      verdict,
      fidelity_classification: fidelityReport
        ? String(fidelityReport.classification || "")
        : classifyFidelityFromGoNoGo(goNoGo),
      preview_classification: previewReport
        ? String(previewReport.classification || "")
        : (goNoGo.preview_validation_warning ? String(goNoGo.preview_validation_warning.classification || "") : "no_report"),
      perceptual_quality_score: fidelityReport ? String(fidelityReport?.perceptual_quality?.score || "") : "",
      generated_at: String(goNoGo.generated_at || ""),
    });
  }

  const goRate = totals.runs_scanned > 0 ? totals.go_count / totals.runs_scanned : 0;
  const runsWithFidelityReport = totals.runs_scanned - totals.fidelity_no_report;
  const fidelityPassRate = runsWithFidelityReport > 0
    ? totals.fidelity_demo_usable / runsWithFidelityReport
    : null; // null = insufficient data
  const goWithFidelityReport = runsWithFidelityReport > 0
    ? Math.min(totals.go_count, runsWithFidelityReport)
    : 0;
  const falsePositiveRate = goWithFidelityReport > 0
    ? (goWithFidelityReport - totals.fidelity_demo_usable) / goWithFidelityReport
    : null; // null = insufficient data (no fidelity reports yet)

  return {
    generated_at: new Date().toISOString(),
    workspace_root: workspaceRoot,
    since: since ? since.toISOString() : null,
    totals: { ...totals, runs_with_fidelity_report: runsWithFidelityReport },
    rates: {
      go_rate: Number(goRate.toFixed(4)),
      fidelity_pass_rate: fidelityPassRate !== null ? Number(fidelityPassRate.toFixed(4)) : null,
      preview_mismatch_rate: totals.runs_scanned > 0
        ? Number(((totals.preview_mismatch + totals.preview_domain_pattern_mismatch) / totals.runs_scanned).toFixed(4))
        : 0,
      false_positive_estimate: falsePositiveRate !== null ? Number(falsePositiveRate.toFixed(4)) : null,
      note: runsWithFidelityReport === 0 ? "no fidelity reports found — all runs predate fidelity gate" : null,
      perceptual_quality_distribution: totals.runs_scanned > 0
        ? {
          low: Number((totals.perceptual_quality_low / totals.runs_scanned).toFixed(4)),
          mid: Number((totals.perceptual_quality_mid / totals.runs_scanned).toFixed(4)),
          high: Number((totals.perceptual_quality_high / totals.runs_scanned).toFixed(4)),
          no_report: Number((totals.perceptual_quality_no_report / totals.runs_scanned).toFixed(4)),
        }
        : { low: 0, mid: 0, high: 0, no_report: 1 },
    },
    samples: samples.slice(-20),
  };
}

export function main() {
  const opts = parseArgs();
  const workspaceRoot = opts.workspace
    ? path.resolve(opts.workspace)
    : path.resolve(ORCHESTRATOR_ROOT, "..");
  const result = collectFidelityMetrics({ workspaceRoot, since: opts.since });

  console.log("=== Fidelity Metrics Report ===");
  console.log(`generated_at: ${result.generated_at}`);
  console.log(`runs_scanned: ${result.totals.runs_scanned} (with_fidelity_report: ${result.totals.runs_with_fidelity_report})`);
  console.log(`go_rate: ${(result.rates.go_rate * 100).toFixed(1)}%`);
  console.log(`fidelity_pass_rate: ${result.rates.fidelity_pass_rate !== null ? (result.rates.fidelity_pass_rate * 100).toFixed(1) + "%" : "n/a (no fidelity reports)"}`);
  console.log(`false_positive_estimate: ${result.rates.false_positive_estimate !== null ? (result.rates.false_positive_estimate * 100).toFixed(1) + "%" : "n/a (no fidelity reports)"}`);
  console.log(`preview_mismatch_rate: ${(result.rates.preview_mismatch_rate * 100).toFixed(1)}%`);
  if (result.rates.note) console.log(`note: ${result.rates.note}`);
  console.log(`fidelity_warning_issued: ${result.totals.fidelity_warning_issued}`);
  console.log(`preview_warning_issued: ${result.totals.preview_warning_issued}`);
  console.log(`perceptual_quality: low=${result.totals.perceptual_quality_low} mid=${result.totals.perceptual_quality_mid} high=${result.totals.perceptual_quality_high} no_report=${result.totals.perceptual_quality_no_report}`);
  console.log("");
  console.log(JSON.stringify(result, null, 2));
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
