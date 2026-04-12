#!/usr/bin/env node
/**
 * run_static_audit.mjs — entry point for the static_audit workflow step.
 *
 * Usage:
 *   node run_static_audit.mjs --artifact-root <relpath> [--port 13101] [--mode dry_run|blocking]
 *
 * Behaviour:
 *   - Loads all scanners under scanners/
 *   - Runs them sequentially, aggregating findings
 *   - Writes verify/static_audit.json
 *   - Exit code 0 in dry_run mode always; in blocking mode, 1 if any scanner reports fail
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const args = parseArgs(process.argv.slice(2));
const workspaceRoot = process.env.WORKSPACE_ROOT || path.resolve(__dirname, "../../..");
const artifactRoot = args["artifact-root"] || "";
const port = Number(args["port"] || 13101);
const mode = args["mode"] || process.env.STATIC_AUDIT_MODE || "dry_run";

if (!artifactRoot) {
  console.error("[static_audit] ERROR: --artifact-root is required");
  process.exit(2);
}

const auditStarted = Date.now();
console.log(`[static_audit] mode=${mode} workspace=${workspaceRoot} artifact_root=${artifactRoot}`);

const scannerFiles = [
  "scanners/xss_scanner.mjs",
  "scanners/class_injection.mjs",
  "scanners/delete_semantics.mjs",
  "scanners/be_contract_checker.mjs",
];

const scannerResults = {};
let portOffset = 0;
for (const rel of scannerFiles) {
  const scannerPath = path.resolve(__dirname, rel);
  if (!fs.existsSync(scannerPath)) {
    console.warn(`[static_audit] scanner missing, skipping: ${rel}`);
    continue;
  }
  try {
    const mod = await import(`file://${scannerPath.replace(/\\/g, "/")}`);
    // Scanners that need an HTTP server get a unique port so they can run sequentially
    // without port-in-use races.
    const result = await mod.run({ workspaceRoot, artifactRoot, port: port + portOffset });
    portOffset++;
    scannerResults[result.scanner_id || rel] = result;
    console.log(`[static_audit] ${result.scanner_id}: ${result.status} (${result.findings.length} finding(s), ${result.duration_ms}ms)`);
  } catch (err) {
    console.error(`[static_audit] scanner ${rel} crashed: ${err?.message || err}`);
    scannerResults[rel] = {
      scanner_id: rel,
      status: "error",
      findings: [],
      error: String(err?.message || err),
      duration_ms: 0,
    };
  }
}

// Aggregate
const allFindings = Object.values(scannerResults).flatMap((r) => r.findings || []);
const totals = {
  critical: allFindings.filter((f) => f.severity === "critical").length,
  high: allFindings.filter((f) => f.severity === "high").length,
  medium: allFindings.filter((f) => f.severity === "medium").length,
  low: allFindings.filter((f) => f.severity === "low").length,
};
const hasBlocking = totals.critical > 0 || totals.high > 0;
const anyError = Object.values(scannerResults).some((r) => r.status === "error");
const anyFail = Object.values(scannerResults).some((r) => r.status === "fail");

let overallStatus;
if (hasBlocking || anyFail) overallStatus = "fail";
else if (totals.medium > 0) overallStatus = "pass_with_warnings";
else if (anyError) overallStatus = "pass_with_warnings";
else overallStatus = "pass";

const report = {
  generated_at: new Date().toISOString(),
  artifact_root: artifactRoot,
  audit_mode: mode,
  overall_status: overallStatus,
  blocking: mode === "blocking" && hasBlocking,
  total_findings: totals,
  scanners: scannerResults,
  duration_ms: Date.now() - auditStarted,
};

const outDir = path.resolve(workspaceRoot, artifactRoot, "verify");
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
const outPath = path.join(outDir, "static_audit.json");
fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
console.log(`[static_audit] wrote ${path.relative(workspaceRoot, outPath).replace(/\\/g, "/")} overall=${overallStatus}`);

// Emit targeted feedback hints for retry
try {
  writeFeedbackFiles({ workspaceRoot, artifactRoot, report });
} catch (err) {
  console.warn(`[static_audit] feedback emit warning: ${err?.message || err}`);
}

if (mode === "blocking" && report.blocking) {
  console.error(`[static_audit] BLOCKING: ${totals.critical} critical + ${totals.high} high findings`);
  process.exit(1);
}
process.exit(0);

// ── helpers ────────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) { out[key] = next; i++; }
      else out[key] = true;
    }
  }
  return out;
}

function writeFeedbackFiles({ workspaceRoot, artifactRoot, report }) {
  const metaDir = path.resolve(workspaceRoot, artifactRoot, "meta");
  if (!fs.existsSync(metaDir)) fs.mkdirSync(metaDir, { recursive: true });

  // Group findings by target step
  const feFindings = [];
  const beFindings = [];
  for (const f of Object.values(report.scanners).flatMap((r) => r.findings || [])) {
    const file = String(f.file || "");
    if (file.includes("fe_changes") || f.code?.startsWith("XSS") || f.code?.startsWith("HTML_")) {
      feFindings.push(f);
    } else {
      beFindings.push(f);
    }
  }

  if (feFindings.length > 0) {
    const body = renderFeedback("impl_fe_modules", feFindings);
    fs.writeFileSync(path.join(metaDir, "static_audit_feedback_impl_fe_modules.txt"), body);
  }
  if (beFindings.length > 0) {
    const body = renderFeedback("impl_be", beFindings);
    fs.writeFileSync(path.join(metaDir, "static_audit_feedback_impl_be.txt"), body);
  }
}

function renderFeedback(stepId, findings) {
  const lines = [`[STATIC AUDIT — ${findings.length} finding(s) for ${stepId}]`];
  for (const f of findings) {
    const loc = f.file && f.line ? `${f.file}:${f.line}` : f.endpoint || f.file || "(unknown)";
    lines.push(`- [${f.severity}] ${f.code} @ ${loc}`);
    if (f.detail) lines.push(`  detail: ${f.detail}`);
    if (f.fix_hint) lines.push(`  fix: ${f.fix_hint}`);
  }
  lines.push("");
  lines.push("Fix the issues above — do NOT reduce scope, ADD the missing checks.");
  return lines.join("\n");
}
