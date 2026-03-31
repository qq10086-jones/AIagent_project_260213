import fs from "fs";
import path from "path";

import { resolveOrchestratorArtifactPath } from "./_paths.js";
import { main as runConfigPreflight } from "./validate_config_preflight.js";
import { main as runLiveVnextRuntime } from "./live_validate_vnext_runtime.js";
import { main as runLiveM9Workflow } from "./live_validate_m9_workflow.js";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function boolArg(name) {
  return process.argv.includes(`--${name}`);
}

function normalizePath(input) {
  return String(input || "").replace(/\\/g, "/");
}

function readJsonSafe(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

async function runStep(step) {
  const startedAt = new Date().toISOString();
  const finishedAt = new Date().toISOString();
  let ok = true;
  let error = null;
  let reportPath = step.fallbackReportPath || "";
  let report = null;
  try {
    const value = await step.run();
    report = value?.report || null;
    reportPath = String(value?.reportPath || reportPath || "").trim();
  } catch (err) {
    ok = false;
    error = String(err?.message || err);
  }
  return {
    id: step.id,
    label: step.label,
    started_at: startedAt,
    finished_at: finishedAt,
    ok,
    exit_code: ok ? 0 : 1,
    timed_out: false,
    error,
    report_path: reportPath ? normalizePath(path.resolve(process.cwd(), reportPath)) : null,
    report,
  };
}

function summarize(steps = []) {
  return {
    total: steps.length,
    passed: steps.filter((step) => step.ok).length,
    failed: steps.filter((step) => !step.ok).length,
  };
}

async function main() {
  const baseUrl = String(arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const approvalToken = String(arg("approval-token", process.env.APPROVAL_TOKEN || "dev-approval-token"));
  const runtimeTimeoutMs = Math.max(60000, Number(arg("runtime-timeout-ms", "240000")));
  const workflowTimeoutMs = Math.max(120000, Number(arg("workflow-timeout-ms", "420000")));
  const includeLive = !boolArg("skip-live");

  const steps = [
    {
      id: "config_preflight",
      label: "Config Preflight",
      async run() {
        return runConfigPreflight();
      },
    },
  ];

  if (includeLive) {
    steps.push(
      {
        id: "live_vnext_runtime",
        label: "Live vNext Runtime",
        fallbackReportPath: path.join("artifacts", "canary", "live_vnext_runtime", "live_vnext_runtime_report.json"),
        async run() {
          return runLiveVnextRuntime({
            baseUrl,
            approvalToken,
            timeoutMs: runtimeTimeoutMs,
          });
        },
      },
      {
        id: "live_m9_workflow",
        label: "Live M9 Workflow",
        fallbackReportPath: path.join("artifacts", "canary", "live_m9_workflow", "live_m9_workflow_report.json"),
        async run() {
          return runLiveM9Workflow({
            baseUrl,
            approvalToken,
            timeoutMs: workflowTimeoutMs,
          });
        },
      },
    );
  }

  const report = {
    generated_at: new Date().toISOString(),
    overall: "pass",
    base_url: baseUrl,
    include_live: includeLive,
    steps: [],
  };

  for (const step of steps) {
    const result = await runStep(step);
    if (!result.report && result.report_path && fs.existsSync(result.report_path)) {
      result.report = readJsonSafe(result.report_path);
    }
    report.steps.push(result);
    if (!result.ok) {
      report.overall = "fail";
      break;
    }
  }

  report.summary = summarize(report.steps);

  const outDir = resolveOrchestratorArtifactPath("validation", "next_stage_release_gate");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "next_stage_release_gate_summary.json");
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");

  console.log("# Next-Stage Release Gate");
  console.log(`- summary: ${normalizePath(outPath)}`);
  console.log(`- overall: ${report.overall}`);
  console.log(`- steps_passed: ${report.summary.passed}/${report.summary.total}`);
  if (!includeLive) console.log("- mode: config_preflight_only");
  if (report.overall !== "pass") {
    const failed = report.steps.find((step) => !step.ok);
    if (failed) console.log(`- failed_step: ${failed.id}`);
    process.exit(1);
  }
}

try {
  await main();
} catch (err) {
  console.error(err?.message || err);
  process.exit(1);
}

