import fs from "fs";
import path from "path";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function loadJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function parseBool(v, fallback = true) {
  const s = String(v || "").trim().toLowerCase();
  if (!s) return fallback;
  return ["1", "true", "yes", "on"].includes(s);
}

function makeReportMd(report) {
  const lines = [
    "# Canary Report (coding_team_v0)",
    "",
    `- generated_at: ${report.generated_at}`,
    `- total_runs: ${report.total_runs}`,
    `- pass_runs: ${report.pass_runs}`,
    `- fail_runs: ${report.fail_runs}`,
    `- consecutive_green: ${report.consecutive_green}`,
    "",
    "## Runs",
  ];
  for (const r of report.runs) {
    lines.push(
      `- #${r.index} workflow=${r.workflow_run_id} run=${r.run_id} status=${r.workflow_status} gonogo=${r.go_no_go_verdict} duration_s=${r.duration_s}`
    );
    if (r.error_code || r.error_message) {
      lines.push(`  - error_code=${r.error_code || ""} error_message=${r.error_message || ""}`);
    }
  }
  return lines.join("\n");
}

async function startWorkflow(baseUrl, payload) {
  const r = await fetch(`${baseUrl}/workflow-runs/start`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) {
    throw new Error(`start failed: ${r.status} ${await r.text()}`);
  }
  return await r.json();
}

async function pollWorkflow(baseUrl, workflowRunId, timeoutSec = 1800, intervalMs = 4000) {
  const started = Date.now();
  while (Date.now() - started < timeoutSec * 1000) {
    const r = await fetch(`${baseUrl}/workflow-runs/${workflowRunId}`);
    if (!r.ok) throw new Error(`poll failed: ${r.status}`);
    const j = await r.json();
    const status = String(j?.run?.status || "");
    if (status === "succeeded" || status === "failed") {
      return j;
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting workflow ${workflowRunId}`);
}

function readGoNoGo(workspaceRoot, runId) {
  const p = path.resolve(workspaceRoot, "artifacts", "release", String(runId || ""), "qa", "go_no_go_result.json");
  if (!fs.existsSync(p)) {
    return { verdict: "NO_GO", file_exists: false, path: p };
  }
  try {
    const j = loadJson(p);
    return {
      verdict: String(j?.verdict || "NO_GO"),
      file_exists: true,
      path: p,
      checks_failed: Number(j?.failed_checks || 0),
    };
  } catch {
    return { verdict: "NO_GO", file_exists: true, path: p, parse_error: true };
  }
}

async function main() {
  const n = Math.max(1, Number(arg("n", "1")));
  const strict = parseBool(arg("strict", "true"), true);
  const inputArg = arg("input", "crm_mini.json");
  const timeoutSec = Math.max(120, Number(arg("timeout-sec", "1800")));
  const baseUrl = arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000");
  const workspaceRoot = path.resolve(arg("workspace-root", path.resolve(process.cwd(), "..")));
  const inputPath = path.resolve(process.cwd(), "canary_inputs", inputArg);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`input file not found: ${inputPath}`);
  }
  const payload = loadJson(inputPath);
  const runs = [];

  for (let i = 1; i <= n; i++) {
    const t0 = Date.now();
    let item = {
      index: i,
      workflow_run_id: "",
      run_id: "",
      workflow_status: "unknown",
      go_no_go_verdict: "NO_GO",
      duration_s: 0,
      error_code: "",
      error_message: "",
    };
    try {
      const startRes = await startWorkflow(baseUrl, payload);
      item.workflow_run_id = String(startRes.workflow_run_id || "");
      item.run_id = String(startRes.run_id || "");
      const terminal = await pollWorkflow(baseUrl, item.workflow_run_id, timeoutSec, 4000);
      item.workflow_status = String(terminal?.run?.status || "unknown");
      item.error_code = String(terminal?.run?.error_code || "");
      item.error_message = String(terminal?.run?.error_message || "");
      const go = readGoNoGo(workspaceRoot, item.run_id);
      item.go_no_go_verdict = go.verdict;
      item.go_no_go_file = go.path;
      item.go_no_go_file_exists = Boolean(go.file_exists);
      item.go_no_go_failed_checks = Number(go.checks_failed || 0);
    } catch (err) {
      item.workflow_status = "failed";
      item.error_message = err.message || String(err);
    } finally {
      item.duration_s = Math.round((Date.now() - t0) / 1000);
      runs.push(item);
    }
  }

  let consecutiveGreen = 0;
  for (const r of runs) {
    const green = r.workflow_status === "succeeded" && r.go_no_go_verdict === "GO";
    if (green) consecutiveGreen += 1;
    else consecutiveGreen = 0;
  }
  const passRuns = runs.filter((r) => r.workflow_status === "succeeded" && r.go_no_go_verdict === "GO").length;
  const failRuns = runs.length - passRuns;
  const report = {
    generated_at: new Date().toISOString(),
    strict,
    input_file: inputPath,
    total_runs: runs.length,
    pass_runs: passRuns,
    fail_runs: failRuns,
    consecutive_green: consecutiveGreen,
    runs,
  };
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outDir = path.resolve(workspaceRoot, "artifacts", "canary", "coding_team", stamp);
  fs.mkdirSync(outDir, { recursive: true });
  const jsonPath = path.join(outDir, "canary_report.json");
  const mdPath = path.join(outDir, "canary_report.md");
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), "utf8");
  fs.writeFileSync(mdPath, makeReportMd(report), "utf8");

  console.log(`# Canary Report`);
  console.log(`- json: ${jsonPath.replace(/\\/g, "/")}`);
  console.log(`- md: ${mdPath.replace(/\\/g, "/")}`);
  console.log(`- pass_runs: ${passRuns}`);
  console.log(`- fail_runs: ${failRuns}`);
  console.log(`- consecutive_green: ${consecutiveGreen}`);
  if (strict && failRuns > 0) process.exit(1);
}

main().catch((err) => {
  console.error(`[canary] failed: ${err.message || String(err)}`);
  process.exit(1);
});
