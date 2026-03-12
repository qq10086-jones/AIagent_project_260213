#!/usr/bin/env node

import fs from "fs";
import path from "path";
import http from "http";
import pg from "pg";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (!cur.startsWith("--")) continue;
    const key = cur.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) out[key] = true;
    else {
      out[key] = next;
      i += 1;
    }
  }
  return out;
}

function toInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : fallback;
}

function percentile(sorted, pct) {
  if (!sorted.length) return 0;
  const idx = Math.ceil((pct / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, idx))];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function writeText(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, value, "utf8");
}

function sendJsonRequest({ hostname, port, endpoint, timeoutMs }, body) {
  return new Promise((resolve) => {
    const payload = JSON.stringify(body);
    const req = http.request(
      {
        hostname,
        port,
        path: endpoint,
        method: "POST",
        timeout: timeoutMs,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            resolve({
              ok: res.statusCode >= 200 && res.statusCode < 300,
              statusCode: res.statusCode,
              body: data ? JSON.parse(data) : null,
            });
          } catch {
            resolve({
              ok: false,
              statusCode: res.statusCode,
              body: { raw: data },
            });
          }
        });
      }
    );

    req.on("timeout", () => req.destroy(new Error("request timeout")));
    req.on("error", (err) => {
      resolve({ ok: false, statusCode: 0, body: { error: err.message || String(err) } });
    });

    req.write(payload);
    req.end();
  });
}

function buildPrompt({ index, cls, tag }) {
  const shortTemplates = [
    "Build a lightweight CRM customer list placeholder with release evidence",
    "Create a minimal CRM dashboard stub with frontend-safe backend coordination",
  ];
  const mediumTemplates = [
    "Implement a CRM customer list flow across PM, architecture, backend endpoint, frontend page, QA verification, and release packaging",
    "Build a CRM dashboard feature spanning backend API, frontend UI, QA verification, and release artifacts",
  ];
  const longTemplates = [
    "Deliver a full CRM feature with PM spec, architecture, backend service wiring, frontend implementation, QA verification, release notes, and final artifact packaging under enforced parallel routing",
    "Plan and implement a CRM module requiring release-pack evidence, QA traceability, and full frontend/backend DAG completion under enforced routing",
  ];
  const pool = cls === "short" ? shortTemplates : cls === "medium" ? mediumTemplates : longTemplates;
  const base = pool[index % pool.length];
  return `${base} ${tag} class=${cls} seq=${index + 1}`;
}

function buildClasses(warmupCount, baselineCount) {
  const classes = [];
  const baseline = baselineCount;
  const shortCount = Math.round(baseline * 0.4);
  const mediumCount = Math.round(baseline * 0.4);
  const longCount = Math.max(0, baseline - shortCount - mediumCount);
  for (let i = 0; i < warmupCount; i += 1) classes.push("short");
  for (let i = 0; i < shortCount; i += 1) classes.push("short");
  for (let i = 0; i < mediumCount; i += 1) classes.push("medium");
  for (let i = 0; i < longCount; i += 1) classes.push("long");
  return classes;
}

function summarizeCounts(rows, key = "status") {
  const out = {};
  for (const row of rows) out[String(row[key] || "")] = Number(row.count || 0);
  return out;
}

function computeStageStats(rows) {
  const byStage = new Map();
  for (const row of rows) {
    const stage = String(row.stage || "");
    const duration = Number(row.duration_ms || 0);
    if (!byStage.has(stage)) byStage.set(stage, []);
    byStage.get(stage).push(duration);
  }
  const out = {};
  for (const [stage, values] of byStage.entries()) {
    values.sort((a, b) => a - b);
    out[stage] = {
      sample_size: values.length,
      p50_ms: percentile(values, 50),
      p95_ms: percentile(values, 95),
    };
  }
  return out;
}

function findReleaseRoot(repoRoot, runId) {
  const candidates = [
    path.join(repoRoot, "artifacts", "release", runId),
    path.join(repoRoot, "orchestrator", "artifacts", "release", runId),
  ];
  return candidates.find((item) => fs.existsSync(item)) || candidates[0];
}

function loadJsonIfExists(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const hostname = String(args.host || "localhost");
  const port = toInt(args.port, 3000);
  const endpoint = String(args.endpoint || "/vnext/dispatch");
  const provider = String(args.provider || "opencode");
  const model = String(args.model || "alibaba-coding-plan/qwen3-coder-plus");
  const warmupCount = toInt(args.warmup, 6);
  const baselineCount = toInt(args.baseline, 30);
  const intervalMs = toInt(args["interval-ms"], 1500);
  const timeoutMs = toInt(args["timeout-ms"], 30000);
  const pollMs = toInt(args["poll-ms"], 10000);
  const settleTimeoutSec = toInt(args["settle-timeout-sec"], 1800);
  const repoRoot = process.cwd();
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const tag = `[M10LoadTest][${stamp}]`;
  const artifactDir = path.join(repoRoot, "orchestrator", "artifacts", "validation", "m10_load_test", stamp);

  const pool = process.env.DATABASE_URL
    ? new pg.Pool({ connectionString: process.env.DATABASE_URL })
    : new pg.Pool({
        host: process.env.PGHOST || "localhost",
        port: Number(process.env.PGPORT || 5432),
        user: process.env.PGUSER || "nexus",
        password: process.env.PGPASSWORD || "nexus",
        database: process.env.PGDATABASE || "nexus",
      });

  const startedAt = new Date();
  const classes = buildClasses(warmupCount, baselineCount);
  const dispatchResults = [];

  try {
    console.log(`[m10-load-test] tag=${tag}`);
    console.log(`[m10-load-test] target=${hostname}:${port}${endpoint} warmup=${warmupCount} baseline=${baselineCount} interval_ms=${intervalMs} provider=${provider} model=${model}`);

    for (let i = 0; i < classes.length; i += 1) {
      const cls = classes[i];
      const prompt = buildPrompt({ index: i, cls, tag });
      const body =
        endpoint === "/vnext/dispatch"
          ? {
              raw_input: prompt,
              provider,
              model,
              fast_mode: false,
              payload: { destructive: false, validation_window: "m10_load_test", input_class: "fe_led" },
            }
          : {
              message: `/coder: ${prompt}`,
              payload: { destructive: false, validation_window: "m10_load_test" },
            };
      const result = await sendJsonRequest(
        { hostname, port, endpoint, timeoutMs },
        body
      );
      const runId = String(result.body?.run_id || "");
      dispatchResults.push({
        index: i + 1,
        class: cls,
        ok: Boolean(result.ok && result.body?.ok !== false && runId),
        status_code: result.statusCode,
        run_id: runId,
        mode: String(result.body?.mode || ""),
        workflow_run_id: String(result.body?.workflow_run_id || ""),
        raw_error: result.body?.error || result.body?.raw || "",
      });
      console.log(`[m10-load-test] dispatch ${i + 1}/${classes.length} class=${cls} status=${result.statusCode} run=${runId || "n/a"} ok=${dispatchResults[i].ok}`);
      if (i < classes.length - 1) await sleep(intervalMs);
    }

    const runIds = dispatchResults.map((item) => item.run_id).filter(Boolean);
    const settleDeadline = Date.now() + settleTimeoutSec * 1000;
    let pendingRunIds = [...runIds];

    while (pendingRunIds.length > 0 && Date.now() < settleDeadline) {
      const runRes = await pool.query(
        `SELECT run_id, status FROM runs WHERE run_id = ANY($1::text[])`,
        [pendingRunIds]
      );
      const nonTerminal = runRes.rows
        .filter((row) => !["completed", "failed"].includes(String(row.status || "")))
        .map((row) => String(row.run_id || ""));
      pendingRunIds = nonTerminal;
      console.log(`[m10-load-test] settling pending_runs=${pendingRunIds.length}`);
      if (pendingRunIds.length > 0) await sleep(pollMs);
    }

    const finishedAt = new Date();
    const runRows = (
      await pool.query(
        `SELECT run_id, status, input_text
           FROM runs
          WHERE run_id = ANY($1::text[])
          ORDER BY run_id ASC`,
        [runIds]
      )
    ).rows;
    const workflowRunRows = (
      await pool.query(
        `SELECT workflow_run_id, run_id, workflow_id, project_type, status, error_code, created_at, updated_at
           FROM workflow_runs
          WHERE run_id = ANY($1::text[])
          ORDER BY created_at ASC`,
        [runIds]
      )
    ).rows;
    const workflowStepRows = (
      await pool.query(
        `SELECT workflow_run_id, step_id, status, error_code, started_at, ended_at
           FROM workflow_steps
          WHERE workflow_run_id = ANY($1::text[])
          ORDER BY workflow_run_id ASC, step_id ASC`,
        [workflowRunRows.map((item) => item.workflow_run_id)]
      )
    ).rows;
    const routingRows = (
      await pool.query(
        `SELECT run_id, workflow_run_id, routing_decision_source, final_execution_decision, created_at
           FROM routing_decision_log
          WHERE run_id = ANY($1::text[])
          ORDER BY created_at ASC`,
        [runIds]
      )
    ).rows;
    const waterfallRows = (
      await pool.query(
        `SELECT run_id, workflow_run_id, stage, duration_ms
           FROM waterfall_stage_log
          WHERE run_id = ANY($1::text[])
          ORDER BY run_id ASC, started_at ASC`,
        [runIds]
      )
    ).rows;

    const runStatusCounts = summarizeCounts(
      (await pool.query(`SELECT status, COUNT(*)::int AS count FROM runs WHERE run_id = ANY($1::text[]) GROUP BY status`, [runIds])).rows
    );
    const workflowStatusCounts = summarizeCounts(
      (await pool.query(`SELECT status, COUNT(*)::int AS count FROM workflow_runs WHERE run_id = ANY($1::text[]) GROUP BY status`, [runIds])).rows
    );

    const duplicateWorkflowRuns = Object.entries(
      workflowRunRows.reduce((acc, row) => {
        acc[row.run_id] = (acc[row.run_id] || 0) + 1;
        return acc;
      }, {})
    ).filter(([, count]) => count > 1);

    const releaseChecks = runIds.map((runId) => {
      const releaseRoot = findReleaseRoot(repoRoot, runId);
      const goNoGoPath = path.join(releaseRoot, "qa", "go_no_go_result.json");
      const strictCanaryPath = path.join(releaseRoot, "qa", "strict_canary_report.json");
      const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
      const goNoGo = loadJsonIfExists(goNoGoPath);
      const strictCanary = loadJsonIfExists(strictCanaryPath);
      const manifest = loadJsonIfExists(manifestPath);
      return {
        run_id: runId,
        release_root: path.relative(repoRoot, releaseRoot).replace(/\\/g, "/"),
        go_no_go_exists: Boolean(goNoGo),
        go_no_go_verdict: String(goNoGo?.verdict || ""),
        strict_canary_verdict: String(strictCanary?.verdict || ""),
        manifest_exists: Boolean(manifest),
      };
    });

    const immediateFailures = [];
    if (duplicateWorkflowRuns.length > 0) immediateFailures.push("duplicate workflow finalization candidates detected");
    if (releaseChecks.some((item) => !item.go_no_go_exists)) immediateFailures.push("missing go_no_go_result.json on at least one run");
    if (releaseChecks.some((item) => item.go_no_go_exists && item.go_no_go_verdict !== "GO")) immediateFailures.push("at least one succeeded release pack is not GO");

    const stageStats = computeStageStats(waterfallRows);
    const routingOverheadP95 =
      Number(stageStats.intake?.p95_ms || 0) +
      Number(stageStats.routing?.p95_ms || 0) +
      Number(stageStats.policy_evaluation?.p95_ms || 0);

    const report = {
      generated_at: new Date().toISOString(),
      tag,
      config: {
        hostname,
        port,
        endpoint,
        provider,
        model,
        warmup_count: warmupCount,
        baseline_count: baselineCount,
        interval_ms: intervalMs,
        timeout_ms: timeoutMs,
        poll_ms: pollMs,
        settle_timeout_sec: settleTimeoutSec,
      },
      execution_window: {
        started_at: startedAt.toISOString(),
        finished_at: finishedAt.toISOString(),
        elapsed_ms: finishedAt.getTime() - startedAt.getTime(),
      },
      dispatch: {
        attempted: dispatchResults.length,
        accepted: dispatchResults.filter((item) => item.ok).length,
        failed: dispatchResults.filter((item) => !item.ok).length,
        results: dispatchResults,
      },
      runs: {
        total_run_ids: runIds.length,
        run_status_counts: runStatusCounts,
        workflow_status_counts: workflowStatusCounts,
        duplicate_workflow_runs: duplicateWorkflowRuns.map(([run_id, count]) => ({ run_id, count })),
        unsettled_run_ids: runRows.filter((row) => !["completed", "failed"].includes(String(row.status || ""))).map((row) => row.run_id),
      },
      routing: {
        rows: routingRows.length,
        gated_parallel_allowed: routingRows.filter((row) => row.final_execution_decision === "gated_parallel_allowed").length,
      },
      latency: {
        by_stage: stageStats,
        routing_overhead_p95_ms: routingOverheadP95,
      },
      release_pack: {
        checks: releaseChecks,
        go_count: releaseChecks.filter((item) => item.go_no_go_verdict === "GO").length,
      },
      workflow_steps: {
        total: workflowStepRows.length,
        failed: workflowStepRows.filter((row) => String(row.status || "") === "failed").length,
      },
      verdict: immediateFailures.length === 0 ? "PASS" : "FAIL",
      failure_reasons: immediateFailures,
    };

    const mdLines = [
      "# M10 Load Test Report",
      "",
      `- tag: ${tag}`,
      `- verdict: ${report.verdict}`,
      `- attempted: ${report.dispatch.attempted}`,
      `- accepted: ${report.dispatch.accepted}`,
      `- run_ids: ${report.runs.total_run_ids}`,
      `- workflow_status_counts: ${JSON.stringify(report.runs.workflow_status_counts)}`,
      `- go_count: ${report.release_pack.go_count}`,
      `- routing_rows: ${report.routing.rows}`,
      `- gated_parallel_allowed: ${report.routing.gated_parallel_allowed}`,
      `- policy_evaluation_p95_ms: ${Number(report.latency.by_stage.policy_evaluation?.p95_ms || 0)}`,
      `- routing_overhead_p95_ms: ${Number(report.latency.routing_overhead_p95_ms || 0)}`,
      "",
      "## Failure Reasons",
      ...(report.failure_reasons.length > 0 ? report.failure_reasons.map((item) => `- ${item}`) : ["- none"]),
    ];

    writeJson(path.join(artifactDir, "m10_load_test_report.json"), report);
    writeText(path.join(artifactDir, "m10_load_test_report.md"), `${mdLines.join("\n")}\n`);
    console.log(`[m10-load-test] verdict=${report.verdict}`);
    console.log(`[m10-load-test] artifact=orchestrator/artifacts/validation/m10_load_test/${stamp}/m10_load_test_report.json`);
    if (report.verdict !== "PASS") process.exitCode = 1;
  } finally {
    await pool.end();
  }
}

main().catch((err) => {
  console.error(`[m10-load-test] fatal: ${err?.stack || err?.message || err}`);
  process.exit(1);
});
