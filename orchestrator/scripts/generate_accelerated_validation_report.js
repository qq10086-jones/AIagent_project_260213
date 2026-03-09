#!/usr/bin/env node
/**
 * generate_accelerated_validation_report.js
 *
 * Builds a compressed-validation evidence report for the last N minutes.
 *
 * Usage:
 *   node scripts/generate_accelerated_validation_report.js
 *   node scripts/generate_accelerated_validation_report.js --since-minutes 30
 *   node scripts/generate_accelerated_validation_report.js --since-minutes 30 --out orchestrator/artifacts/m6_trial/report.json
 *
 * Environment:
 *   DATABASE_URL, or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE
 */

import { writeFileSync } from "fs";
import { resolve } from "path";
import pg from "pg";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const cur = argv[i];
    if (!cur.startsWith("--")) continue;
    const key = cur.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i += 1;
    }
  }
  return out;
}

function toInt(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function percentile(sorted, pct) {
  if (!sorted.length) return 0;
  const idx = Math.ceil((pct / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(sorted.length - 1, idx))];
}

function mapCounts(rows, keyField, valueField = "count") {
  const out = {};
  for (const row of rows) {
    out[String(row[keyField] ?? "(null)")] = Number(row[valueField] ?? 0);
  }
  return out;
}

async function getAnchorTimestamp(pool, anchorMode) {
  if (anchorMode === "now") {
    return new Date();
  }

  const anchorQuery = await pool.query(
    `SELECT MAX(ts) AS anchor_ts
       FROM (
         SELECT MAX(created_at) AS ts FROM routing_decision_log
         UNION ALL
         SELECT MAX(created_at) AS ts FROM runs
         UNION ALL
         SELECT MAX(created_at) AS ts FROM workflow_runs
         UNION ALL
         SELECT MAX(started_at) AS ts FROM waterfall_stage_log
       ) q`
  );

  const value = anchorQuery.rows[0]?.anchor_ts;
  return value ? new Date(value) : new Date();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sinceMinutes = toInt(args["since-minutes"], 30);
  const anchorMode = String(args.anchor || "latest_db");
  const pool = process.env.DATABASE_URL
    ? new pg.Pool({ connectionString: process.env.DATABASE_URL })
    : new pg.Pool({
        host: process.env.PGHOST || "localhost",
        port: Number(process.env.PGPORT || 5432),
        user: process.env.PGUSER || "nexus",
        password: process.env.PGPASSWORD || "nexus",
        database: process.env.PGDATABASE || "nexus",
      });

  try {
    const anchorTime = await getAnchorTimestamp(pool, anchorMode);
    const windowStart = new Date(anchorTime.getTime() - sinceMinutes * 60 * 1000);

    const [
      decisionRows,
      denyRows,
      runRows,
      workflowRows,
      forcedSeqRows,
      stageRows,
    ] = await Promise.all([
      pool.query(
        `SELECT final_execution_decision, routing_decision_source, COUNT(*)::int AS count
           FROM routing_decision_log
          WHERE created_at >= $1
          GROUP BY final_execution_decision, routing_decision_source
          ORDER BY count DESC`,
        [windowStart]
      ),
      pool.query(
        `SELECT COALESCE(classifier_deny_or_degrade_reason, '(null)') AS reason, COUNT(*)::int AS count
           FROM routing_decision_log
          WHERE created_at >= $1
          GROUP BY COALESCE(classifier_deny_or_degrade_reason, '(null)')
          ORDER BY count DESC`,
        [windowStart]
      ),
      pool.query(
        `SELECT status, COUNT(*)::int AS count
           FROM runs
          WHERE created_at >= $1
          GROUP BY status
          ORDER BY count DESC`,
        [windowStart]
      ),
      pool.query(
        `SELECT status, COUNT(*)::int AS count
           FROM workflow_runs
          WHERE created_at >= $1
          GROUP BY status
          ORDER BY count DESC`,
        [windowStart]
      ),
      pool.query(
        `SELECT COUNT(*)::int AS count
           FROM routing_decision_log
          WHERE created_at >= $1
            AND final_execution_decision = 'forced_sequential'`,
        [windowStart]
      ),
      pool.query(
        `SELECT stage, duration_ms
           FROM waterfall_stage_log
          WHERE started_at >= $1
            AND duration_ms IS NOT NULL`,
        [windowStart]
      ),
    ]);

    const totalRouting = decisionRows.rows.reduce((sum, row) => sum + Number(row.count), 0);
    const totalRuns = runRows.rows.reduce((sum, row) => sum + Number(row.count), 0);
    const totalWorkflowRuns = workflowRows.rows.reduce((sum, row) => sum + Number(row.count), 0);

    const byDecision = {};
    const bySource = {};
    for (const row of decisionRows.rows) {
      byDecision[row.final_execution_decision] =
        (byDecision[row.final_execution_decision] || 0) + Number(row.count);
      bySource[row.routing_decision_source] =
        (bySource[row.routing_decision_source] || 0) + Number(row.count);
    }

    const stageBuckets = new Map();
    for (const row of stageRows.rows) {
      const stage = String(row.stage);
      const duration = Number(row.duration_ms || 0);
      if (!stageBuckets.has(stage)) stageBuckets.set(stage, []);
      stageBuckets.get(stage).push(duration);
    }

    const stagePercentiles = {};
    for (const [stage, values] of stageBuckets.entries()) {
      values.sort((a, b) => a - b);
      stagePercentiles[stage] = {
        sample_size: values.length,
        p50_ms: percentile(values, 50),
        p95_ms: percentile(values, 95),
      };
    }

    const forcedSequentialCount = Number(forcedSeqRows.rows[0]?.count || 0);
    const report = {
      generated_at: new Date().toISOString(),
      window: {
        anchor_mode: anchorMode,
        anchor_time: anchorTime.toISOString(),
        since_minutes: sinceMinutes,
        from: windowStart.toISOString(),
        to: anchorTime.toISOString(),
      },
      summary: {
        routing_samples: totalRouting,
        run_samples: totalRuns,
        workflow_run_samples: totalWorkflowRuns,
        gated_parallel_allowed: byDecision.gated_parallel_allowed || 0,
        forced_sequential: forcedSequentialCount,
        forced_sequential_ratio: totalRouting > 0 ? forcedSequentialCount / totalRouting : null,
      },
      routing: {
        by_final_execution_decision: byDecision,
        by_decision_source: bySource,
        deny_or_degrade_reasons: mapCounts(denyRows.rows, "reason"),
      },
      execution: {
        run_status_counts: mapCounts(runRows.rows, "status"),
        workflow_status_counts: mapCounts(workflowRows.rows, "status"),
      },
      latency: {
        by_stage: stagePercentiles,
      },
      compressed_go_no_go: {
        target_window_minutes: 30,
        thresholds: {
          min_routing_samples: 60,
          max_high_risk_misroutes: 0,
          max_forced_sequential_ratio: 0.85,
          require_execution_dispatch_samples: true,
        },
        evaluation: {
          enough_routing_samples: totalRouting >= 60,
          execution_dispatch_observed: Boolean(stagePercentiles.execution_dispatch?.sample_size),
          forced_sequential_ratio_within_limit:
            totalRouting > 0 ? forcedSequentialCount / totalRouting <= 0.85 : false,
        },
      },
    };

    const output = JSON.stringify(report, null, 2);
    if (args.out) {
      const outPath = resolve(process.cwd(), args.out);
      writeFileSync(outPath, output, "utf8");
      console.error(`[accelerated_validation_report] written to ${outPath}`);
    }
    process.stdout.write(output + "\n");
  } finally {
    await pool.end();
  }
}

main().catch((err) => {
  console.error("[accelerated_validation_report] fatal:", err.message || String(err));
  process.exit(1);
});
