#!/usr/bin/env node
/**
 * generate_routing_evaluation_report.js
 *
 * WS-30-03: CLI artifact writer for the Routing Evaluation Report.
 *
 * Usage:
 *   node scripts/generate_routing_evaluation_report.js \
 *     [--from 2026-03-01] [--to 2026-03-09] \
 *     [--manifest path/to/m6_staging_replay_manifest.json] \
 *     [--out path/to/output.json]
 *
 * Environment variables:
 *   DATABASE_URL   PostgreSQL connection string (required)
 *
 * The report is written to stdout (pretty-printed JSON) and optionally
 * to a file via --out.
 */

import { createRequire } from "module";
import { writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import pg from "pg";

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKSPACE_ROOT = resolve(__dirname, "..");

// ── Lazy-import domain services ───────────────────────────────────────────────

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const { createRoutingEvaluationReportService } = await import(
    "../src/domain/routing_evaluation_report.js"
  );
  const { createWaterfallTraceService } = await import(
    "../src/domain/waterfall_trace_service.js"
  );

  const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

  try {
    const waterfallTraceService = createWaterfallTraceService({ pool });
    const reportService = createRoutingEvaluationReportService({
      pool,
      workspaceRoot: WORKSPACE_ROOT,
      waterfallTraceService,
    });

    const manifestPath = args.manifest
      ? resolve(WORKSPACE_ROOT, args.manifest)
      : resolve(WORKSPACE_ROOT, "replay/manifests/m6_staging_replay_manifest.json");

    console.error("[routing_evaluation_report] generating report...");
    const report = await reportService.generateReport({
      from:               args.from   ? new Date(args.from)  : null,
      to:                 args.to     ? new Date(args.to)    : null,
      replayManifestPath: manifestPath,
    });

    const output = JSON.stringify(report, null, 2);

    if (args.out) {
      const outPath = resolve(process.cwd(), args.out);
      writeFileSync(outPath, output, "utf8");
      console.error(`[routing_evaluation_report] written to ${outPath}`);
    }

    process.stdout.write(output + "\n");
  } finally {
    await pool.end();
  }
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      out[key] = argv[i + 1] ?? true;
      i++;
    }
  }
  return out;
}

main().catch((err) => {
  console.error("[routing_evaluation_report] fatal:", err.message);
  process.exit(1);
});
