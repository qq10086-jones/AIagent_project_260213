#!/usr/bin/env node

import pg from "pg";

async function main() {
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
    const sql = `
      SELECT
        run_id,
        workflow_id,
        router_mode,
        dynamic_routing_enabled,
        classifier_domain_lead,
        classifier_confidence_band,
        routing_decision_source,
        final_execution_decision,
        created_at
      FROM routing_decision_log
      ORDER BY created_at DESC
      LIMIT 20
    `;
    const result = await pool.query(sql);
    process.stdout.write(JSON.stringify(result.rows, null, 2) + "\n");
  } finally {
    await pool.end();
  }
}

main().catch((err) => {
  console.error("[query_latest_routing_rows] fatal:", err.message || String(err));
  process.exit(1);
});
