import pg from "pg";

const pool = new pg.Pool({
  host: process.env.PGHOST || "localhost",
  port: Number(process.env.PGPORT || 5432),
  user: process.env.PGUSER || "nexus",
  password: process.env.PGPASSWORD || "nexuspassword",
  database: process.env.PGDATABASE || "nexus",
});

async function checkDb() {
  try {
    const rules = await pool.query("SELECT * FROM rules");
    console.log("--- Rules Generated ---");
    console.table(rules.rows);

    const memories = await pool.query("SELECT * FROM mem_items");
    console.log("--- SOP Memories Generated ---");
    console.table(memories.rows);

  } catch (err) {
    console.error("DB Error:", err);
  } finally {
    pool.end();
  }
}

checkDb();
