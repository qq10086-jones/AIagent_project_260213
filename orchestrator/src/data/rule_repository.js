/**
 * Data-access layer for rules.
 */

/**
 * @param {import('pg').Pool} pool
 * @param {string} project_id
 * @param {number} limit
 */
export async function listLatestRulesForProject(pool, project_id, limit = 5) {
  const row = await pool.query(
    "SELECT rule_json FROM rules WHERE project_id=$1 ORDER BY updated_at DESC LIMIT $2",
    [project_id, limit]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {{ rule_id:string, project_id:string, scope?:string, rule_type?:string, rule_json:object|string, weight?:number }} rule
 */
export async function insertRule(pool, rule) {
  await pool.query(
    `INSERT INTO rules(rule_id, project_id, scope, rule_type, rule_json, weight, updated_at)
     VALUES ($1, $2, $3, $4, $5, $6, NOW())`,
    [
      rule.rule_id,
      rule.project_id,
      rule.scope || "task",
      rule.rule_type || "soft",
      typeof rule.rule_json === "string" ? rule.rule_json : JSON.stringify(rule.rule_json || {}),
      Number.isFinite(rule.weight) ? rule.weight : 1,
    ]
  );
}
