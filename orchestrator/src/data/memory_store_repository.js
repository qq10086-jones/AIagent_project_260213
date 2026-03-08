/**
 * Data-access layer for mem_items.
 */

/**
 * @param {import('pg').Pool} pool
 * @param {string} project_id
 * @param {number} limit
 */
export async function listRecentMemoryItemsForProject(pool, project_id, limit = 3) {
  const row = await pool.query(
    "SELECT content FROM mem_items WHERE project_id=$1 ORDER BY created_at DESC LIMIT $2",
    [project_id, limit]
  );
  return row.rows || [];
}

/**
 * @param {import('pg').Pool} pool
 * @param {{ mem_id:string, project_id:string, type?:string, content:object|string, tags?:string }} item
 */
export async function insertMemoryItem(pool, item) {
  await pool.query(
    `INSERT INTO mem_items(mem_id, project_id, type, content, tags, created_at)
     VALUES ($1, $2, $3, $4, $5, NOW())`,
    [
      item.mem_id,
      item.project_id,
      item.type || "sop",
      typeof item.content === "string" ? item.content : JSON.stringify(item.content || {}),
      item.tags || "auto_generated",
    ]
  );
}
