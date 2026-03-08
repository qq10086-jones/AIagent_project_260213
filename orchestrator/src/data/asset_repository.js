export async function insertAssetRecord(pool, asset) {
  await pool.query(
    `INSERT INTO assets(task_id, object_key, sha256, mime_type, file_size, metadata_json)
     VALUES ($1,$2,$3,$4,$5,$6)`,
    [
      asset.task_id,
      asset.object_key,
      asset.sha256,
      asset.mime_type,
      Number(asset.file_size || 0),
      JSON.stringify(asset.metadata_json || {}),
    ]
  );
}
