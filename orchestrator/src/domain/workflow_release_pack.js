import crypto from "crypto";
import fs from "fs";
import path from "path";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { buildArtifactMetadata, validateArtifactMetadata } from "../artifact_registry.js";
import { insertAssetRecord } from "../data/asset_repository.js";

export function createWorkflowReleasePackService({
  pool,
  recordEvent,
  workspaceRoot = "/workspace",
  minio = null,
}) {
  const minioCfg = minio || {};
  const minioEnabled = Boolean(minioCfg.enabled);
  const minioBucket = String(minioCfg.bucket || "nexus-artifacts");
  const s3 = minioEnabled
    ? new S3Client({
        endpoint: String(minioCfg.endpoint || "http://nexus-minio:9000"),
        credentials: {
          accessKeyId: String(minioCfg.accessKey || "nexus"),
          secretAccessKey: String(minioCfg.secretKey || "nexuspassword"),
        },
        region: "us-east-1",
        forcePathStyle: true,
      })
    : null;

  async function archiveReleasePackToMinio({ run, manifestPath, summaryPath, extraPaths = [] }) {
    if (!s3) return [];
    const out = [];
    const files = [manifestPath, summaryPath, ...(Array.isArray(extraPaths) ? extraPaths : [])].filter(
      (filePath) => filePath && fs.existsSync(filePath)
    );
    for (const filePath of files) {
      try {
        const data = fs.readFileSync(filePath);
        const ext = path.extname(filePath).replace(/^\./, "") || "bin";
        const key = `release/${String(run.run_id || run.workflow_run_id)}/${path.basename(filePath, path.extname(filePath))}.${ext}`;
        await s3.send(
          new PutObjectCommand({
            Bucket: minioBucket,
            Key: key,
            Body: data,
            ContentType: filePath.endsWith(".json") ? "application/json" : "text/markdown",
          })
        );
        const sha256 = crypto.createHash("sha256").update(data).digest("hex");
        out.push({
          object_key: key,
          bucket: minioBucket,
          sha256,
          mime_type: filePath.endsWith(".json") ? "application/json" : "text/markdown",
          file_size: Number(data.length || 0),
          local_path: filePath.replace(/\\/g, "/"),
        });
      } catch (err) {
        await recordEvent(run.workflow_run_id, "artifact.pack.minio.archive_failed", {
          file: filePath,
          error: err.message || String(err),
        });
      }
    }
    return out;
  }

  async function indexReleasePackToDb({
    run,
    manifestPath,
    summaryPath,
    extraPaths = [],
    stepArtifacts,
    minioArchived = [],
  }) {
    const files = [manifestPath, summaryPath, ...(Array.isArray(extraPaths) ? extraPaths : [])].filter(
      (filePath) => filePath && fs.existsSync(filePath)
    );
    for (const filePath of files) {
      try {
        const buf = fs.readFileSync(filePath);
        const sha256 = crypto.createHash("sha256").update(buf).digest("hex");
        const stat = fs.statSync(filePath);
        const relPath = path.relative(workspaceRoot, filePath).replace(/\\/g, "/");
        const metadata = buildArtifactMetadata({
          taskId: `workflow_run:${run.workflow_run_id}`,
          role: "release",
          objectKey: relPath,
          mime: filePath.endsWith(".json") ? "application/json" : "text/markdown",
          createdAt: new Date().toISOString(),
          source: "release_pack_local",
        });
        const checked = validateArtifactMetadata(metadata);
        if (!checked.ok) continue;
        await insertAssetRecord(pool, {
          task_id: metadata.task_id,
          object_key: metadata.path,
          sha256,
          mime_type: metadata.mime,
          file_size: Number(stat.size || 0),
          metadata_json: {
            artifact_metadata: metadata,
            run_id: run.run_id,
            workflow_run_id: run.workflow_run_id,
            source: "release_pack_local",
          },
        });
      } catch { /* ignore: individual asset record insert is non-fatal */ }
    }

    for (const item of stepArtifacts || []) {
      for (const art of item.artifacts || []) {
        if (!art || (!art.object_key && !art.name)) continue;
        try {
          const metadata = buildArtifactMetadata({
            taskId: `workflow_run:${run.workflow_run_id}`,
            role: item.step_id || "worker",
            objectKey: String(art.object_key || art.name || "unknown"),
            mime: String(art.mime || "application/octet-stream"),
            createdAt: new Date().toISOString(),
            source: "step_artifact_ref",
          });
          const checked = validateArtifactMetadata(metadata);
          if (!checked.ok) continue;
          await insertAssetRecord(pool, {
            task_id: metadata.task_id,
            object_key: metadata.path,
            sha256: String(art.sha256 || ""),
            mime_type: metadata.mime,
            file_size: Number(art.file_size || 0),
            metadata_json: {
              artifact_metadata: metadata,
              run_id: run.run_id,
              workflow_run_id: run.workflow_run_id,
              source: "step_artifact_ref",
              step_index: Number(item.step_index),
              step_id: item.step_id,
              bucket: art.bucket || null,
            },
          });
        } catch { /* ignore: individual asset record insert is non-fatal */ }
      }
    }

    for (const art of minioArchived || []) {
      try {
        const metadata = buildArtifactMetadata({
          taskId: `workflow_run:${run.workflow_run_id}`,
          role: "release",
          objectKey: String(art.object_key || ""),
          mime: String(art.mime_type || "application/octet-stream"),
          createdAt: new Date().toISOString(),
          source: "release_pack_minio",
        });
        const checked = validateArtifactMetadata(metadata);
        if (!checked.ok) continue;
        await insertAssetRecord(pool, {
          task_id: metadata.task_id,
          object_key: metadata.path,
          sha256: String(art.sha256 || ""),
          mime_type: metadata.mime,
          file_size: Number(art.file_size || 0),
          metadata_json: {
            artifact_metadata: metadata,
            run_id: run.run_id,
            workflow_run_id: run.workflow_run_id,
            source: "release_pack_minio",
            bucket: art.bucket || minioBucket,
            local_path: art.local_path || null,
          },
        });
      } catch { /* ignore: individual asset record insert is non-fatal */ }
    }
  }

  return {
    archiveReleasePackToMinio,
    indexReleasePackToDb,
    minioBucket,
  };
}
