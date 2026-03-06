import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { validateJsonSchemaLite } from "./schema_lite_validator.js";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const CONTRACTS_DIR = path.resolve(MODULE_DIR, "..", "contracts");
const ARTIFACT_METADATA_SCHEMA = JSON.parse(
  fs.readFileSync(path.resolve(CONTRACTS_DIR, "artifact_metadata.schema.json"), "utf8")
);

function inferArtifactType(objectKey = "", mime = "") {
  const key = String(objectKey || "").toLowerCase();
  const mimeType = String(mime || "").toLowerCase();
  if (key.endsWith(".patch") || key.endsWith(".diff")) return "patch";
  if (key.endsWith(".md")) return key.includes("runbook") || key.includes("/run/") ? "runbook" : "document";
  if (key.endsWith(".json")) return "report";
  if (mimeType.startsWith("image/")) return "screenshot";
  return "artifact";
}

function inferSummary({ role = "", type = "", objectKey = "", source = "" }) {
  return [String(role || "system"), String(type || "artifact"), String(source || "unknown"), String(objectKey || "")]
    .filter(Boolean)
    .join(":");
}

export function buildArtifactMetadata({
  artifactId = "",
  taskId = "",
  role = "",
  objectKey = "",
  mime = "application/octet-stream",
  createdAt = "",
  source = "",
  summary = "",
}) {
  const type = inferArtifactType(objectKey, mime);
  return {
    artifact_id: String(artifactId || crypto.createHash("sha1").update(`${taskId}|${objectKey}|${createdAt}`).digest("hex")),
    task_id: String(taskId || ""),
    role: String(role || "system"),
    type,
    path: String(objectKey || ""),
    mime: String(mime || "application/octet-stream"),
    created_at: String(createdAt || new Date().toISOString()),
    summary: String(summary || inferSummary({ role, type, objectKey, source })),
    source: String(source || ""),
  };
}

export function validateArtifactMetadata(metadata) {
  const errors = validateJsonSchemaLite(ARTIFACT_METADATA_SCHEMA, metadata, "$");
  return {
    ok: errors.length === 0,
    errors,
    schema_id: ARTIFACT_METADATA_SCHEMA.$id,
  };
}
