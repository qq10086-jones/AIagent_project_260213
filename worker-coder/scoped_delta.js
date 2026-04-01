import fs from "fs";
import path from "path";
import crypto from "crypto";

import { normalizeRelPath } from "./scope_guard.js";

function shouldSkipScopedEntry(relPath) {
  const safe = normalizeRelPath(relPath);
  return (
    safe.startsWith("artifacts/runs/") ||
    safe.startsWith("node_modules/") ||
    safe.startsWith(".git/") ||
    safe.startsWith("dist/") ||
    safe.startsWith("build/")
  );
}

function hashFile(filePath) {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash("sha1").update(buf).digest("hex");
}

function walkScopedFiles(workspaceRoot, relPath, bucket) {
  const safeRel = normalizeRelPath(relPath);
  if (!safeRel || shouldSkipScopedEntry(safeRel)) return;
  const abs = path.resolve(workspaceRoot, safeRel);
  if (!abs.startsWith(path.resolve(workspaceRoot)) || !fs.existsSync(abs)) return;
  const stat = fs.statSync(abs);
  if (stat.isFile()) {
    bucket.push(safeRel);
    return;
  }
  const entries = fs.readdirSync(abs, { withFileTypes: true });
  for (const entry of entries) {
    const childRel = normalizeRelPath(path.posix.join(safeRel, entry.name));
    if (entry.isDirectory()) {
      if (shouldSkipScopedEntry(childRel)) continue;
      walkScopedFiles(workspaceRoot, childRel, bucket);
    } else if (entry.isFile()) {
      if (shouldSkipScopedEntry(childRel)) continue;
      bucket.push(childRel);
    }
  }
}

export function captureScopedSnapshot(workspaceRoot, targetPaths = []) {
  const files = [];
  for (const rel of Array.isArray(targetPaths) ? targetPaths : []) {
    walkScopedFiles(workspaceRoot, rel, files);
  }
  const snapshot = new Map();
  for (const rel of files) {
    const abs = path.resolve(workspaceRoot, rel);
    try {
      const stat = fs.statSync(abs);
      snapshot.set(rel, {
        hash: hashFile(abs),
        size: Number(stat.size || 0),
      });
    } catch {
      // ignore volatile files
    }
  }
  return snapshot;
}

export async function gatherGitSummary(workspaceRoot, taskDir, baselineSnapshot = new Map(), targetPaths = []) {
  const currentSnapshot = captureScopedSnapshot(workspaceRoot, targetPaths);
  const filesChanged = [];
  let added = 0;
  let deleted = 0;

  for (const [rel, meta] of currentSnapshot.entries()) {
    const before = baselineSnapshot.get(rel);
    if (!before || before.hash !== meta.hash) {
      filesChanged.push(rel);
      added += Number(meta.size || 0);
    }
  }
  for (const [rel, meta] of baselineSnapshot.entries()) {
    if (!currentSnapshot.has(rel)) {
      filesChanged.push(rel);
      deleted += Number(meta.size || 0);
    }
  }

  const dedupChanged = Array.from(new Set(filesChanged)).sort();
  let diffPath = null;
  try {
    diffPath = path.join(taskDir, `delegate_diff_${Date.now()}.patch`);
    const summary = {
      generated_at: new Date().toISOString(),
      target_paths: Array.isArray(targetPaths) ? targetPaths : [],
      files_changed: dedupChanged,
      baseline_count: baselineSnapshot.size,
      current_count: currentSnapshot.size,
    };
    fs.writeFileSync(diffPath, JSON.stringify(summary, null, 2), "utf8");
  } catch {
    diffPath = null;
  }

  return {
    filesChanged: dedupChanged,
    diffStats: { added, deleted, files: dedupChanged.length },
    diffPath,
    git: { base_ref: "SCOPED_SNAPSHOT", branch: "scoped", commit_sha: null, dirty: dedupChanged.length > 0 },
  };
}

export async function ensureImplementationDelta({
  workspaceRoot,
  stepId,
  taskId,
  executionAdapterPacket,
  targetPaths,
  taskPrompt,
  taskDir,
  baselineSnapshot,
  current,
}) {
  const safeStepId = String(stepId || "");
  if (!["impl_fe", "impl_be"].includes(safeStepId)) {
    return current;
  }
  if (Array.isArray(current?.filesChanged) && current.filesChanged.length > 0 && Number(current?.diffStats?.files || 0) > 0) {
    return current;
  }

  const effectiveTargetPaths = Array.isArray(executionAdapterPacket?.target_paths) && executionAdapterPacket.target_paths.length > 0
    ? executionAdapterPacket.target_paths
    : Array.isArray(targetPaths) && targetPaths.length > 0
      ? targetPaths
      : ["workspace/sandbox/crm_site/"];
  const firstTargetPath = String(effectiveTargetPaths[0] || "workspace/sandbox/crm_site/").replace(/\\/g, "/").replace(/\/+$/, "");
  const isFileTarget = /\.[A-Za-z0-9]+$/.test(firstTargetPath);
  if (isFileTarget) {
    return current;
  }

  const safeTaskId = String(taskId || "standalone").replace(/[^a-zA-Z0-9_-]/g, "_");
  const fileName = safeStepId === "impl_fe"
    ? `workflow_impl_fe_stub_${safeTaskId}.js`
    : `workflow_impl_be_stub_${safeTaskId}.js`;
  // Write the stub as an audit artifact inside taskDir only — NOT into workspaceRoot/sandbox.
  // impl_be and impl_fe are parallel steps so the stub cannot serve as a real BE→FE handoff.
  // Promoting it into the shared workspace pollutes candidate_files for all subsequent tasks.
  const targetAbs = path.resolve(taskDir, fileName);
  const stamp = new Date().toISOString();
  const promptSnippet = String(taskPrompt || "").slice(0, 160);
  fs.mkdirSync(path.dirname(targetAbs), { recursive: true });
  fs.writeFileSync(
    targetAbs,
    `// Deterministic scaffold emitted because no implementation delta was produced.\n` +
    `// step_id=${safeStepId}\n` +
    `// generated_at=${stamp}\n` +
    `export const workflow${safeStepId === "impl_fe" ? "Frontend" : "Backend"}Stub = ${JSON.stringify({
      step_id: safeStepId,
      generated_at: stamp,
      task_prompt: promptSnippet,
    }, null, 2)};\n`,
    "utf8"
  );
  // Return the original (empty) summary — workspace is unchanged, promotion is a no-op.
  return current;
}
