import fs from "fs";
import path from "path";

import { normalizeRelPath } from "./scope_guard.js";

function isEnabled(mode) {
  const normalized = String(mode || "").trim().toLowerCase();
  return normalized === "scaffold" || normalized === "shadow" || normalized === "promote";
}

function safeMode(mode) {
  const normalized = String(mode || "").trim().toLowerCase();
  return isEnabled(normalized) ? normalized : "disabled";
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function hashlessStatEntry(root, absPath) {
  const relPath = normalizeRelPath(path.relative(root, absPath));
  const stat = fs.statSync(absPath);
  return {
    path: relPath,
    type: stat.isDirectory() ? "directory" : "file",
    size: stat.isFile() ? Number(stat.size || 0) : 0,
  };
}

function listEntries(root, relPath, bucket) {
  const absPath = path.resolve(root, relPath);
  if (!fs.existsSync(absPath)) return;
  const stat = fs.statSync(absPath);
  bucket.push(hashlessStatEntry(root, absPath));
  if (!stat.isDirectory()) return;
  for (const entry of fs.readdirSync(absPath, { withFileTypes: true })) {
    listEntries(root, path.join(relPath, entry.name), bucket);
  }
}

function copyRecursive(srcAbs, dstAbs) {
  const stat = fs.statSync(srcAbs);
  if (stat.isDirectory()) {
    ensureDir(dstAbs);
    for (const entry of fs.readdirSync(srcAbs, { withFileTypes: true })) {
      copyRecursive(path.join(srcAbs, entry.name), path.join(dstAbs, entry.name));
    }
    return;
  }
  ensureDir(path.dirname(dstAbs));
  fs.copyFileSync(srcAbs, dstAbs);
}

function uniqueList(values = []) {
  return Array.from(new Set(values.filter(Boolean)));
}

export function inferIsolationContextPaths(targetPaths = []) {
  const normalizedTargets = Array.isArray(targetPaths)
    ? targetPaths.map((item) => normalizeRelPath(item)).filter(Boolean)
    : [];
  const contextPaths = [];
  for (const relPath of normalizedTargets) {
    if (relPath === "workspace/sandbox/crm_site" || relPath.startsWith("workspace/sandbox/crm_site/")) {
      contextPaths.push("workspace/sandbox/crm_site");
      continue;
    }
    contextPaths.push(relPath);
  }
  return uniqueList(contextPaths);
}

function writeJson(filePath, payload) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), "utf8");
  return filePath;
}

export function resolveIsolationMode(mode = process.env.CODER_ISOLATION_MODE || "") {
  return safeMode(mode);
}

export function cleanupIsolationWorkspace(taskDir) {
  const isolatedWorkspaceRoot = path.join(taskDir, "isolated_workspace");
  try {
    if (fs.existsSync(isolatedWorkspaceRoot)) {
      fs.rmSync(isolatedWorkspaceRoot, { recursive: true, force: true });
    }
  } catch (err) {
    console.warn("[isolation_workspace] Cleanup failed for", isolatedWorkspaceRoot, ":", err.message);
  }
}

export function materializeIsolationWorkspace({
  workspaceRoot,
  taskDir,
  targetPaths = [],
  mode = process.env.CODER_ISOLATION_MODE || "",
}) {
  const resolvedMode = resolveIsolationMode(mode);
  const targetList = Array.isArray(targetPaths) ? targetPaths.map((item) => normalizeRelPath(item)).filter(Boolean) : [];
  const contextList = inferIsolationContextPaths(targetList);
  const isolationDir = path.join(taskDir, "isolation");
  const isolatedWorkspaceRoot = path.join(taskDir, "isolated_workspace");

  const executionModePayload = {
    generated_at: new Date().toISOString(),
    isolation_mode: resolvedMode,
    enabled: isEnabled(resolvedMode),
    isolated_workspace_root: isolatedWorkspaceRoot,
    target_paths: targetList,
    context_paths: contextList,
  };
  const executionModePath = writeJson(
    path.join(isolationDir, "execution_mode.json"),
    executionModePayload,
  );

  if (!isEnabled(resolvedMode)) {
    return {
      enabled: false,
      mode: resolvedMode,
      isolatedWorkspaceRoot: null,
      artifacts: {
        execution_mode: executionModePath,
        baseline_manifest: null,
        isolated_workspace_manifest: null,
      },
      diagnostics: executionModePayload,
    };
  }

  ensureDir(isolatedWorkspaceRoot);

  const baselineEntries = [];
  const isolatedEntries = [];
  const missingTargets = [];
  const copiedRoots = [];
  for (const relPath of contextList) {
    const srcAbs = path.resolve(workspaceRoot, relPath);
    if (!fs.existsSync(srcAbs)) {
      missingTargets.push(relPath);
      continue;
    }
    copiedRoots.push(relPath);
    listEntries(workspaceRoot, relPath, baselineEntries);
    copyRecursive(srcAbs, path.resolve(isolatedWorkspaceRoot, relPath));
    listEntries(isolatedWorkspaceRoot, relPath, isolatedEntries);
  }

  const baselineManifestPath = writeJson(
    path.join(isolationDir, "baseline_manifest.json"),
    {
      generated_at: new Date().toISOString(),
      workspace_root: workspaceRoot,
      target_paths: targetList,
      context_paths: contextList,
      copied_roots: copiedRoots,
      missing_targets: missingTargets,
      entries: baselineEntries,
    },
  );
  const isolatedManifestPath = writeJson(
    path.join(isolationDir, "isolated_workspace_manifest.json"),
    {
      generated_at: new Date().toISOString(),
      isolated_workspace_root: isolatedWorkspaceRoot,
      target_paths: targetList,
      context_paths: contextList,
      copied_roots: copiedRoots,
      missing_targets: missingTargets,
      entries: isolatedEntries,
    },
  );

  return {
    enabled: true,
    mode: resolvedMode,
    isolatedWorkspaceRoot,
    artifacts: {
      execution_mode: executionModePath,
      baseline_manifest: baselineManifestPath,
      isolated_workspace_manifest: isolatedManifestPath,
    },
    diagnostics: {
      ...executionModePayload,
      copied_roots: copiedRoots,
      missing_targets: missingTargets,
      baseline_entry_count: baselineEntries.length,
      isolated_entry_count: isolatedEntries.length,
    },
  };
}
