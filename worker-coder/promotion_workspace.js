import fs from "fs";
import path from "path";
import { captureScopedSnapshot } from "./scoped_delta.js";

import { normalizeRelPath, validateChangedFilesWithinScope } from "./scope_guard.js";

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, payload) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), "utf8");
  return filePath;
}

function copyFileWithParents(srcAbs, dstAbs) {
  ensureDir(path.dirname(dstAbs));
  fs.copyFileSync(srcAbs, dstAbs);
}

function backupExistingFile(workspaceRoot, backupRoot, relPath) {
  const srcAbs = path.resolve(workspaceRoot, relPath);
  if (!fs.existsSync(srcAbs)) return null;
  const backupAbs = path.resolve(backupRoot, relPath);
  copyFileWithParents(srcAbs, backupAbs);
  return backupAbs;
}

function restoreBackup(workspaceRoot, backupRoot, relPath) {
  const dstAbs = path.resolve(workspaceRoot, relPath);
  const backupAbs = path.resolve(backupRoot, relPath);
  if (fs.existsSync(backupAbs)) {
    copyFileWithParents(backupAbs, dstAbs);
    return;
  }
  if (fs.existsSync(dstAbs)) {
    fs.unlinkSync(dstAbs);
  }
}

function detectBaselineDrift(workspaceRoot, targetPaths, baselineSnapshot) {
  const currentSnapshot = captureScopedSnapshot(workspaceRoot, targetPaths);
  const driftedFiles = [];

  for (const [rel, currentMeta] of currentSnapshot.entries()) {
    const baseMeta = baselineSnapshot.get(rel);
    if (!baseMeta || baseMeta.hash !== currentMeta.hash) {
      driftedFiles.push(rel);
    }
  }
  
  for (const rel of baselineSnapshot.keys()) {
    if (!currentSnapshot.has(rel)) {
      driftedFiles.push(rel);
    }
  }
  
  return Array.from(new Set(driftedFiles)).sort();
}

export function promoteIsolatedChanges({
  workspaceRoot,
  isolatedWorkspaceRoot,
  taskDir,
  filesChanged = [],
  allowedTargetPaths = [],
  mode = "disabled",
  baselineSnapshot = new Map(),
}) {
  const normalizedMode = String(mode || "").trim().toLowerCase();
  const promotionDir = path.join(taskDir, "isolation");
  const changedFiles = Array.isArray(filesChanged)
    ? filesChanged.map((item) => normalizeRelPath(item)).filter(Boolean)
    : [];
    
  const driftedFiles = detectBaselineDrift(workspaceRoot, allowedTargetPaths, baselineSnapshot);
  const isDrifted = driftedFiles.length > 0;
  
  const preflight = {
    generated_at: new Date().toISOString(),
    mode: normalizedMode,
    changed_files: changedFiles,
    allowed_target_paths: Array.isArray(allowedTargetPaths) ? allowedTargetPaths : [],
    isolated_workspace_root: isolatedWorkspaceRoot,
    promotion_attempted: normalizedMode === "promote",
    baseline_drift_detected: isDrifted,
    drifted_files: driftedFiles,
  };

  const scopeCheck = validateChangedFilesWithinScope({
    filesChanged: changedFiles,
    allowedTargetPaths,
  });
  
  const preflightOk = scopeCheck.ok && !isDrifted;
  let preflightError = null;
  if (!scopeCheck.ok) preflightError = scopeCheck.error;
  else if (isDrifted) preflightError = `PROMOTION_CONFLICT: Workspace drifted since task start. Drifted files: ${driftedFiles.join(", ")}`;

  const preflightPath = writeJson(
    path.join(promotionDir, "promotion_preflight.json"),
    {
      ...preflight,
      ok: preflightOk,
      error: preflightError,
    },
  );
  
  if (!preflightOk) {
    const resultPath = writeJson(
      path.join(promotionDir, "promotion_result.json"),
      {
        ...preflight,
        ok: false,
        applied: false,
        rollback_performed: false,
        error: preflightError,
      },
    );
    return {
      ok: false,
      applied: false,
      rollbackPerformed: false,
      error: preflightError,
      artifacts: {
        promotion_preflight: preflightPath,
        promotion_result: resultPath,
      },
    };
  }

  if (normalizedMode !== "promote") {
    const resultPath = writeJson(
      path.join(promotionDir, "promotion_result.json"),
      {
        ...preflight,
        ok: true,
        applied: false,
        rollback_performed: false,
        status: "shadow_only",
      },
    );
    return {
      ok: true,
      applied: false,
      rollbackPerformed: false,
      error: null,
      artifacts: {
        promotion_preflight: preflightPath,
        promotion_result: resultPath,
      },
    };
  }

  const backupRoot = path.join(promotionDir, "promotion_backup");
  const applied = [];
  try {
    for (const relPath of changedFiles) {
      const srcAbs = path.resolve(isolatedWorkspaceRoot, relPath);
      const dstAbs = path.resolve(workspaceRoot, relPath);
      backupExistingFile(workspaceRoot, backupRoot, relPath);
      if (fs.existsSync(srcAbs)) {
        copyFileWithParents(srcAbs, dstAbs);
      } else if (fs.existsSync(dstAbs)) {
        fs.unlinkSync(dstAbs);
      }
      applied.push(relPath);
    }
    const resultPath = writeJson(
      path.join(promotionDir, "promotion_result.json"),
      {
        ...preflight,
        ok: true,
        applied: true,
        applied_files: applied,
        rollback_performed: false,
        status: "promoted",
      },
    );
    return {
      ok: true,
      applied: true,
      rollbackPerformed: false,
      error: null,
      artifacts: {
        promotion_preflight: preflightPath,
        promotion_result: resultPath,
      },
    };
  } catch (err) {
    for (const relPath of applied.reverse()) {
      restoreBackup(workspaceRoot, backupRoot, relPath);
    }
    const resultPath = writeJson(
      path.join(promotionDir, "promotion_result.json"),
      {
        ...preflight,
        ok: false,
        applied: false,
        applied_files: applied,
        rollback_performed: applied.length > 0,
        error: `PROMOTION_PARTIAL_ABORT: ${String(err?.message || err || "promotion failed")}`,
        status: "promotion_failed",
      },
    );
    return {
      ok: false,
      applied: false,
      rollbackPerformed: applied.length > 0,
      error: `PROMOTION_PARTIAL_ABORT: ${String(err?.message || err || "promotion failed")}`,
      artifacts: {
        promotion_preflight: preflightPath,
        promotion_result: resultPath,
      },
    };
  }
}
