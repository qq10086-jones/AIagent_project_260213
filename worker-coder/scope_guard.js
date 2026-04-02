import path from "path";

export function normalizeRelPath(value) {
  const normalized = String(value || "").replace(/\\/g, "/").replace(/^\/+/, "");
  if (normalized === "sandbox" || normalized.startsWith("sandbox/")) {
    return `workspace/${normalized}`;
  }
  if (normalized === "scratch" || normalized.startsWith("scratch/")) {
    return `workspace/${normalized}`;
  }
  return normalized;
}

export function isProtectedRoot(relPath) {
  const safe = normalizeRelPath(relPath).replace(/\/+$/, "");
  return [".git", "infra", "docker-compose.yml", "configs", "orchestrator", "worker-coder", "worker-quant"].some((item) => {
    if (!item.includes("/")) return safe === item || safe.startsWith(`${item}/`);
    return safe === item;
  });
}

export function validateAllowedTargetPaths({ workspaceRoot, allowedTargetPaths = [] }) {
  const workspaceAbs = path.resolve(workspaceRoot);
  const safeTargets = Array.isArray(allowedTargetPaths) ? allowedTargetPaths : [];
  if (safeTargets.length === 0) {
    return { ok: false, error: "E_UNAUTHORIZED_WRITE: target_paths required for write-capable coding task." };
  }
  for (const rel of safeTargets) {
    const normalized = normalizeRelPath(rel);
    if (!normalized) {
      return { ok: false, error: "E_UNAUTHORIZED_WRITE: empty target path is not allowed." };
    }
    if (isProtectedRoot(normalized)) {
      return { ok: false, error: `E_UNAUTHORIZED_WRITE: protected target path '${normalized}' is not allowed.` };
    }
    const abs = path.resolve(workspaceAbs, normalized);
    if (!abs.startsWith(workspaceAbs)) {
      return { ok: false, error: `E_UNAUTHORIZED_WRITE: target path '${normalized}' escapes workspace root.` };
    }
  }
  return { ok: true };
}

export function validateRequestedWrite({ workspaceRoot, targetPath, allowedTargetPaths = [] }) {
  const rootsCheck = validateAllowedTargetPaths({ workspaceRoot, allowedTargetPaths });
  if (!rootsCheck.ok) return rootsCheck;
  const normalized = normalizeRelPath(targetPath);
  const inScope = allowedTargetPaths
    .map((item) => normalizeRelPath(item).replace(/\/+$/, ""))
    .some((prefix) => normalized === prefix || normalized.startsWith(`${prefix}/`));
  if (!inScope) {
    return { ok: false, error: `E_UNAUTHORIZED_WRITE: '${normalized}' is outside allowed target_paths.` };
  }
  return { ok: true };
}

export function validateChangedFilesWithinScope({ filesChanged = [], allowedTargetPaths = [] }) {
  if (!Array.isArray(filesChanged) || filesChanged.length === 0) {
    return { ok: true };
  }
  const prefixes = (Array.isArray(allowedTargetPaths) ? allowedTargetPaths : [])
    .map((item) => normalizeRelPath(item).replace(/\/+$/, ""))
    .filter(Boolean);
  if (prefixes.length === 0) {
    return { ok: false, error: "E_UNAUTHORIZED_WRITE: changed files present but target_paths missing." };
  }
  const outOfScope = filesChanged
    .map((item) => normalizeRelPath(item))
    .filter((file) => !prefixes.some((prefix) => file === prefix || file.startsWith(`${prefix}/`)));
  if (outOfScope.length > 0) {
    return {
      ok: false,
      error: `E_UNAUTHORIZED_WRITE: changed files outside scope: ${outOfScope.join(", ")}`,
    };
  }
  return { ok: true };
}
