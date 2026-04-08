/**
 * native_file_tools.js — Read-only file access for worker-coder.
 * Phase 1 (P1-3): Provides safe, scoped file reading without write capability.
 * All reads go through scope_guard path validation and content redaction.
 */
import fs from "fs";
import path from "path";
import { normalizeRelPath, isProtectedRoot } from "./scope_guard.js";
import { redactSensitiveText } from "./verification_runner.js";

/**
 * Read a file within workspace with path traversal protection and content redaction.
 * @param {{ workspaceRoot: string, relPath: string, maxBytes?: number }} opts
 * @returns {{ ok: boolean, content?: string, truncated?: boolean, bytes_read?: number, error?: string }}
 */
export function readWorkspaceFile({ workspaceRoot, relPath, maxBytes = 32768 }) {
  try {
    const normalized = normalizeRelPath(relPath);
    if (!normalized) return { ok: false, error: "empty relative path" };

    if (isProtectedRoot(normalized)) {
      return { ok: false, error: `protected root: ${normalized}` };
    }

    const workspaceAbs = path.resolve(workspaceRoot);
    const absPath = path.resolve(workspaceAbs, normalized);

    if (!absPath.startsWith(workspaceAbs + path.sep) && absPath !== workspaceAbs) {
      return { ok: false, error: "path traversal blocked" };
    }

    if (!fs.existsSync(absPath)) {
      return { ok: false, error: "file not found" };
    }

    const stat = fs.statSync(absPath);
    if (!stat.isFile()) return { ok: false, error: "not a file" };

    const raw = fs.readFileSync(absPath, "utf8");
    const truncated = raw.length > maxBytes;
    const content = redactSensitiveText(truncated ? raw.slice(0, maxBytes) : raw);

    return { ok: true, content, truncated, bytes_read: Math.min(raw.length, maxBytes) };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

/**
 * List directory entries within workspace, scoped.
 * @param {{ workspaceRoot: string, relPath: string, maxEntries?: number }} opts
 * @returns {{ ok: boolean, entries?: Array<{name: string, type: string, size: number}>, truncated?: boolean, error?: string }}
 */
export function listWorkspaceDir({ workspaceRoot, relPath, maxEntries = 200 }) {
  try {
    const normalized = normalizeRelPath(relPath || ".");
    if (isProtectedRoot(normalized)) {
      return { ok: false, error: `protected root: ${normalized}` };
    }

    const workspaceAbs = path.resolve(workspaceRoot);
    const absPath = path.resolve(workspaceAbs, normalized);

    if (!absPath.startsWith(workspaceAbs + path.sep) && absPath !== workspaceAbs) {
      return { ok: false, error: "path traversal blocked" };
    }

    if (!fs.existsSync(absPath) || !fs.statSync(absPath).isDirectory()) {
      return { ok: false, error: "not a directory" };
    }

    const dirents = fs.readdirSync(absPath, { withFileTypes: true });
    const entries = [];
    for (const d of dirents) {
      if (entries.length >= maxEntries) break;
      const type = d.isFile() ? "file" : d.isDirectory() ? "directory" : "other";
      let size = 0;
      if (d.isFile()) {
        try { size = fs.statSync(path.join(absPath, d.name)).size; } catch { /* skip */ }
      }
      entries.push({ name: d.name, type, size });
    }

    return { ok: true, entries, truncated: dirents.length > maxEntries };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

/**
 * Read a file only if it falls within allowedTargetPaths scope.
 * @param {{ workspaceRoot: string, relPath: string, allowedTargetPaths: string[], maxBytes?: number }} opts
 * @returns {{ ok: boolean, content?: string, truncated?: boolean, bytes_read?: number, error?: string }}
 */
export function readScopedFile({ workspaceRoot, relPath, allowedTargetPaths = [], maxBytes = 32768 }) {
  const normalized = normalizeRelPath(relPath);
  if (!normalized) return { ok: false, error: "empty relative path" };

  const prefixes = (Array.isArray(allowedTargetPaths) ? allowedTargetPaths : [])
    .map((p) => normalizeRelPath(p).replace(/\/+$/, ""))
    .filter(Boolean);

  if (prefixes.length === 0) {
    return { ok: false, error: "no target_paths specified for scoped read" };
  }

  const inScope = prefixes.some((prefix) => normalized === prefix || normalized.startsWith(`${prefix}/`));
  if (!inScope) {
    return { ok: false, error: `out of scope: ${normalized}` };
  }

  return readWorkspaceFile({ workspaceRoot, relPath, maxBytes });
}
