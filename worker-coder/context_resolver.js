/**
 * context_resolver.js — Phase 2 (P2-1): Structured Context Pipeline.
 *
 * Given target_paths and task_class, auto-discovers structurally related files
 * via regex-based import graph analysis (JS/TS/Python). Implements v4.2 Layer E
 * ContextRequest → ContextResponse interface (structure layer only).
 *
 * Design: v3.2 Section 4.3 (P2-1)
 * Feature flag: context_resolver_enabled (default false)
 */
import fs from "fs";
import path from "path";

// --- Import regex patterns (no tree-sitter dependency) ---

// JS/TS: import ... from "path", import("path"), require("path")
const JS_IMPORT_RE = /(?:import\s+(?:[\s\S]*?\s+from\s+)?['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\)|import\s*\(\s*['"]([^'"]+)['"]\s*\))/g;

// Python: import module, from module import ...
const PY_IMPORT_RE = /(?:^|\n)\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))/g;

const JS_EXTENSIONS = [".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"];
const PY_EXTENSIONS = [".py"];

/**
 * Detect file language from extension.
 * @param {string} filePath
 * @returns {"js"|"python"|null}
 */
function detectLang(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (JS_EXTENSIONS.includes(ext)) return "js";
  if (PY_EXTENSIONS.includes(ext)) return "python";
  return null;
}

/**
 * Extract raw import specifiers from file content.
 * @param {string} content
 * @param {"js"|"python"} lang
 * @returns {string[]}
 */
function extractImports(content, lang) {
  const imports = [];
  if (lang === "js") {
    let m;
    JS_IMPORT_RE.lastIndex = 0;
    while ((m = JS_IMPORT_RE.exec(content)) !== null) {
      const spec = m[1] || m[2] || m[3];
      if (spec) imports.push(spec);
    }
  } else if (lang === "python") {
    let m;
    PY_IMPORT_RE.lastIndex = 0;
    while ((m = PY_IMPORT_RE.exec(content)) !== null) {
      const spec = m[1] || m[2];
      if (spec) imports.push(spec);
    }
  }
  return imports;
}

/**
 * Resolve a JS/TS import specifier to an absolute file path.
 * Only resolves relative imports (./xxx, ../xxx). Node_modules are ignored.
 * @param {string} specifier
 * @param {string} fromFile - absolute path of the importing file
 * @param {string} workspaceRoot
 * @returns {string|null} - absolute path or null
 */
function resolveJsImport(specifier, fromFile, workspaceRoot) {
  if (!specifier.startsWith(".")) return null; // skip node_modules
  const dir = path.dirname(fromFile);
  const base = path.resolve(dir, specifier);

  // Try exact, then with extensions, then index files
  const candidates = [
    base,
    ...JS_EXTENSIONS.map((ext) => base + ext),
    ...JS_EXTENSIONS.map((ext) => path.join(base, "index" + ext)),
  ];

  for (const candidate of candidates) {
    if (candidate.startsWith(workspaceRoot) && fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate;
    }
  }
  return null;
}

/**
 * Resolve a Python import module name to an absolute file path.
 * Converts dotted path to directory path relative to workspace.
 * @param {string} moduleName
 * @param {string} fromFile
 * @param {string} workspaceRoot
 * @returns {string|null}
 */
function resolvePyImport(moduleName, fromFile, workspaceRoot) {
  const parts = moduleName.split(".");
  const dir = path.dirname(fromFile);

  // Try relative to the importing file's directory first
  const relCandidates = [
    path.join(dir, ...parts) + ".py",
    path.join(dir, ...parts, "__init__.py"),
  ];
  for (const candidate of relCandidates) {
    if (candidate.startsWith(workspaceRoot) && fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate;
    }
  }

  // Try relative to workspace root
  const rootCandidates = [
    path.join(workspaceRoot, ...parts) + ".py",
    path.join(workspaceRoot, ...parts, "__init__.py"),
  ];
  for (const candidate of rootCandidates) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate;
    }
  }

  return null;
}

/**
 * Estimate token count for file content (rough: ~4 chars per token).
 * @param {string} content
 * @returns {number}
 */
function estimateTokens(content) {
  return Math.ceil((content || "").length / 4);
}

/**
 * Resolve context for given target paths by walking the import graph.
 *
 * @param {Object} request - v4.2 ContextRequest
 * @param {string} request.task_class
 * @param {string[]} request.target_paths - relative paths within workspace
 * @param {number} [request.max_files=20]
 * @param {number} [request.max_tokens=8000]
 * @param {number} [request.dependency_depth=2]
 * @param {string} workspaceRoot
 * @returns {Object} v4.2 ContextResponse
 */
export function resolveContext(request, workspaceRoot) {
  const startTime = Date.now();
  const maxFiles = Math.max(1, Math.trunc(Number.isFinite(Number(request.max_files)) ? Number(request.max_files) : 20));
  const maxTokens = Math.max(100, Math.trunc(Number.isFinite(Number(request.max_tokens)) ? Number(request.max_tokens) : 8000));
  const dependencyDepth = Math.max(0, Math.trunc(Number.isFinite(Number(request.dependency_depth)) ? Number(request.dependency_depth) : 2));
  const targetPaths = Array.isArray(request.target_paths) ? request.target_paths : [];

  const absRoot = path.resolve(workspaceRoot);
  const files = [];
  const seen = new Set();
  const missingContext = [];
  let totalTokens = 0;
  let successCount = 0;
  let failCount = 0;

  // Seed: collect actual files from target_paths
  const seedFiles = [];
  for (const tp of targetPaths) {
    const absPath = path.resolve(absRoot, tp);
    if (!absPath.startsWith(absRoot)) continue;
    if (!fs.existsSync(absPath)) {
      missingContext.push(tp);
      continue;
    }
    const stat = fs.statSync(absPath);
    if (stat.isFile()) {
      seedFiles.push(absPath);
    } else if (stat.isDirectory()) {
      // Scan top-level files in directory
      try {
        const entries = fs.readdirSync(absPath, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isFile() && detectLang(entry.name)) {
            seedFiles.push(path.join(absPath, entry.name));
          }
        }
      } catch { /* ignore permission errors */ }
    }
  }

  // BFS import graph traversal
  const queue = seedFiles.map((f) => ({ absPath: f, depth: 0, role: "target" }));

  while (queue.length > 0 && files.length < maxFiles && totalTokens < maxTokens) {
    const { absPath, depth, role } = queue.shift();
    if (seen.has(absPath)) continue;
    seen.add(absPath);

    const lang = detectLang(absPath);
    if (!lang) continue;

    let content;
    try {
      content = fs.readFileSync(absPath, "utf8");
    } catch {
      failCount++;
      missingContext.push(path.relative(absRoot, absPath));
      continue;
    }

    const tokens = estimateTokens(content);
    if (totalTokens + tokens > maxTokens && files.length > 0) {
      // Would exceed budget — stop adding files but continue to record missing
      break;
    }

    const relPath = path.relative(absRoot, absPath).replace(/\\/g, "/");
    files.push({
      path: relPath,
      role: depth === 0 ? role : "dependency",
      token_estimate: tokens,
    });
    totalTokens += tokens;
    successCount++;

    // Recurse into imports if within depth budget
    if (depth < dependencyDepth) {
      const importSpecifiers = extractImports(content, lang);
      for (const spec of importSpecifiers) {
        const resolved = lang === "js"
          ? resolveJsImport(spec, absPath, absRoot)
          : resolvePyImport(spec, absPath, absRoot);
        if (resolved && !seen.has(resolved)) {
          queue.push({ absPath: resolved, depth: depth + 1, role: "dependency" });
        }
      }
    }
  }

  const resolutionTimeMs = Date.now() - startTime;
  const confidence = successCount + failCount > 0
    ? successCount / (successCount + failCount)
    : 0;

  return {
    status: files.length >= maxFiles || totalTokens >= maxTokens ? "partial" : "complete",
    files,
    token_usage: totalTokens,
    missing_context: missingContext,
    confidence: Math.round(confidence * 100) / 100,
    resolution_method: "import_graph",
    resolution_time_ms: resolutionTimeMs,
  };
}

/**
 * Build a prompt-injectable context block from ContextResponse files.
 * Reads actual file content and formats for prompt injection.
 * @param {Object} contextResponse
 * @param {string} workspaceRoot
 * @param {number} [maxBytes=32768]
 * @returns {string}
 */
export function buildContextBlock(contextResponse, workspaceRoot, maxBytes = 32768) {
  if (!contextResponse || !Array.isArray(contextResponse.files) || contextResponse.files.length === 0) {
    return "";
  }
  const absRoot = path.resolve(workspaceRoot);
  const lines = ["", "[Auto-Resolved Context (import_graph)]"];
  let totalBytes = 0;

  for (const file of contextResponse.files) {
    if (totalBytes >= maxBytes) break;
    const absPath = path.resolve(absRoot, file.path);
    try {
      const content = fs.readFileSync(absPath, "utf8");
      const truncated = content.slice(0, Math.min(content.length, maxBytes - totalBytes));
      lines.push(`--- ${file.path} (${file.role}) ---`);
      lines.push(truncated);
      lines.push("");
      totalBytes += truncated.length;
    } catch { /* skip unreadable */ }
  }

  return lines.join("\n");
}
