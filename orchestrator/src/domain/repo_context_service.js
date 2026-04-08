import fs from "fs";
import path from "path";
import crypto from "crypto";
import { exec } from "child_process";

import { isCacheEnabled } from "../../../shared/feature_flags.js";

const TEXT_EXTENSIONS = new Set([
  ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".txt", ".py", ".rb", ".java", ".go", ".rs", ".css", ".scss", ".html", ".yml", ".yaml"
]);

const ENTRYPOINT_PATTERNS = [
  /(^|\/)(index|main|app|server)\.(js|ts|jsx|tsx|py)$/i,
  /(^|\/)package\.json$/i,
  /(^|\/)tsconfig\.json$/i,
  /(^|\/)vite\.config\.(js|ts)$/i,
];

const TEST_PATTERNS = [
  /\.test\.(js|jsx|ts|tsx|py)$/i,
  /\.spec\.(js|jsx|ts|tsx|py)$/i,
  /(^|\/)__tests__\//i,
  /(^|\/)tests?\//i,
];

function normalizeRel(relPath = "") {
  return String(relPath || "").replace(/\\/g, "/").replace(/^\/+/, "");
}

function isIgnoredDir(name = "") {
  return [".git", "node_modules", "dist", "build", ".next", ".turbo", "coverage"].includes(String(name || ""));
}

function safeReadText(filePath, maxChars = 2000) {
  try {
    return fs.readFileSync(filePath, "utf8").slice(0, maxChars);
  } catch {
    return "";
  }
}

function detectToolchainFiles(workspaceRoot) {
  const candidates = [
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "tsconfig.json",
    "pyproject.toml",
    "requirements.txt",
    "ruff.toml",
    ".eslintrc",
    ".eslintrc.json",
    "eslint.config.js",
    "eslint.config.mjs",
  ];
  return candidates.filter((rel) => fs.existsSync(path.join(workspaceRoot, rel)));
}

function extractToolchainFacts(workspaceRoot) {
  const facts = [];
  for (const rel of detectToolchainFiles(workspaceRoot)) {
    const abs = path.join(workspaceRoot, rel);
    if (rel === "package.json") {
      try {
        const pkg = JSON.parse(fs.readFileSync(abs, "utf8"));
        facts.push({
          file: rel,
          type: "package.json",
          package_name: String(pkg?.name || ""),
          scripts: Object.keys(pkg?.scripts || {}),
        });
        continue;
      } catch {
        // fall through to raw snippet
      }
    }
    facts.push({
      file: rel,
      type: "config",
      snippet: safeReadText(abs, 400),
    });
  }
  return facts;
}

function collectDirectorySummary(absPath, rootAbs, maxEntries = 20) {
  const entries = [];
  try {
    for (const entry of fs.readdirSync(absPath, { withFileTypes: true })) {
      if (isIgnoredDir(entry.name)) continue;
      entries.push({
        path: normalizeRel(path.relative(rootAbs, path.join(absPath, entry.name))),
        kind: entry.isDirectory() ? "dir" : "file",
      });
      if (entries.length >= maxEntries) break;
    }
  } catch {
    return [];
  }
  return entries;
}

function collectFilesUnder(absPath, rootAbs, limit = 60, bucket = []) {
  if (bucket.length >= limit) return bucket;
  let entries = [];
  try {
    entries = fs.readdirSync(absPath, { withFileTypes: true });
  } catch {
    return bucket;
  }
  for (const entry of entries) {
    if (bucket.length >= limit) break;
    if (isIgnoredDir(entry.name)) continue;
    const childAbs = path.join(absPath, entry.name);
    if (entry.isDirectory()) {
      collectFilesUnder(childAbs, rootAbs, limit, bucket);
      continue;
    }
    const rel = normalizeRel(path.relative(rootAbs, childAbs));
    const ext = path.extname(rel).toLowerCase();
    if (!TEXT_EXTENSIONS.has(ext) && !TEST_PATTERNS.some((pattern) => pattern.test(rel))) continue;
    bucket.push(rel);
  }
  return bucket;
}

function inferEntrypoints(files = []) {
  return files.filter((rel) => ENTRYPOINT_PATTERNS.some((pattern) => pattern.test(rel))).slice(0, 12);
}

function inferTests(files = []) {
  return files.filter((rel) => TEST_PATTERNS.some((pattern) => pattern.test(rel))).slice(0, 12);
}

function extractSymbolHints(workspaceRoot, files = []) {
  const hints = [];
  for (const rel of files.slice(0, 8)) {
    const abs = path.join(workspaceRoot, rel);
    const raw = safeReadText(abs, 2400);
    if (!raw) continue;
    const matches = raw.match(/(?:export\s+(?:class|function|const)\s+\w+|class\s+\w+|function\s+\w+|const\s+\w+\s*=|module\.exports\s*=)/g) || [];
    hints.push({
      file: rel,
      symbols: matches.slice(0, 10),
    });
  }
  return hints;
}

async function getGitHeadSha(cwd) {
  return new Promise((resolve) => {
    exec("git rev-parse HEAD", { cwd, timeout: 5000 }, (error, stdout) => {
      resolve(error ? null : String(stdout || "").trim());
    });
  });
}

function makeCacheKey(workspaceAbs, sha, targetPaths) {
  const raw = `${workspaceAbs}:${sha}:${JSON.stringify(targetPaths || [])}`;
  return `repo_map:${crypto.createHash("sha256").update(raw).digest("hex")}`;
}

export function createRepoContextService({ workspaceRoot = ".", redis = null } = {}) {
  const workspaceAbs = path.resolve(workspaceRoot);

  function resolveScopedPaths(targetPaths = []) {
    const safe = [];
    for (const targetPath of Array.isArray(targetPaths) ? targetPaths : []) {
      const rel = normalizeRel(targetPath);
      if (!rel) continue;
      const abs = path.resolve(workspaceAbs, rel);
      if (!abs.startsWith(workspaceAbs) || !fs.existsSync(abs)) continue;
      safe.push({ rel, abs, kind: fs.statSync(abs).isDirectory() ? "dir" : "file" });
    }
    return safe;
  }

  function buildRepoMap({ targetPaths = [], recentChangedFiles = [] } = {}) {
    const scoped = resolveScopedPaths(targetPaths);
    const repoMap = {
      workspace_root: workspaceAbs.replace(/\\/g, "/"),
      target_paths: scoped.map((item) => item.rel),
      directory_summary: [],
      candidate_files: [],
      entrypoints: [],
      related_tests: [],
      key_config_files: detectToolchainFiles(workspaceAbs),
      toolchain_facts: extractToolchainFacts(workspaceAbs),
      symbol_hints: [],
      recent_changed_files: (Array.isArray(recentChangedFiles) ? recentChangedFiles : []).map((item) => normalizeRel(item)).filter(Boolean).slice(0, 20),
    };

    const candidateSet = new Set();
    for (const item of scoped) {
      if (item.kind === "dir") {
        repoMap.directory_summary.push({
          path: item.rel,
          entries: collectDirectorySummary(item.abs, workspaceAbs),
        });
        for (const file of collectFilesUnder(item.abs, workspaceAbs, 60, [])) candidateSet.add(file);
      } else {
        candidateSet.add(item.rel);
        repoMap.directory_summary.push({
          path: item.rel,
          entries: [{ path: item.rel, kind: "file" }],
        });
      }
    }

    repoMap.candidate_files = Array.from(candidateSet).slice(0, 60);
    repoMap.entrypoints = inferEntrypoints(repoMap.candidate_files);
    repoMap.related_tests = inferTests(repoMap.candidate_files);
    repoMap.symbol_hints = extractSymbolHints(workspaceAbs, repoMap.candidate_files);
    return repoMap;
  }

  function buildContextPacket({
    role = "",
    stepId = "",
    targetPaths = [],
    recentChangedFiles = [],
    memoryHints = [],
  } = {}) {
    const repoMap = buildRepoMap({ targetPaths, recentChangedFiles });
    return {
      step_id: String(stepId || ""),
      role: String(role || ""),
      target_paths: Array.isArray(targetPaths) ? targetPaths.map((item) => normalizeRel(item)).filter(Boolean) : [],
      candidate_files: repoMap.candidate_files,
      entrypoints: repoMap.entrypoints,
      related_tests: repoMap.related_tests,
      toolchain_facts: repoMap.toolchain_facts,
      recent_changed_files: repoMap.recent_changed_files,
      memory_hints: Array.isArray(memoryHints) ? memoryHints.filter(Boolean).slice(0, 10) : [],
    };
  }

  async function buildRepoMapCached(opts = {}) {
    if (!redis || !isCacheEnabled()) return buildRepoMap(opts);
    try {
      const sha = await getGitHeadSha(workspaceAbs);
      if (!sha) return buildRepoMap(opts);
      const key = makeCacheKey(workspaceAbs, sha, opts.targetPaths);
      const cached = await redis.get(key);
      if (cached) {
        try { return JSON.parse(cached); } catch { /* fall through */ }
      }
      const result = buildRepoMap(opts);
      try { await redis.set(key, JSON.stringify(result), "EX", 3600); } catch { /* ignore */ }
      return result;
    } catch (err) {
      console.warn("[repo_context] cache error, falling back:", err.message);
      return buildRepoMap(opts);
    }
  }

  async function buildContextPacketCached(opts = {}) {
    if (!redis || !isCacheEnabled()) return buildContextPacket(opts);
    try {
      const sha = await getGitHeadSha(workspaceAbs);
      if (!sha) return buildContextPacket(opts);
      const raw = `${workspaceAbs}:ctx:${sha}:${JSON.stringify(opts.targetPaths || [])}:${String(opts.stepId || "")}`;
      const key = `context_packet:${crypto.createHash("sha256").update(raw).digest("hex")}`;
      const cached = await redis.get(key);
      if (cached) {
        try { return JSON.parse(cached); } catch { /* fall through */ }
      }
      const result = buildContextPacket(opts);
      try { await redis.set(key, JSON.stringify(result), "EX", 3600); } catch { /* ignore */ }
      return result;
    } catch (err) {
      console.warn("[repo_context] cache error, falling back:", err.message);
      return buildContextPacket(opts);
    }
  }

  return {
    buildRepoMap: buildRepoMapCached,
    buildContextPacket: buildContextPacketCached,
  };
}
