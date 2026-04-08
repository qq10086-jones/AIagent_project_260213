import fs from "fs";
import path from "path";
import crypto from "crypto";
import { exec } from "child_process";

import { normalizeRelPath } from "./scope_guard.js";
import { isCacheEnabled } from "../shared/feature_flags.js";

export function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

async function execFileCapture(command, args, cwd) {
  return new Promise((resolve) => {
    try {
      const child = exec(`"${command}" ${args.map((item) => `"${String(item).replace(/"/g, '\\"')}"`).join(" ")}`, { cwd, timeout: 30000 }, (error, stdout, stderr) => {
        resolve({
          ok: !error,
          stdout: String(stdout || ""),
          stderr: String(stderr || ""),
          exitCode: error?.code ?? 0,
        });
      });
      child.on("error", (err) => {
        resolve({
          ok: false,
          stdout: "",
          stderr: String(err?.message || err || ""),
          exitCode: 1,
        });
      });
    } catch (err) {
      resolve({
        ok: false,
        stdout: "",
        stderr: err.message,
        exitCode: 1,
      });
    }
  });
}

function flushStaticCheck(taskDir, records, error) {
  let logPath = null;
  try {
    logPath = path.join(taskDir, `static_check_${Date.now()}.json`);
    fs.writeFileSync(logPath, JSON.stringify({
      generated_at: new Date().toISOString(),
      ok: !error,
      records,
      error: error || null,
    }, null, 2), "utf8");
  } catch {
    /* ignore: failed to write static check log is not terminal */
  }
  return {
    checked: true,
    ok: !error,
    commands: records.map((r) => `${r.kind}:${r.file}`),
    records,
    error: error || null,
    logPath,
  };
}

function contentHash(content) {
  return crypto.createHash("sha256").update(content).digest("hex");
}

async function cacheGet(redis, key) {
  if (!redis || !isCacheEnabled()) return null;
  try {
    return await redis.get(key);
  } catch (err) {
    console.warn("[static_checks] cache lookup error, falling back:", err.message);
    return null;
  }
}

async function cacheSet(redis, key, value, ttl) {
  if (!redis || !isCacheEnabled()) return;
  try {
    await redis.set(key, value, "EX", ttl);
  } catch { /* ignore cache write errors */ }
}

export async function runStaticChecks({ workspaceRoot, filesChanged = [], taskDir, redis = null }) {
  const changed = Array.isArray(filesChanged) ? filesChanged.map((item) => normalizeRelPath(item)).filter(Boolean) : [];
  if (changed.length === 0) {
    return { checked: false, ok: true, commands: [], logPath: null };
  }
  const records = [];
  for (const rel of changed) {
    if (/(^|\/)node_modules(\/|$)/.test(rel)) continue;
    const abs = path.resolve(workspaceRoot, rel);
    if (!abs.startsWith(path.resolve(workspaceRoot)) || !fs.existsSync(abs)) continue;
    const ext = path.extname(rel).toLowerCase();

    const checkKind = [".js", ".mjs", ".cjs"].includes(ext) ? "node_syntax"
      : ext === ".json" ? "json_parse"
      : ext === ".py" ? "py_compile" : null;
    if (!checkKind) continue;

    // 读文件内容（复用于 hash + JSON parse）
    let fileContent = null;
    try {
      fileContent = fs.readFileSync(abs, "utf8");
    } catch {
      continue;
    }
    const hash = contentHash(fileContent);
    const cacheKey = `static_check:${checkKind}:${rel}:${hash}`;

    // Cache lookup
    const cached = await cacheGet(redis, cacheKey);
    if (cached === "pass") {
      records.push({ file: rel, kind: checkKind, ok: true, exit_code: 0, stderr: "", cached: true });
      continue;
    }

    // 执行原始检查
    if (checkKind === "node_syntax") {
      const proc = await execFileCapture("node", ["--check", abs], workspaceRoot);
      records.push({ file: rel, kind: "node_syntax", ok: proc.ok, exit_code: proc.exitCode, stderr: proc.stderr.trim() });
      if (!proc.ok) return flushStaticCheck(taskDir, records, "E_STATIC_CHECK_FAILED: node syntax check failed");
      await cacheSet(redis, cacheKey, "pass", 86400);
      continue;
    }
    if (checkKind === "json_parse") {
      try {
        JSON.parse(fileContent);
        records.push({ file: rel, kind: "json_parse", ok: true, exit_code: 0, stderr: "" });
        await cacheSet(redis, cacheKey, "pass", 86400);
      } catch (err) {
        records.push({ file: rel, kind: "json_parse", ok: false, exit_code: 1, stderr: err.message });
        return flushStaticCheck(taskDir, records, "E_STATIC_CHECK_FAILED: json parse failed");
      }
      continue;
    }
    if (checkKind === "py_compile") {
      const proc = await execFileCapture("python", ["-m", "py_compile", abs], workspaceRoot);
      records.push({ file: rel, kind: "py_compile", ok: proc.ok, exit_code: proc.exitCode, stderr: proc.stderr.trim() });
      if (!proc.ok) return flushStaticCheck(taskDir, records, "E_STATIC_CHECK_FAILED: python compile failed");
      await cacheSet(redis, cacheKey, "pass", 86400);
    }
  }
  return flushStaticCheck(taskDir, records, null);
}
