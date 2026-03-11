import fs from "fs";
import path from "path";
import { exec } from "child_process";

import { normalizeRelPath } from "./scope_guard.js";
import { runInlineNodeSyntaxCheck } from "./verification_runner.js";

async function execFileCapture(command, args, cwd) {
  if (String(command || "").trim() === "node" && Array.isArray(args) && args[0] === "--check" && args[1]) {
    return runInlineNodeSyntaxCheck(String(args[1]));
  }
  return new Promise((resolve) => {
    let child = null;
    try {
      child = exec(`"${command}" ${args.map((item) => `"${String(item).replace(/"/g, '\\"')}"`).join(" ")}`, { cwd, timeout: 30000 }, (error, stdout, stderr) => {
        resolve({
          ok: !error,
          stdout: String(stdout || ""),
          stderr: String(stderr || ""),
          exitCode: error?.code ?? 0,
        });
      });
    } catch (err) {
      resolve({
        ok: false,
        stdout: "",
        stderr: String(err?.message || err || ""),
        exitCode: null,
      });
      return;
    }
    child.on("error", (err) => {
      resolve({
        ok: false,
        stdout: "",
        stderr: String(err?.message || err || ""),
        exitCode: null,
      });
    });
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
    logPath = null;
  }
  return {
    checked: records.length > 0,
    ok: !error,
    commands: records.map((item) => `${item.kind}:${item.file}`),
    records,
    error: error || null,
    logPath,
  };
}

export async function runStaticChecks({ workspaceRoot, filesChanged = [], taskDir }) {
  const changed = Array.isArray(filesChanged) ? filesChanged.map((item) => normalizeRelPath(item)).filter(Boolean) : [];
  if (changed.length === 0) {
    return { checked: false, ok: true, commands: [], logPath: null };
  }
  const records = [];
  for (const rel of changed) {
    const abs = path.resolve(workspaceRoot, rel);
    if (!abs.startsWith(path.resolve(workspaceRoot)) || !fs.existsSync(abs)) continue;
    const ext = path.extname(rel).toLowerCase();
    if ([".js", ".mjs", ".cjs"].includes(ext)) {
      const proc = await execFileCapture("node", ["--check", abs], workspaceRoot);
      records.push({ file: rel, kind: "node_syntax", ok: proc.ok, exit_code: proc.exitCode, stderr: proc.stderr.trim() });
      if (!proc.ok) return flushStaticCheck(taskDir, records, "E_STATIC_CHECK_FAILED: node syntax check failed");
      continue;
    }
    if (ext === ".json") {
      try {
        JSON.parse(fs.readFileSync(abs, "utf8"));
        records.push({ file: rel, kind: "json_parse", ok: true, exit_code: 0, stderr: "" });
      } catch (err) {
        records.push({ file: rel, kind: "json_parse", ok: false, exit_code: 1, stderr: String(err?.message || err || "") });
        return flushStaticCheck(taskDir, records, "E_STATIC_CHECK_FAILED: json parse failed");
      }
      continue;
    }
    if (ext === ".py") {
      const proc = await execFileCapture("python", ["-m", "py_compile", abs], workspaceRoot);
      records.push({ file: rel, kind: "py_compile", ok: proc.ok, exit_code: proc.exitCode, stderr: proc.stderr.trim() });
      if (!proc.ok) return flushStaticCheck(taskDir, records, "E_STATIC_CHECK_FAILED: python compile failed");
    }
  }
  return flushStaticCheck(taskDir, records, null);
}

export function clampInt(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}
