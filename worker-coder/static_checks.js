import fs from "fs";
import path from "path";
import { exec } from "child_process";

import { normalizeRelPath } from "./scope_guard.js";

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
        records.push({ file: rel, kind: "json_parse", ok: false, exit_code: 1, stderr: err.message });
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
