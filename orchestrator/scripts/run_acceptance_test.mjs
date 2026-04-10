/**
 * run_acceptance_test.mjs
 *
 * v3.3 Patch B — Deterministic acceptance test runner.
 * Reads plan/acceptance.json, executes verify_command for each deterministic criterion,
 * and writes acceptance_result.json alongside smoke_result.json.
 *
 * Usage:
 *   node run_acceptance_test.mjs --artifact-root <path> [--port <port>]
 *
 * Accepts both legacy string[] criteria and structured object[] criteria.
 */

import fs from "fs";
import path from "path";
import { exec } from "child_process";

// 安全白名单 — 与 verification_runner.js 对齐
const ALLOWED_CMD_PREFIXES = new Set([
  "python", "pytest", "npm", "node", "git", "ls", "cat", "echo", "pwd", "grep", "rg", "fd", "ruff", "black", "curl",
]);

const BLOCKED_SUBCOMMANDS = new Map([
  ["git", new Set(["clean", "push", "reset", "remote", "config", "rebase", "merge", "checkout", "restore"])],
  ["npm", new Set(["publish", "unpublish", "owner", "access", "token", "login", "logout"])],
  ["node", new Set(["-e", "--eval"])],
  ["python", new Set(["-c"])],
]);

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key.startsWith("--")) continue;
    out[key.slice(2)] = value;
    i += 1;
  }
  return out;
}

function readJson(absPath) {
  try {
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

function validateSafeCommand(command) {
  const raw = String(command || "").trim();
  if (!raw) return { ok: false, error: "empty command" };
  if (/[;|><`$(){}[\]\\*?~\n\r]/.test(raw)) {
    return { ok: false, error: "forbidden shell meta-character" };
  }
  if (/(^|[^&])&([^&]|$)/.test(raw)) {
    return { ok: false, error: "unsupported & operator" };
  }
  const segments = raw.split("&&").map((s) => s.trim()).filter(Boolean);
  for (const segment of segments) {
    const tokens = segment.split(/\s+/);
    const prefix = tokens[0];
    if (!ALLOWED_CMD_PREFIXES.has(prefix)) {
      return { ok: false, error: `'${prefix}' not whitelisted` };
    }
    const blocked = BLOCKED_SUBCOMMANDS.get(prefix);
    if (blocked && tokens.length > 1 && blocked.has(tokens[1])) {
      return { ok: false, error: `'${prefix} ${tokens[1]}' restricted` };
    }
  }
  return { ok: true };
}

function execCapture(command, cwd, timeoutMs = 20000) {
  return new Promise((resolve) => {
    exec(command, { cwd, timeout: timeoutMs }, (error, stdout, stderr) => {
      resolve({
        ok: !error,
        stdout: String(stdout || ""),
        stderr: String(stderr || ""),
        code: error ? (error.code || 1) : 0,
      });
    });
  });
}

function normalizeCriteria(acceptance) {
  const raw = Array.isArray(acceptance?.criteria) ? acceptance.criteria : [];
  return raw.map((item, i) => {
    if (typeof item === "string") {
      return {
        id: `AC-${i + 1}`,
        description: item.trim(),
        verify_command: "",
        expected_pattern: "",
        verify_tier: "semantic",
      };
    }
    if (item && typeof item === "object") {
      return {
        id: String(item.id || `AC-${i + 1}`).trim(),
        description: String(item.description || "").trim(),
        verify_command: String(item.verify_command || "").trim(),
        expected_pattern: String(item.expected_pattern || "").trim(),
        verify_tier: String(item.verify_tier || (item.verify_command ? "deterministic" : "semantic")).trim(),
      };
    }
    return null;
  }).filter(Boolean);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const artifactRoot = String(args["artifact-root"] || "").trim();
  const workspaceRoot = String(args["workspace-root"] || "").trim();
  const rootAbs = path.resolve(process.cwd(), artifactRoot);
  // 命令执行目录: 优先 --workspace-root, 否则 cwd (即 workspace 根)
  const execRoot = workspaceRoot ? path.resolve(process.cwd(), workspaceRoot) : process.cwd();
  const acceptancePath = path.join(rootAbs, "plan", "acceptance.json");
  const verifyDir = path.join(rootAbs, "verify");
  fs.mkdirSync(verifyDir, { recursive: true });
  const resultPath = path.join(verifyDir, "acceptance_result.json");

  const result = {
    criteria_results: [],
    deterministic_pass: 0,
    deterministic_fail: 0,
    semantic_pending: 0,
    verdict: "no_deterministic_criteria",
    generated_at: new Date().toISOString(),
  };

  const acceptance = readJson(acceptancePath);
  if (!acceptance) {
    // fail-closed: acceptance.json 缺失视为失败，不静默放行
    result.verdict = "fail";
    result.runner_error = "acceptance.json not found or unreadable";
    fs.writeFileSync(resultPath, JSON.stringify(result, null, 2));
    process.exit(1);
  }

  const criteria = normalizeCriteria(acceptance);
  let detPass = 0;
  let detFail = 0;
  let semPending = 0;

  for (const c of criteria) {
    const entry = {
      id: c.id,
      description: c.description,
      verify_tier: c.verify_tier,
      verify_command: c.verify_command,
      expected_pattern: c.expected_pattern,
      status: "skip",
      stdout: "",
      stderr: "",
      error: "",
      evidence: "",
    };

    if (c.verify_tier !== "deterministic" || !c.verify_command) {
      entry.status = "skip";
      entry.evidence = "no verify_command, deferred to semantic QA";
      semPending += 1;
      result.criteria_results.push(entry);
      continue;
    }

    const safeCheck = validateSafeCommand(c.verify_command);
    if (!safeCheck.ok) {
      entry.status = "error";
      entry.error = `command blocked: ${safeCheck.error}`;
      detFail += 1;
      result.criteria_results.push(entry);
      continue;
    }

    // 从 workspace root 执行，确保 npm test 等命令能找到 package.json
    const execCwd = execRoot;
    const proc = await execCapture(c.verify_command, execCwd);
    entry.stdout = proc.stdout.slice(0, 2000);
    entry.stderr = proc.stderr.slice(0, 1000);

    if (!proc.ok) {
      entry.status = "fail";
      entry.error = `exit code ${proc.code}`;
      detFail += 1;
    } else if (c.expected_pattern) {
      try {
        const re = new RegExp(c.expected_pattern, "i");
        if (re.test(proc.stdout)) {
          entry.status = "pass";
          entry.evidence = `output matched /${c.expected_pattern}/i`;
          detPass += 1;
        } else {
          entry.status = "fail";
          entry.error = `output did not match /${c.expected_pattern}/i`;
          detFail += 1;
        }
      } catch (regexErr) {
        // fail-closed: 无效正则 → error，不能静默放行
        entry.status = "error";
        entry.error = `invalid expected_pattern regex: ${regexErr?.message || regexErr}`;
        detFail += 1;
      }
    } else {
      entry.status = "pass";
      entry.evidence = "command exited 0";
      detPass += 1;
    }

    result.criteria_results.push(entry);
  }

  result.deterministic_pass = detPass;
  result.deterministic_fail = detFail;
  result.semantic_pending = semPending;

  if (detPass + detFail === 0) {
    result.verdict = "no_deterministic_criteria";
  } else {
    result.verdict = detFail === 0 ? "pass" : "fail";
  }

  fs.writeFileSync(resultPath, JSON.stringify(result, null, 2));
  process.exit(0);
}

main().catch((error) => {
  const args = parseArgs(process.argv.slice(2));
  const artifactRoot = String(args["artifact-root"] || "").trim();
  const rootAbs = path.resolve(process.cwd(), artifactRoot);
  const verifyDir = path.join(rootAbs, "verify");
  fs.mkdirSync(verifyDir, { recursive: true });
  // fail-closed: runner 内部异常 → verdict fail + exit 1
  fs.writeFileSync(path.join(verifyDir, "acceptance_result.json"), JSON.stringify({
    criteria_results: [],
    deterministic_pass: 0,
    deterministic_fail: 0,
    semantic_pending: 0,
    verdict: "fail",
    generated_at: new Date().toISOString(),
    runner_error: String(error?.message || error),
  }, null, 2));
  process.exit(1);
});
