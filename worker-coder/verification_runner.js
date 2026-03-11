import fs from "fs";
import path from "path";
import { exec } from "child_process";
import vm from "vm";

const ALLOWED_CMD_PREFIXES = new Set([
  "python",
  "pytest",
  "npm",
  "node",
  "git",
  "ls",
  "cat",
  "echo",
  "pwd",
  "grep",
  "rg",
  "fd",
  "ruff",
  "black",
]);

function splitCommandChain(command) {
  return String(command || "")
    .split("&&")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function redactSensitiveText(value) {
  let text = String(value || "");
  if (!text) return text;
  const rules = [
    { re: /\bsk-[A-Za-z0-9_-]{20,}\b/g, to: "[REDACTED_OPENAI_KEY]" },
    { re: /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, to: "[REDACTED_GITHUB_TOKEN]" },
    { re: /\bAIza[0-9A-Za-z\-_]{20,}\b/g, to: "[REDACTED_GOOGLE_API_KEY]" },
    { re: /\bAKIA[0-9A-Z]{16}\b/g, to: "[REDACTED_AWS_ACCESS_KEY]" },
    { re: /\b(?:xoxb|xoxp|xoxa|xoxr)-[A-Za-z0-9-]{10,}\b/g, to: "[REDACTED_SLACK_TOKEN]" },
    { re: /\b(token|api[_-]?key|secret|password)\s*[:=]\s*['"]?[A-Za-z0-9_\-\/+=.]{8,}['"]?/gi, to: "$1=[REDACTED]" },
  ];
  for (const rule of rules) {
    text = text.replace(rule.re, rule.to);
  }
  return text;
}

export async function execCapture(command, cwd) {
  return new Promise((resolve) => {
    exec(command, { cwd, timeout: 20000 }, (error, stdout, stderr) => {
      resolve({
        ok: !error,
        stdout: (stdout || "").toString(),
        stderr: (stderr || "").toString(),
      });
    });
  });
}

export function validateSafeCommand(command) {
  const raw = String(command || "").trim();
  if (!raw) return { ok: false, error: "Command blocked: empty command." };
  if (/[;|><`$(){}[\]\\*?~\n\r]/.test(raw)) {
    return { ok: false, error: "Command blocked: forbidden shell meta-character detected." };
  }
  if (/(^|[^&])&([^&]|$)/.test(raw)) {
    return { ok: false, error: "Command blocked: unsupported '&' operator." };
  }
  const segments = splitCommandChain(raw);
  if (segments.length === 0) {
    return { ok: false, error: "Command blocked: empty command segment." };
  }
  for (const segment of segments) {
    const prefix = segment.trim().split(/\s+/)[0];
    if (!ALLOWED_CMD_PREFIXES.has(prefix)) {
      return { ok: false, error: `Command blocked: '${prefix}' is not whitelisted.` };
    }
  }
  return { ok: true };
}

export async function runInlineNodeSyntaxCheck(filePath) {
  try {
    const code = fs.readFileSync(filePath, "utf8");
    if (typeof vm.SourceTextModule === "function") {
      new vm.SourceTextModule(code, { identifier: filePath });
    } else {
      new vm.Script(code, { filename: filePath });
    }
    return {
      ok: true,
      stdout: "",
      stderr: "",
      exitCode: 0,
    };
  } catch (err) {
    return {
      ok: false,
      stdout: "",
      stderr: String(err?.message || err || ""),
      exitCode: 1,
    };
  }
}

function flushVerificationResult(taskDir, payload) {
  let logPath = null;
  try {
    logPath = path.join(taskDir, `verification_${Date.now()}.json`);
    fs.writeFileSync(logPath, JSON.stringify({
      generated_at: new Date().toISOString(),
      ...payload,
    }, null, 2), "utf8");
  } catch {
    logPath = null;
  }
  return {
    ...payload,
    logPath,
  };
}

function flushVerificationPlanResult(taskDir, payload) {
  let logPath = null;
  try {
    logPath = path.join(taskDir, `verification_plan_${Date.now()}.json`);
    fs.writeFileSync(logPath, JSON.stringify({
      generated_at: new Date().toISOString(),
      ...payload,
    }, null, 2), "utf8");
  } catch {
    logPath = null;
  }
  return {
    ...payload,
    logPath,
  };
}

export function normalizeVerificationPlan({ verificationPlan = null, verificationCommand = "" }) {
  const normalized = [];
  if (Array.isArray(verificationPlan)) {
    for (const item of verificationPlan) {
      if (!item || typeof item !== "object") continue;
      normalized.push({
        tier: String(item.tier || "").trim().toLowerCase() || "custom",
        command: String(item.command || "").trim(),
        required: item.required !== false,
        source: String(item.source || "").trim() || "verification_plan",
      });
    }
  }
  if (normalized.length === 0 && String(verificationCommand || "").trim()) {
    normalized.push({
      tier: "legacy",
      command: String(verificationCommand || "").trim(),
      required: true,
      source: "verification_command",
    });
  }
  return normalized;
}

export async function runVerificationCommand({ workspaceRoot, verificationCommand, taskDir }) {
  const command = String(verificationCommand || "").trim();
  if (!command) {
    return { checked: false, ok: true, command: "", logPath: null };
  }
  const checked = validateSafeCommand(command);
  if (!checked.ok) {
    return flushVerificationResult(taskDir, {
      checked: true,
      ok: false,
      command,
      exit_code: null,
      stdout: "",
      stderr: "",
      error: checked.error,
      error_code: "E_VERIFICATION_COMMAND_INVALID",
    });
  }
  let proc = null;
  const inlineNodeCheckMatch = command.match(/^node\s+--check\s+(.+)$/);
  if (inlineNodeCheckMatch) {
    const targetPath = inlineNodeCheckMatch[1].trim().replace(/^"|"$/g, "");
    proc = await runInlineNodeSyntaxCheck(path.resolve(workspaceRoot, targetPath));
  } else {
    proc = await execCapture(command, workspaceRoot);
  }
  return flushVerificationResult(taskDir, {
    checked: true,
    ok: proc.ok,
    command,
    exit_code: proc.ok ? 0 : 1,
    stdout: redactSensitiveText(proc.stdout || ""),
    stderr: redactSensitiveText(proc.stderr || ""),
    error: proc.ok ? null : `verification command failed: ${command}`,
    error_code: proc.ok ? null : "E_VERIFICATION_FAILED",
  });
}

export async function runVerificationPlan({
  workspaceRoot,
  verificationPlan = null,
  verificationCommand = "",
  taskDir,
}) {
  const plan = normalizeVerificationPlan({ verificationPlan, verificationCommand });
  if (plan.length === 0) {
    return {
      checked: false,
      ok: true,
      command: "",
      commands: [],
      records: [],
      achieved_tiers: [],
      unresolved_tiers: [],
      logPath: null,
      error: null,
      errorCode: null,
    };
  }

  const records = [];
  const achievedTiers = [];
  const unresolvedTiers = [];
  let firstFailure = null;

  for (const item of plan) {
    if (!item.command) {
      unresolvedTiers.push(item.tier);
      records.push({
        tier: item.tier,
        source: item.source,
        required: item.required,
        checked: false,
        ok: null,
        command: "",
        status: "unresolved",
      });
      continue;
    }
    const record = await runVerificationCommand({
      workspaceRoot,
      verificationCommand: item.command,
      taskDir,
    });
    const normalizedRecord = {
      tier: item.tier,
      source: item.source,
      required: item.required,
      checked: Boolean(record.checked),
      ok: Boolean(record.ok),
      command: record.command,
      exit_code: record.exit_code ?? null,
      stdout: record.stdout || "",
      stderr: record.stderr || "",
      error: record.error || null,
      errorCode: record.error_code || null,
      logPath: record.logPath || null,
      status: record.ok ? "passed" : "failed",
    };
    records.push(normalizedRecord);
    if (record.ok) {
      achievedTiers.push(item.tier);
      continue;
    }
    if (!firstFailure) firstFailure = normalizedRecord;
    break;
  }

  const summary = flushVerificationPlanResult(taskDir, {
    checked: records.some((item) => item.checked),
    ok: !firstFailure,
    command: records.map((item) => item.command).filter(Boolean).join(" && "),
    commands: records.map((item) => item.command).filter(Boolean),
    records,
    achieved_tiers: achievedTiers,
    unresolved_tiers: unresolvedTiers,
    error: firstFailure?.error || null,
    errorCode: firstFailure?.errorCode || null,
  });
  return {
    ...summary,
    errorCode: summary.errorCode,
  };
}
