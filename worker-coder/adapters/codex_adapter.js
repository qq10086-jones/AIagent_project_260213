import { spawn } from "child_process";
import fs from "fs";

function runProcess({ command, args = [], cwd, timeoutMs = 600000, stdinText = "" }) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
      shell: false,
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      try {
        child.kill("SIGKILL");
      } catch {}
    }, timeoutMs);

    child.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    child.stderr.on("data", (d) => {
      stderr += d.toString();
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({
        ok: false,
        exitCode: null,
        stdout,
        stderr: `${stderr}\n${err.message}`.trim(),
        timedOut,
      });
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({
        ok: !timedOut && code === 0,
        exitCode: code,
        stdout,
        stderr,
        timedOut,
      });
    });

    if (stdinText) {
      child.stdin.write(stdinText);
    }
    child.stdin.end();
  });
}

export function buildCodexInvocation({
  taskPrompt,
  model,
  codexCommand,
}) {
  if (Array.isArray(codexCommand) && codexCommand.length > 0) {
    return {
      command: String(codexCommand[0]),
      args: codexCommand.slice(1).map((x) => String(x)),
      stdinText: "",
      commandSource: "payload.codex_command",
    };
  }

  const command = process.env.CODEX_BIN || "codex";
  const sandboxMode = process.env.CODEX_SANDBOX || "workspace-write";
  const args = [
    "exec",
    "--sandbox",
    sandboxMode,
    "--skip-git-repo-check",
    String(taskPrompt || "").trim(),
  ];
  if (model) {
    args.push("--model", String(model));
  }

  return {
    command,
    args,
    stdinText: "",
    commandSource: "default",
  };
}

export async function runCodexTask({
  workspaceRoot,
  taskPrompt,
  model,
  maxRuntimeS = 600,
  codexCommand,
}) {
  if (!taskPrompt || !String(taskPrompt).trim()) {
    return {
      ok: false,
      error: "task_prompt is required for coding.delegate",
      diagnostics: { error_code: "E_INVALID_INPUT" },
    };
  }

  const hasApiKey = !!String(process.env.OPENAI_API_KEY || "").trim();
  const hasCodexAuth = fs.existsSync("/root/.codex/auth.json");
  if (!hasApiKey && !hasCodexAuth) {
    return {
      ok: false,
      provider_used: "codex",
      model_used: model || null,
      stdout: "",
      stderr: "",
      diagnostics: { error_code: "E_PROVIDER_UNAVAILABLE", exit_code: null, timeout: false },
      error: "Codex auth missing: set OPENAI_API_KEY or mount /root/.codex/auth.json into worker-coder",
    };
  }

  const invocation = buildCodexInvocation({
    taskPrompt,
    model,
    codexCommand,
  });

  const proc = await runProcess({
    command: invocation.command,
    args: invocation.args,
    cwd: workspaceRoot,
    timeoutMs: Math.max(1, Number(maxRuntimeS || 600)) * 1000,
    stdinText: invocation.stdinText,
  });

  const stderrText = String(proc.stderr || "");
  const commandNotFound =
    stderrText.includes("ENOENT") && invocation.command.toLowerCase() === "codex";
  const errorCode = proc.timedOut
    ? "E_TIMEOUT"
    : (commandNotFound ? "E_PROVIDER_UNAVAILABLE" : (proc.ok ? null : "E_PROVIDER_UNAVAILABLE"));
  const errorMsg = commandNotFound
    ? "Codex CLI not found in worker-coder container (spawn codex ENOENT)"
    : (proc.ok ? null : (proc.timedOut ? "Codex command timed out" : "Codex command failed"));

  return {
    ok: proc.ok,
    provider_used: "codex",
    model_used: model || null,
    command_used: [invocation.command, ...invocation.args].join(" "),
    command_source: invocation.commandSource,
    stdout: proc.stdout,
    stderr: proc.stderr,
    diagnostics: {
      error_code: errorCode,
      exit_code: proc.exitCode,
      timeout: proc.timedOut,
    },
    error: errorMsg,
  };
}
