import { spawn } from "child_process";

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

export function buildOpenCodeInvocation({
  taskPrompt,
  model,
  opencodeCommand,
}) {
  if (Array.isArray(opencodeCommand) && opencodeCommand.length > 0) {
    return {
      command: String(opencodeCommand[0]),
      args: opencodeCommand.slice(1).map((x) => String(x)),
      stdinText: "",
      commandSource: "payload.opencode_command",
    };
  }

  const command = process.env.OPENCODE_BIN || "opencode";
  const args = ["run", String(taskPrompt || "").trim()];
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

function mapErrorCode({ proc, command }) {
  if (proc.timedOut) return "E_TIMEOUT";
  const stderrText = String(proc.stderr || "");
  const commandNotFound = stderrText.includes("ENOENT");
  if (commandNotFound) return "E_PROVIDER_UNAVAILABLE";
  if (!proc.ok && /(apply|patch).*(fail|error)/i.test(stderrText)) return "E_APPLY_FAILED";
  if (!proc.ok) return "E_PROVIDER_UNAVAILABLE";
  return null;
}

export async function runOpenCodeTask({
  workspaceRoot,
  taskPrompt,
  model,
  maxRuntimeS = 600,
  opencodeCommand,
}) {
  try {
    if (!taskPrompt || !String(taskPrompt).trim()) {
      return {
        ok: false,
        error: "task_prompt is required for coding.delegate",
        diagnostics: { error_code: "E_INVALID_INPUT" },
      };
    }

    const invocation = buildOpenCodeInvocation({
      taskPrompt,
      model,
      opencodeCommand,
    });

    const proc = await runProcess({
      command: invocation.command,
      args: invocation.args,
      cwd: workspaceRoot,
      timeoutMs: Math.max(1, Number(maxRuntimeS || 600)) * 1000,
      stdinText: invocation.stdinText,
    });

    const errorCode = mapErrorCode({ proc, command: invocation.command });
    const errorMsg = proc.ok
      ? null
      : (proc.timedOut
        ? "OpenCode command timed out"
        : (errorCode === "E_APPLY_FAILED"
          ? "OpenCode apply phase failed"
          : "OpenCode command failed"));

    return {
      ok: proc.ok,
      provider_used: "opencode",
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
  } catch (err) {
    return {
      ok: false,
      provider_used: "opencode",
      model_used: model || null,
      stdout: "",
      stderr: "",
      diagnostics: {
        error_code: "E_INTERNAL",
        exit_code: null,
        timeout: false,
      },
      error: `OpenCode internal error: ${err.message}`,
    };
  }
}
