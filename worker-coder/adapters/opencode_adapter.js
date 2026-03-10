import fs from "fs";
import path from "path";
import { spawn } from "child_process";

function runProcess({ command, args = [], cwd, timeoutMs = 600000, stdinText = "" }) {
  return new Promise((resolve) => {
    console.log(`[opencode] Executing: ${command} ${args.join(" ")} (timeout: ${timeoutMs}ms)`);
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
    const promptText = String(taskPrompt || "").trim();
    const commandName = String(opencodeCommand[0]);
    if (commandName === "mock-inline-autofix") {
      return {
        command: commandName,
        args: opencodeCommand.slice(1).map((x) => String(x)
          .replace(/\{\{task_prompt\}\}/g, promptText)
          .replace(/\{\{model\}\}/g, String(model || ""))),
        stdinText: "",
        commandSource: "payload.opencode_command",
      };
    }
    return {
      command: commandName,
      args: opencodeCommand.slice(1).map((x) => String(x)
        .replace(/\{\{task_prompt\}\}/g, promptText)
        .replace(/\{\{model\}\}/g, String(model || ""))),
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

async function runInlineMockProcess({ cwd, args = [] }) {
  const targetRel = String(args[0] || "sandbox/crm_site/app.js").replace(/\\/g, "/");
  const prompt = String(args[1] || "");
  const targetAbs = path.resolve(cwd, targetRel);
  fs.mkdirSync(path.dirname(targetAbs), { recursive: true });
  if (prompt.includes("[Auto-Fix Retry]")) {
    fs.writeFileSync(targetAbs, "const status = 'fixed';\nmodule.exports = { status };\n", "utf8");
    return {
      ok: true,
      exitCode: 0,
      stdout: "inline mock autofix provider: repaired file\n",
      stderr: "",
      timedOut: false,
    };
  }
  fs.writeFileSync(targetAbs, "const status = ;\nmodule.exports = { status };\n", "utf8");
  return {
    ok: true,
    exitCode: 0,
    stdout: "inline mock autofix provider: wrote broken file\n",
    stderr: "",
    timedOut: false,
  };
}

function mapErrorCode({ proc, command }) {
  if (proc.timedOut) return "E_TIMEOUT";
  const stderrText = String(proc.stderr || "");
  const lower = stderrText.toLowerCase();
  const commandNotFound =
    stderrText.includes("ENOENT") ||
    lower.includes("not recognized as an internal or external command") ||
    lower.includes("command not found");
  if (commandNotFound) return "E_PROVIDER_UNAVAILABLE";
  if (!proc.ok && /(apply|patch).*(fail|error)/i.test(stderrText)) return "E_APPLY_FAILED";
  if (!proc.ok) return "E_EXEC_FAILED";
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

    const effectiveProc = invocation.command === "mock-inline-autofix"
      ? await runInlineMockProcess({ cwd: workspaceRoot, args: invocation.args })
      : await runProcess({
        command: invocation.command,
        args: invocation.args,
        cwd: workspaceRoot,
        timeoutMs: Math.max(1, Number(maxRuntimeS || 600)) * 1000,
        stdinText: invocation.stdinText,
      });

    const errorCode = mapErrorCode({ proc: effectiveProc, command: invocation.command });
    const errorMsg = effectiveProc.ok
      ? null
      : (effectiveProc.timedOut
        ? "OpenCode command timed out"
        : (errorCode === "E_APPLY_FAILED"
          ? "OpenCode apply phase failed"
          : "OpenCode command failed"));

    return {
      ok: effectiveProc.ok,
      provider_used: "opencode",
      model_used: model || null,
      command_used: [invocation.command, ...invocation.args].join(" "),
      command_source: invocation.commandSource,
      stdout: effectiveProc.stdout,
      stderr: effectiveProc.stderr,
      diagnostics: {
        error_code: errorCode,
        exit_code: effectiveProc.exitCode,
        timeout: effectiveProc.timedOut,
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
