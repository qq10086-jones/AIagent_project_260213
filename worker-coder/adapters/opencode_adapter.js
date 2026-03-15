import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";

function writeText(absPath, content) {
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf8");
}

function writeJson(absPath, value) {
  writeText(absPath, JSON.stringify(value, null, 2));
}

function normalizeOpenCodeModelRef(model) {
  const safe = String(model || "").trim();
  if (!safe) return "";
  if (safe.includes("/")) return safe;
  if (/^qwen3-coder-plus(?:-\d{4}-\d{2}-\d{2})?$/i.test(safe)) {
    return "alibaba-coding-plan/qwen3-coder-plus";
  }
  if (/^qwen/i.test(safe)) return `alibaba-coding-plan/${safe}`;
  if (/^(gpt|o\d|text-embedding)/i.test(safe)) return `openai/${safe}`;
  if (/^claude/i.test(safe)) return `anthropic/${safe}`;
  return safe;
}

function validateOpenCodeModelRef(model) {
  const normalized = normalizeOpenCodeModelRef(model);
  if (!normalized) {
    return { ok: true, normalizedModel: normalized };
  }
  const providerId = String(normalized.split("/")[0] || "").trim().toLowerCase();
  if (providerId === "dashscope") {
    return {
      ok: false,
      normalizedModel: normalized,
      errorCode: "E_PROVIDER_CONFIG",
      providerErrorClass: "PROVIDER_CONFIG_ERROR",
      error: "OpenCode does not accept dashscope/* model refs; use alibaba-coding-plan/* or opencode/*.",
    };
  }
  return { ok: true, normalizedModel: normalized };
}

function validateOpenCodeCredentials(model) {
  const normalized = normalizeOpenCodeModelRef(model);
  const providerId = String(normalized.split("/")[0] || "").trim().toLowerCase();
  const hasOpenCodeAuth = Boolean(String(process.env.OPENCODE_API_KEY || process.env.OPENCODE_ZEN_API_KEY || "").trim())
    || fs.existsSync(path.join(os.homedir(), ".local", "share", "opencode", "auth.json"));
  if (providerId === "opencode" || providerId === "opencode-go") {
    if (!hasOpenCodeAuth) {
      return {
        ok: false,
        errorCode: "E_AUTH_FAILED",
        providerErrorClass: "AUTH_FAILURE",
        error: "OpenCode auth missing: set OPENCODE_API_KEY/OPENCODE_ZEN_API_KEY or login via opencode auth login.",
      };
    }
  }
  if (providerId === "alibaba-coding-plan" || providerId === "alibaba-coding-plan-cn") {
    if (!String(process.env.ALIBABA_CODING_PLAN_API_KEY || "").trim()) {
      return {
        ok: false,
        errorCode: "E_AUTH_FAILED",
        providerErrorClass: "AUTH_FAILURE",
        error: "Alibaba Coding Plan auth missing: set ALIBABA_CODING_PLAN_API_KEY.",
      };
    }
  }
  return { ok: true };
}

function extractPromptValue(prompt, label) {
  const escaped = String(label).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = String(prompt || "").match(new RegExp(`${escaped}:\\s*(.+)`, "i"));
  return match ? String(match[1]).trim() : "";
}

function extractRequiredOutputs(prompt) {
  const match = String(prompt || "").match(/required_outputs:\s*([^\n]+)/i);
  if (!match) return [];
  return String(match[1])
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildArtifactRoot(cwd, prompt, artifactWorkspaceRoot = null) {
  const raw = extractPromptValue(prompt, "Absolute artifact output root");
  if (!raw) return null;
  if (/^[A-Za-z]:[\\/]/.test(raw) || raw.startsWith("/")) {
    if (/^\/workspace/i.test(raw)) {
      const baseRoot = artifactWorkspaceRoot || cwd;
      const rel = raw
        .replace(/^\/workspace/i, "")
        .replace(/^\/+/, "")
        .replace(/\//g, path.sep);
      return rel ? path.resolve(baseRoot, rel) : path.resolve(baseRoot);
    }
    return path.resolve(raw);
  }
  return path.resolve(cwd, raw.replace(/\//g, path.sep));
}

function createPmArtifacts({ cwd, prompt, artifactWorkspaceRoot }) {
  const artifactRoot = buildArtifactRoot(cwd, prompt, artifactWorkspaceRoot);
  if (!artifactRoot) return false;
  const now = new Date().toISOString();
  writeText(path.join(artifactRoot, "plan", "spec.md"), [
    "# Scope",
    "",
    "Deliver a minimal CRM web app with customer list, detail page, and add/edit form.",
    "",
    "# User Stories",
    "",
    "- As a user, I can browse customers.",
    "- As a user, I can inspect one customer record.",
    "- As a user, I can create and edit customer records.",
    "",
    "# Acceptance Criteria",
    "",
    "- Customer list renders deterministically.",
    "- Customer detail can be viewed from the list.",
    "- Create and edit flows persist through the backend API.",
    "",
    "# Non-Goals",
    "",
    "- No auth system.",
    "- No bulk import.",
    "",
    "# Artifact List",
    "",
    "- plan/spec.md",
    "- plan/acceptance.json",
    "- plan/milestones.md",
    "- handoff/pm_to_architect.json",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "plan", "acceptance.json"), {
    criteria: [
      "Customer list renders deterministically",
      "Customer detail is viewable",
      "Create and edit customer flows are supported",
    ],
    artifacts: ["plan/spec.md", "plan/milestones.md", "handoff/pm_to_architect.json"],
    owner: "pm",
    version: "v1",
    generated_at: now,
    step_id: "pm_spec",
  });
  writeText(path.join(artifactRoot, "plan", "milestones.md"), [
    "# Milestones",
    "",
    "1. PM scope and acceptance finalized.",
    "2. Architecture and interfaces defined.",
    "3. Backend and frontend implementation completed.",
    "4. QA verification and release pack assembled.",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "handoff", "pm_to_architect.json"), {
    from_step: "pm_spec",
    to_steps: ["arch_design"],
    scope_summary: "Minimal CRM with list, detail, create, and edit flows.",
    artifacts: ["plan/spec.md", "plan/acceptance.json", "plan/milestones.md"],
    acceptance: {
      criteria: [
        "Customer list renders deterministically",
        "Customer detail is viewable",
        "Create and edit customer flows are supported",
      ],
    },
  });
  return true;
}

function createArchArtifacts({ cwd, prompt, artifactWorkspaceRoot }) {
  const artifactRoot = buildArtifactRoot(cwd, prompt, artifactWorkspaceRoot);
  if (!artifactRoot) return false;
  const now = new Date().toISOString();
  writeText(path.join(artifactRoot, "plan", "arch.md"), [
    "# Module Breakdown",
    "",
    "## Modules",
    "- frontend app for customer list/detail/form",
    "- backend API for customer CRUD",
    "- in-memory repository for deterministic validation",
    "",
    "## Interfaces",
    "",
    "- GET /api/customers",
    "- GET /api/customers/:id",
    "- POST /api/customers",
    "- PUT /api/customers/:id",
    "",
    "## Dependency Choices",
    "",
    "- Express for backend routing",
    "- Vanilla browser JS for frontend runtime",
    "",
    "## Risk Notes",
    "",
    "- Keep response shapes stable between FE and BE.",
    "- Limit edits to CRM sandbox files only.",
  ].join("\n"));
  writeText(path.join(artifactRoot, "plan", "interfaces.md"), [
    "# API Interfaces",
    "",
    "## GET /api/customers",
    "Returns a JSON array of customers.",
    "",
    "## GET /api/customers/:id",
    "Returns one customer by id.",
    "",
    "## POST /api/customers",
    "Creates a new customer from JSON payload.",
    "",
    "## PUT /api/customers/:id",
    "Updates an existing customer from JSON payload.",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "risk", "risk_report.json"), {
    risks: [
      { title: "API/frontend contract drift", mitigation: "Keep interface headings and shared JSON fields aligned", level: "medium" },
      { title: "Out-of-scope file edits", mitigation: "Restrict edits to sandbox/crm_site target paths", level: "low" },
    ],
    decision_log: [
      "Use Express handlers for deterministic CRUD endpoints",
      "Keep frontend logic in a single sandbox app.js for validation",
    ],
    generated_at: now,
  });
  writeText(path.join(artifactRoot, "plan", "workplan.md"), [
    "# Workplan",
    "",
    "- module breakdown approved for frontend and backend paths",
    "- interface contract frozen before implementation",
    "- dependency choices limited to existing runtime stack",
    "- risk notes reviewed before QA verification",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "handoff", "architect_to_impl.json"), {
    from_step: "arch_design",
    to_steps: ["impl_be", "impl_fe", "qa_verify"],
    modules: ["frontend app", "backend api", "customer repository"],
    interfaces: [
      "GET /api/customers",
      "GET /api/customers/:id",
      "POST /api/customers",
      "PUT /api/customers/:id",
    ],
    decisions: [
      { adr_id: "ADR-001", title: "Use Express for backend API", status: "accepted" },
      { adr_id: "ADR-002", title: "Keep frontend in sandbox/crm_site/app.js", status: "accepted" },
    ],
    risks: ["API/frontend contract drift", "Out-of-scope file edits"],
  });
  return true;
}

function createReleaseArtifacts({ cwd, prompt, artifactWorkspaceRoot }) {
  const artifactRoot = buildArtifactRoot(cwd, prompt, artifactWorkspaceRoot);
  if (!artifactRoot) return false;
  const now = new Date().toISOString();
  const runId = String(artifactRoot).split(path.sep).pop() || "unknown-run";
  const requiredOutputs = extractRequiredOutputs(prompt);
  const artifacts = requiredOutputs.length > 0
    ? requiredOutputs.map((rel) => ({
      path: rel,
      type: rel.endsWith(".json") ? "json" : "markdown",
      size_bytes: 128,
    }))
    : [
      { path: "release/release_notes.md", type: "markdown", size_bytes: 180 },
      { path: "release/artifact_manifest.json", type: "json", size_bytes: 240 },
    ];
  writeText(path.join(artifactRoot, "release", "release_notes.md"), [
    "# Artifact Traceability",
    "",
    "- Verified PM, architecture, implementation, and QA artifacts are packaged.",
    "",
    "# Go / No-Go Summary",
    "",
    "- GO for controlled M9 workflow validation.",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "release", "artifact_manifest.json"), {
    run_id: runId,
    workflow_id: "coding_team_v0",
    completed_at: now,
    artifacts,
  });
  return true;
}

function createBackendImplArtifacts({ cwd, prompt, fixed = false, artifactWorkspaceRoot }) {
  const artifactRoot = buildArtifactRoot(cwd, prompt, artifactWorkspaceRoot);
  if (!artifactRoot) return false;
  const targetFile = "sandbox/crm_site/server.js";
  const finalContent = fixed
    ? "const status = 'fixed';\nmodule.exports = { status };\n"
    : "const status = ;\nmodule.exports = { status };\n";
  writeText(path.join(artifactRoot, "impl", "be_changes", "server.js"), finalContent);
  writeJson(path.join(artifactRoot, "impl", "be_patch_bundle.json"), {
    bundle_id: "mock-be-bundle",
    step_id: "impl_be",
    mode: "full_file_fallback",
    operations: [
      {
        type: "create_file",
        target_file: targetFile,
        file_content: finalContent,
      },
    ],
    target_files: [targetFile],
    summary: "Mock backend patch bundle for live workflow validation.",
  });
  writeText(path.join(artifactRoot, "impl", "be_notes.md"), [
    "# Backend Notes",
    "",
    "## API Contracts",
    "",
    "- GET /api/customers",
    "- GET /api/customers/:id",
    "- POST /api/customers",
    "- PUT /api/customers/:id",
    "",
    "## Shared Types",
    "",
    "- Customer: id, name, email",
    "",
    "## Scope Constraints",
    "",
    "- Only CRM sandbox backend file is modified.",
    "",
    "## Run Instructions",
    "",
    "1. Run node --check sandbox/crm_site/server.js",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "handoff", "be_to_fe.json"), {
    from_step: "impl_be",
    to_step: "impl_fe",
    be_changes_path: "impl/be_changes",
    api_contracts: [
      { name: "List Customers", method: "GET", path: "/api/customers", response_shape: "Customer[]" },
      { name: "Get Customer", method: "GET", path: "/api/customers/:id", response_shape: "Customer" },
      { name: "Create Customer", method: "POST", path: "/api/customers", response_shape: "Customer" },
      { name: "Update Customer", method: "PUT", path: "/api/customers/:id", response_shape: "Customer" },
    ],
    shared_types: [
      { name: "Customer", description: "Shared CRM customer record.", fields: ["id", "name", "email"] },
    ],
    scope_constraints: ["Only sandbox/crm_site/server.js is modified."],
  });
  return true;
}

function createFrontendImplArtifacts({ cwd, prompt, fixed = false, artifactWorkspaceRoot }) {
  const artifactRoot = buildArtifactRoot(cwd, prompt, artifactWorkspaceRoot);
  if (!artifactRoot) return false;
  const targetFile = "sandbox/crm_site/app.js";
  const finalContent = fixed
    ? "const status = 'fixed';\nmodule.exports = { status };\n"
    : "const status = ;\nmodule.exports = { status };\n";
  writeText(path.join(artifactRoot, "impl", "fe_changes", "app.js"), finalContent);
  writeJson(path.join(artifactRoot, "impl", "fe_patch_bundle.json"), {
    bundle_id: "mock-fe-bundle",
    step_id: "impl_fe",
    mode: "full_file_fallback",
    operations: [
      {
        type: "create_file",
        target_file: targetFile,
        file_content: finalContent,
      },
    ],
    target_files: [targetFile],
    summary: "Mock frontend patch bundle for live workflow validation.",
  });
  writeText(path.join(artifactRoot, "impl", "fe_notes.md"), [
    "# Frontend Notes",
    "",
    "## UI Scope",
    "",
    "- Customer list",
    "- Customer detail",
    "- Customer create/edit form",
    "",
    "## API Consumption",
    "",
    "- Consume only endpoints declared in handoff/be_to_fe.json",
    "",
    "## Run Instructions",
    "",
    "1. Run node --check sandbox/crm_site/app.js",
  ].join("\n"));
  writeJson(path.join(artifactRoot, "handoff", "impl_to_qa.json"), {
    from_steps: ["impl_be", "impl_fe"],
    to_step: "qa_verify",
    be_changes_path: "impl/be_changes",
    fe_changes_path: "impl/fe_changes",
    run_instructions: "Start backend, start frontend, then verify the CRM list/detail/add-edit flows.",
    known_limitations: [
      "Authentication flow not implemented",
      "Advanced filtering is out of scope",
    ],
    api_contracts_path: "handoff/be_to_fe.json",
  });
  return true;
}

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
  const normalizedModel = normalizeOpenCodeModelRef(model);
  if (Array.isArray(opencodeCommand) && opencodeCommand.length > 0) {
    const promptText = String(taskPrompt || "").trim();
    const commandName = String(opencodeCommand[0]);
    if (commandName === "mock-inline-autofix") {
      return {
        command: commandName,
        args: opencodeCommand.slice(1).map((x) => String(x)
          .replace(/\{\{task_prompt\}\}/g, promptText)
          .replace(/\{\{model\}\}/g, normalizedModel)),
        stdinText: "",
        commandSource: "payload.opencode_command",
      };
    }
    return {
      command: commandName,
      args: opencodeCommand.slice(1).map((x) => String(x)
        .replace(/\{\{task_prompt\}\}/g, promptText)
        .replace(/\{\{model\}\}/g, normalizedModel)),
      stdinText: "",
      commandSource: "payload.opencode_command",
    };
  }

  const command = process.env.OPENCODE_BIN || "opencode";
  const args = ["run", String(taskPrompt || "").trim()];
  if (normalizedModel) {
    args.push("--model", normalizedModel);
  }

  return {
    command,
    args,
    stdinText: "",
    commandSource: "default",
  };
}

async function runInlineMockProcess({ cwd, args = [], artifactWorkspaceRoot = null }) {
  const targetRel = String(args[0] || "sandbox/crm_site/app.js").replace(/\\/g, "/");
  const prompt = String(args[1] || "");
  const targetAbs = path.resolve(cwd, targetRel);
  fs.mkdirSync(path.dirname(targetAbs), { recursive: true });
  if (prompt.includes("Step ID: pm_spec")) {
    createPmArtifacts({ cwd, prompt, artifactWorkspaceRoot });
    writeText(targetAbs, "pm spec placeholder\n");
    return {
      ok: true,
      exitCode: 0,
      stdout: "inline mock autofix provider: wrote pm artifacts\n",
      stderr: "",
      timedOut: false,
    };
  }
  if (prompt.includes("Step ID: arch_design")) {
    createArchArtifacts({ cwd, prompt, artifactWorkspaceRoot });
    writeText(targetAbs, "arch design placeholder\n");
    return {
      ok: true,
      exitCode: 0,
      stdout: "inline mock autofix provider: wrote architect artifacts\n",
      stderr: "",
      timedOut: false,
    };
  }
  if (prompt.includes("Step ID: release_pack")) {
    createReleaseArtifacts({ cwd, prompt, artifactWorkspaceRoot });
    writeText(targetAbs, "release pack placeholder\n");
    return {
      ok: true,
      exitCode: 0,
      stdout: "inline mock autofix provider: wrote release artifacts\n",
      stderr: "",
      timedOut: false,
    };
  }
  if (prompt.includes("Step ID: impl_be")) {
    createBackendImplArtifacts({ cwd, prompt, fixed: prompt.includes("[Auto-Fix Retry]"), artifactWorkspaceRoot });
  }
  if (prompt.includes("Step ID: impl_fe")) {
    createFrontendImplArtifacts({ cwd, prompt, fixed: prompt.includes("[Auto-Fix Retry]"), artifactWorkspaceRoot });
  }
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
  const stderrText = String(proc.stderr || "");
  const lower = stderrText.toLowerCase();
  const commandName = String(command || "").trim().toLowerCase();
  const commandNotFound =
    stderrText.includes("ENOENT") ||
    lower.includes("not recognized as an internal or external command") ||
    lower.includes("command not found");

  if (proc.timedOut) {
    return { errorCode: "E_TIMEOUT", providerErrorClass: "EXECUTION_TIMEOUT" };
  }
  if (proc.ok) {
    return { errorCode: null, providerErrorClass: null };
  }
  if (commandNotFound) {
    return { errorCode: "E_PROVIDER_UNAVAILABLE", providerErrorClass: "PROVIDER_UNAVAILABLE" };
  }
  if (/(invalid access token|token expired|unauthorized|authentication failed|auth failed|incorrect api key provided|apikey-error|401)/i.test(stderrText)) {
    return { errorCode: "E_AUTH_FAILED", providerErrorClass: "AUTH_FAILURE" };
  }
  if (/(unknown model|model not found|no such model|unsupported model)/i.test(stderrText)) {
    return { errorCode: "E_MODEL_NOT_FOUND", providerErrorClass: "MODEL_NOT_FOUND" };
  }
  if (/(missing api key|api key.*missing|base url.*missing|configuration error|config error|env.*not set)/i.test(stderrText)) {
    return { errorCode: "E_PROVIDER_CONFIG", providerErrorClass: "PROVIDER_CONFIG_ERROR" };
  }
  if (/(invalid request|bad request|request shape|malformed request|400 bad request)/i.test(stderrText)) {
    return { errorCode: "E_REQUEST_SHAPE", providerErrorClass: "REQUEST_SHAPE_ERROR" };
  }
  if (!proc.ok && /(apply|patch).*(fail|error)/i.test(stderrText)) {
    return { errorCode: "E_APPLY_FAILED", providerErrorClass: null };
  }
  if (!proc.ok) {
    const providerErrorClass = commandName === "opencode" ? "PROVIDER_EXECUTION_ERROR" : null;
    return { errorCode: "E_EXEC_FAILED", providerErrorClass };
  }
  return { errorCode: null, providerErrorClass: null };
}

export async function runOpenCodeTask({
  workspaceRoot,
  artifactWorkspaceRoot = null,
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
        diagnostics: { error_code: "E_INVALID_INPUT", provider_class: "opencode-provider", provider_error_class: "REQUEST_SHAPE_ERROR" },
      };
    }

    const invocation = buildOpenCodeInvocation({
      taskPrompt,
      model,
      opencodeCommand,
    });
    const modelValidation = validateOpenCodeModelRef(model);
    if (!modelValidation.ok) {
      return {
        ok: false,
        provider_used: "opencode",
        model_used: modelValidation.normalizedModel || normalizeOpenCodeModelRef(model) || null,
        command_used: [invocation.command, ...invocation.args].join(" "),
        command_source: invocation.commandSource,
        stdout: "",
        stderr: "",
        diagnostics: {
          error_code: modelValidation.errorCode,
          exit_code: null,
          timeout: false,
          provider_class: "opencode-provider",
          provider_error_class: modelValidation.providerErrorClass,
        },
        error: modelValidation.error,
      };
    }
    const credentialValidation = validateOpenCodeCredentials(modelValidation.normalizedModel || model);
    if (!credentialValidation.ok) {
      return {
        ok: false,
        provider_used: "opencode",
        model_used: modelValidation.normalizedModel || normalizeOpenCodeModelRef(model) || null,
        command_used: [invocation.command, ...invocation.args].join(" "),
        command_source: invocation.commandSource,
        stdout: "",
        stderr: "",
        diagnostics: {
          error_code: credentialValidation.errorCode,
          exit_code: null,
          timeout: false,
          provider_class: "opencode-provider",
          provider_error_class: credentialValidation.providerErrorClass,
        },
        error: credentialValidation.error,
      };
    }

    const effectiveProc = invocation.command === "mock-inline-autofix"
      ? await runInlineMockProcess({ cwd: workspaceRoot, args: invocation.args, artifactWorkspaceRoot })
      : await runProcess({
        command: invocation.command,
        args: invocation.args,
        cwd: workspaceRoot,
        timeoutMs: Math.max(1, Number(maxRuntimeS || 600)) * 1000,
        stdinText: invocation.stdinText,
      });

    const { errorCode, providerErrorClass } = mapErrorCode({ proc: effectiveProc, command: invocation.command });
    const fatalProviderError = Boolean(errorCode);
    const effectiveOk = Boolean(effectiveProc.ok) && !fatalProviderError;
    let errorMsg = null;
    if (!effectiveOk) {
      if (effectiveProc.timedOut) errorMsg = "OpenCode command timed out";
      else if (errorCode === "E_AUTH_FAILED") errorMsg = "OpenCode authentication failed";
      else if (errorCode === "E_MODEL_NOT_FOUND") errorMsg = "OpenCode model resolution failed";
      else if (errorCode === "E_PROVIDER_CONFIG") errorMsg = "OpenCode provider configuration failed";
      else if (errorCode === "E_REQUEST_SHAPE") errorMsg = "OpenCode request shape invalid";
      else if (errorCode === "E_APPLY_FAILED") errorMsg = "OpenCode apply phase failed";
      else errorMsg = "OpenCode command failed";
    }

    return {
      ok: effectiveOk,
      provider_used: "opencode",
      model_used: normalizeOpenCodeModelRef(model) || null,
      command_used: [invocation.command, ...invocation.args].join(" "),
      command_source: invocation.commandSource,
      stdout: effectiveProc.stdout,
      stderr: effectiveProc.stderr,
      diagnostics: {
        error_code: errorCode,
        exit_code: effectiveProc.exitCode,
        timeout: effectiveProc.timedOut,
        provider_class: "opencode-provider",
        provider_error_class: providerErrorClass,
      },
      error: errorMsg,
    };
  } catch (err) {
    return {
      ok: false,
      provider_used: "opencode",
      model_used: normalizeOpenCodeModelRef(model) || null,
      stdout: "",
      stderr: "",
      diagnostics: {
        error_code: "E_INTERNAL",
        exit_code: null,
        timeout: false,
        provider_class: "opencode-provider",
        provider_error_class: "PROVIDER_EXECUTION_ERROR",
      },
      error: `OpenCode internal error: ${err.message}`,
    };
  }
}
