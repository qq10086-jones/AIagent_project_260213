import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

import { validateJsonSchemaLite } from "../src/schema_lite_validator.js";
import {
  resolveOrchestratorArtifactPath,
  resolveRepoPath,
} from "./_paths.js";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function parseJsonSafe(value, fallback = {}) {
  try {
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

async function getJson(url, init = {}) {
  const res = await fetch(url, init);
  const text = await res.text();
  let json = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = { raw: text };
  }
  return { ok: res.ok, status: res.status, json };
}

async function checkHealth(baseUrl) {
  const res = await fetch(`${baseUrl}/health`);
  const text = await res.text();
  if (!res.ok || String(text).trim() !== "ok") {
    throw new Error(`health failed: ${res.status} ${text}`);
  }
}

async function pollWorkflow(baseUrl, workflowRunId, timeoutMs = 420000, intervalMs = 4000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const state = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (state.ok) {
      const status = String(state.json?.run?.status || "");
      if (["succeeded", "failed", "partial_failure"].includes(status)) {
        return state.json;
      }
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting workflow ${workflowRunId}`);
}

function normalizeRelPath(value) {
  return String(value || "").replace(/\\/g, "/").replace(/^\/+/, "");
}

function sanitizeId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_");
}

function inferFocusedStepId(taskClass) {
  return ["be_create"].includes(String(taskClass || "")) ? "impl_be" : "impl_fe";
}

function inferVerificationTierAchieved(stepOutput = {}) {
  const diagnostics = stepOutput?.diagnostics && typeof stepOutput.diagnostics === "object"
    ? stepOutput.diagnostics
    : {};
  const verification = diagnostics?.verification && typeof diagnostics.verification === "object"
    ? diagnostics.verification
    : {};
  if (Array.isArray(verification.achieved_tiers) && verification.achieved_tiers.length > 0) {
    return verification.achieved_tiers.join(" + ");
  }
  const command = String(verification.command || "").trim().toLowerCase();
  if (!verification.checked) return "none";
  if (!verification.ok) return "verification_failed";
  if (command.includes("eslint") || command.includes(" lint")) return "lint";
  if (command.includes("tsc") || command.includes("type-check") || command.includes("type_check")) return "type_check";
  if (command.includes("pytest") || command.includes("unit") || command.includes("test")) return "unit_test";
  if (command.includes("build")) return "build";
  if (command.includes("node --check") || command.includes("py_compile")) return "syntax_check";
  return "verified";
}

function verificationTargetSatisfied(target, achieved) {
  const normalizedTarget = String(target || "").trim().toLowerCase();
  const normalizedAchieved = String(achieved || "").trim().toLowerCase();
  if (!normalizedTarget) return normalizedAchieved !== "none";
  if (normalizedTarget === normalizedAchieved) return true;
  const targetParts = normalizedTarget.split("+").map((item) => item.trim()).filter(Boolean);
  return targetParts.length === 1 && targetParts[0] === normalizedAchieved;
}

function deriveFailureAttribution(stepOutput = {}, fallbackErrorCode = "") {
  const diagnostics = stepOutput?.diagnostics && typeof stepOutput.diagnostics === "object"
    ? stepOutput.diagnostics
    : {};
  const direct = String(diagnostics.failure_attribution || "").trim();
  if (direct) return direct;
  const finalFailure = diagnostics?.final_failure_summary && typeof diagnostics.final_failure_summary === "object"
    ? diagnostics.final_failure_summary
    : null;
  if (finalFailure?.failure_attribution) return String(finalFailure.failure_attribution);
  const code = String(fallbackErrorCode || diagnostics.error_code || "").trim().toUpperCase();
  if (!code) return "none";
  if (code.includes("CONTEXT")) return "context_failure";
  if (code.includes("VERIFY")) return "verification_failure";
  if (code.includes("TIMEOUT") || code.includes("PROVIDER") || code.includes("DELEGATE")) return "infrastructure_failure";
  return "coding_logic_failure";
}

function buildScenarioGoal(task) {
  const scenario = String(task?.scenario || "").trim();
  const focus = String(task?.validation_focus || "").trim();
  const taskClass = String(task?.task_class || "").trim();
  return [
    `Worker-coding cohort scenario for ${taskClass}.`,
    scenario ? `Scenario: ${scenario}.` : "",
    focus ? `Validation focus: ${focus}.` : "",
    "Keep changes reviewable and artifact-complete.",
    "Stay within the scoped target paths and produce deterministic outputs.",
  ].filter(Boolean).join(" ");
}

function buildWorkflowPayload({ task, template }) {
  const cohortTaskId = sanitizeId(task.cohort_task_id);
  const focusedStepId = inferFocusedStepId(task.task_class);
  const focusedTarget = focusedStepId === "impl_be" ? "sandbox/crm_site/server.js" : "sandbox/crm_site/app.js";
  const focusedPayload = {
    target_paths: [focusedTarget],
    opencode_command: ["mock-inline-autofix", focusedTarget, "{{task_prompt}}"],
    max_attempts: 2,
    same_error_repeat_limit: 2,
    wall_clock_timeout_s: 300,
    task_class: task.task_class,
    beta_template_id: task.beta_template_id,
    context_envelope: template?.context_envelope || null,
  };

  const stepPayloads = {
    pm_spec: {
      target_paths: [`sandbox/worker_coding_cohort/${cohortTaskId}/pm_spec.txt`],
      opencode_command: ["mock-inline-autofix", `sandbox/worker_coding_cohort/${cohortTaskId}/pm_spec.txt`, "{{task_prompt}}"],
    },
    arch_design: {
      target_paths: [`sandbox/worker_coding_cohort/${cohortTaskId}/arch_design.txt`],
      opencode_command: ["mock-inline-autofix", `sandbox/worker_coding_cohort/${cohortTaskId}/arch_design.txt`, "{{task_prompt}}"],
    },
    impl_be: {
      target_paths: ["sandbox/crm_site/server.js"],
      opencode_command: ["mock-inline-autofix", "sandbox/crm_site/server.js", "{{task_prompt}}"],
      max_attempts: 2,
      same_error_repeat_limit: 2,
      wall_clock_timeout_s: 300,
    },
    impl_fe: {
      target_paths: ["sandbox/crm_site/app.js"],
      opencode_command: ["mock-inline-autofix", "sandbox/crm_site/app.js", "{{task_prompt}}"],
      max_attempts: 2,
      same_error_repeat_limit: 2,
      wall_clock_timeout_s: 300,
    },
    qa_verify: {
      command: "node --version",
    },
    release_pack: {
      target_paths: [`sandbox/worker_coding_cohort/${cohortTaskId}/release_pack.txt`],
      opencode_command: ["mock-inline-autofix", `sandbox/worker_coding_cohort/${cohortTaskId}/release_pack.txt`, "{{task_prompt}}"],
    },
  };

  stepPayloads[focusedStepId] = {
    ...stepPayloads[focusedStepId],
    ...focusedPayload,
  };

  return {
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    input: {
      goal: buildScenarioGoal(task),
      provider: "opencode",
      model: "qwen3-coder-next",
      fast_mode: true,
      max_runtime_s: 180,
      step_payloads: stepPayloads,
    },
  };
}

function toStepOutput(step = {}) {
  const raw = parseJsonSafe(step?.result_json || "{}", {});
  if (raw && typeof raw.output === "object" && raw.output !== null) {
    return raw.output;
  }
  return raw;
}

function summarizeResult({ task, terminal, focusedStep }) {
  const workflowStatus = String(terminal?.run?.status || "");
  const focusedOutput = toStepOutput(focusedStep);
  const verificationAchieved = inferVerificationTierAchieved(focusedOutput);
  const target = String(task?.verification_tier_target || "");
  const stepStatus = String(focusedStep?.status || "");
  const fallbackErrorCode = String(focusedStep?.error_code || terminal?.run?.error_code || "");
  const failureAttribution = workflowStatus === "succeeded" && stepStatus === "succeeded"
    ? "none"
    : deriveFailureAttribution(focusedOutput, fallbackErrorCode);

  let result = "fail";
  if (workflowStatus === "succeeded" && stepStatus === "succeeded") {
    result = verificationTargetSatisfied(target, verificationAchieved) ? "pass" : "partial";
  } else if (workflowStatus === "partial_failure") {
    result = "partial";
  }

  const changedFiles = Array.isArray(focusedOutput?.files_changed) ? focusedOutput.files_changed : [];
  const artifactCheck = focusedOutput?.artifact_check && typeof focusedOutput.artifact_check === "object"
    ? focusedOutput.artifact_check
    : null;

  return {
    cohort_id: String(task.cohort_task_id || ""),
    task_class: String(task.task_class || ""),
    beta_template_id: String(task.beta_template_id || ""),
    verification_tier_target: target,
    verification_tier_achieved: verificationAchieved,
    result,
    failure_attribution: failureAttribution,
    workflow_run_id: String(terminal?.run?.workflow_run_id || terminal?.workflow_run_id || ""),
    run_id: String(terminal?.run?.run_id || ""),
    task_id: String(focusedStep?.task_id || ""),
    focused_step_id: String(focusedStep?.step_id || ""),
    workflow_status: workflowStatus,
    step_status: stepStatus,
    files_changed_count: changedFiles.length,
    artifact_completeness: artifactCheck?.checked ? artifactCheck.missing?.length === 0 : null,
    operator_note: [
      `scenario=${String(task.scenario || "")}`,
      `focused_step=${String(focusedStep?.step_id || "")}`,
      `workflow_status=${workflowStatus || "unknown"}`,
    ].join("; "),
  };
}

function makeReportMd(report) {
  const lines = [
    "# Worker-Coding Cohort Report",
    "",
    `- cohort_run_id: ${report.cohort_run_id}`,
    `- generated_at: ${report.generated_at}`,
    `- total_runs: ${report.summary.total_runs}`,
    `- pass_count: ${report.summary.pass_count}`,
    `- fail_count: ${report.summary.fail_count}`,
    `- partial_count: ${report.summary.partial_count}`,
    "",
    "## Results",
  ];
  for (const item of report.results) {
    lines.push(`- ${item.cohort_id} task_class=${item.task_class} result=${item.result} verification=${item.verification_tier_achieved}/${item.verification_tier_target} failure_attribution=${item.failure_attribution}`);
  }
  return lines.join("\n");
}

export async function main(options = {}) {
  const baseUrl = String(options.baseUrl || arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const timeoutMs = Math.max(120000, Number(options.timeoutMs || arg("timeout-ms", "420000")));
  const strict = String(options.strict || arg("strict", "false")).toLowerCase() === "true";
  const planPath = path.resolve(options.planPath || arg("plan", resolveRepoPath("configs", "registry", "worker_coding_cohort_plan_v1.json")));
  const registryPath = path.resolve(options.registryPath || arg("registry", resolveRepoPath("configs", "registry", "worker_coding_beta_templates.json")));
  const schemaPath = path.resolve(options.schemaPath || arg("schema", path.resolve(process.cwd(), "contracts", "worker_coding_cohort_result.schema.json")));

  const plan = readJson(planPath);
  const registry = readJson(registryPath);
  const schema = readJson(schemaPath);
  const templatesById = new Map((registry.templates || []).map((item) => [String(item.template_id || ""), item]));
  const cohortRunId = `worker_coding_cohort_${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const results = [];

  await checkHealth(baseUrl);

  for (const task of plan.tasks || []) {
    const template = templatesById.get(String(task.beta_template_id || ""));
    const focusedStepId = inferFocusedStepId(task.task_class);
    const payload = buildWorkflowPayload({ task, template });
    const item = {
      cohort_id: String(task.cohort_task_id || ""),
      task_class: String(task.task_class || ""),
      beta_template_id: String(task.beta_template_id || ""),
      verification_tier_target: String(task.verification_tier_target || ""),
      verification_tier_achieved: "none",
      result: "fail",
      failure_attribution: "infrastructure_failure",
      operator_note: `scenario=${String(task.scenario || "")}; focused_step=${focusedStepId}`,
    };
    try {
      const startRes = await getJson(`${baseUrl}/workflow-runs/start`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!startRes.ok) {
        throw new Error(`workflow start failed: ${startRes.status}`);
      }
      const workflowRunId = String(startRes.json?.workflow_run_id || "");
      if (!workflowRunId) {
        throw new Error("workflow_run_id missing");
      }
      const terminal = await pollWorkflow(baseUrl, workflowRunId, timeoutMs, 4000);
      const focusedStep = Array.isArray(terminal?.steps)
        ? terminal.steps.find((step) => String(step?.step_id || "") === focusedStepId) || null
        : null;
      if (!focusedStep) {
        throw new Error(`focused step missing: ${focusedStepId}`);
      }
      results.push(summarizeResult({ task, terminal, focusedStep }));
    } catch (err) {
      results.push({
        ...item,
        failure_attribution: "infrastructure_failure",
        operator_note: `${item.operator_note}; error=${err.message || String(err)}`,
      });
    }
  }

  const report = {
    cohort_run_id: cohortRunId,
    generated_at: new Date().toISOString(),
    summary: {
      total_runs: results.length,
      pass_count: results.filter((item) => item.result === "pass").length,
      fail_count: results.filter((item) => item.result === "fail").length,
      partial_count: results.filter((item) => item.result === "partial").length,
    },
    results,
  };

  const errors = validateJsonSchemaLite(schema, report);
  if (errors.length > 0) {
    throw new Error(`cohort report schema invalid: ${errors.join("; ")}`);
  }

  const outDir = resolveOrchestratorArtifactPath("validation", "worker_coding_cohort", cohortRunId);
  fs.mkdirSync(outDir, { recursive: true });
  const jsonPath = path.join(outDir, "worker_coding_cohort_result.json");
  const mdPath = path.join(outDir, "worker_coding_cohort_report.md");
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), "utf8");
  fs.writeFileSync(mdPath, makeReportMd(report), "utf8");

  console.log("# Worker-Coding Cohort Execution");
  console.log(`- json: ${normalizeRelPath(path.relative(process.cwd(), jsonPath))}`);
  console.log(`- md: ${normalizeRelPath(path.relative(process.cwd(), mdPath))}`);
  console.log(`- pass_count: ${report.summary.pass_count}`);
  console.log(`- fail_count: ${report.summary.fail_count}`);
  console.log(`- partial_count: ${report.summary.partial_count}`);

  if (strict && report.summary.fail_count > 0) {
    process.exitCode = 1;
  }

  return {
    report,
    jsonPath,
    mdPath,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(`[worker-coding-cohort] failed: ${err.message || String(err)}`);
    process.exit(1);
  });
}
