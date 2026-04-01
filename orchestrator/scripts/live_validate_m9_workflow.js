import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

import { resolveOrchestratorArtifactPath } from "./_paths.js";

function arg(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function getJson(url, init = {}) {
  const res = await fetch(url, init);
  const text = await res.text();
  let json;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = { raw: text };
  }
  return { ok: res.ok, status: res.status, json };
}

function isReleasePackTerminalState(state) {
  const run = state?.run || {};
  const steps = Array.isArray(state?.steps) ? state.steps : [];
  if (!steps.length) return false;
  const releasePack = steps.find((step) => String(step?.step_id || "") === "release_pack");
  if (String(releasePack?.status || "") !== "succeeded") return false;
  const blockingSteps = steps.filter((step) => {
    const stepId = String(step?.step_id || "");
    const status = String(step?.status || "");
    if (!stepId || ["succeeded", "failed", "partial_failure"].includes(status)) return false;
    return !(stepId === "deploy_preview" && status === "queued");
  });
  if (blockingSteps.length > 0) return false;
  return ["running", "succeeded"].includes(String(run?.status || ""));
}

async function checkHealth(baseUrl) {
  const res = await fetch(`${baseUrl}/health`);
  const text = await res.text();
  assert(res.ok && String(text).trim() === "ok", `health failed: ${res.status} ${text}`);
}

async function approvePendingTasks(baseUrl, runId, approvalToken, approvedTaskIds) {
  if (!approvalToken) return;
  const pending = await getJson(`${baseUrl}/approvals/pending?limit=100`);
  if (!pending.ok || !Array.isArray(pending.json?.tasks)) return;
  for (const task of pending.json.tasks) {
    const taskId = String(task?.task_id || "");
    const taskRunId = String(task?.run_id || "");
    if (!taskId || taskRunId !== String(runId || "") || approvedTaskIds.has(taskId)) continue;
    const approveRes = await getJson(`${baseUrl}/tasks/${encodeURIComponent(taskId)}/approve`, {
      method: "POST",
      headers: {
        "X-Approval-Token": approvalToken,
      },
    });
    assert(approveRes.ok, `approve failed for task ${taskId}: ${approveRes.status}`);
    approvedTaskIds.add(taskId);
  }
}

async function pollWorkflow(baseUrl, workflowRunId, { runId = "", approvalToken = "", timeoutMs = 240000, intervalMs = 4000 } = {}) {
  const started = Date.now();
  const approvedTaskIds = new Set();
  while (Date.now() - started < timeoutMs) {
    await approvePendingTasks(baseUrl, runId, approvalToken, approvedTaskIds);
    const state = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (state.ok) {
      const status = String(state.json?.run?.status || "");
      if (["succeeded", "failed", "partial_failure"].includes(status)) {
        state.json.auto_approved_task_ids = Array.from(approvedTaskIds);
        return state.json;
      }
      if (isReleasePackTerminalState(state.json)) {
        state.json.auto_approved_task_ids = Array.from(approvedTaskIds);
        state.json.validation_notice = "release_pack_succeeded_deploy_preview_still_queued";
        return state.json;
      }
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting workflow ${workflowRunId}`);
}

async function waitForWorkflowCompletion(baseUrl, workflowRunId, { timeoutMs = 30000, intervalMs = 3000 } = {}) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const state = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}`);
    if (state.ok) {
      const status = String(state.json?.run?.status || "");
      if (["succeeded", "failed", "partial_failure"].includes(status)) {
        return state.json;
      }
    }
    await sleep(intervalMs);
  }
  return null;
}

function buildWorkflowPayload() {
  return {
    workflow_id: "coding_team_v0",
    project_type: "webapp_crm",
    input: {
      goal: "Build a minimal CRM web app with customer list, detail page, and add/edit form. Keep changes reviewable and include required artifacts.",
      provider: "opencode",
      model: "minimax-coding-plan/MiniMax-M2.7",
      fast_mode: true,
      max_runtime_s: 180,
      step_payloads: {
        pm_spec: {
          target_paths: ["sandbox/live_validation/pm_spec.txt"],
          opencode_command: ["mock-inline-autofix", "sandbox/live_validation/pm_spec.txt", "{{task_prompt}}"],
        },
        arch_design: {
          target_paths: ["sandbox/live_validation/arch_design.txt"],
          opencode_command: ["mock-inline-autofix", "sandbox/live_validation/arch_design.txt", "{{task_prompt}}"],
        },
        impl_be: {
          target_paths: ["sandbox/crm_site/server.js"],
          opencode_command: ["node", "/workspace/orchestrator/scripts/live_validate_mock_crm_impl.js", "impl_be", "{{task_prompt}}"],
          max_attempts: 2,
          same_error_repeat_limit: 2,
          wall_clock_timeout_s: 300,
        },
        impl_fe: {
          target_paths: ["sandbox/crm_site/app.js"],
          opencode_command: ["node", "/workspace/orchestrator/scripts/live_validate_mock_crm_impl.js", "impl_fe", "{{task_prompt}}"],
          max_attempts: 2,
          same_error_repeat_limit: 2,
          wall_clock_timeout_s: 300,
        },
        qa_verify: {
          command: "node --version",
        },
        release_pack: {
          target_paths: ["sandbox/live_validation/release_pack.txt"],
          opencode_command: ["mock-inline-autofix", "sandbox/live_validation/release_pack.txt", "{{task_prompt}}"],
        },
      },
    },
  };
}

function validateManifest(manifest) {
  assert(Array.isArray(manifest?.coding_execution_evidence), "coding_execution_evidence missing");
  const be = manifest.coding_execution_evidence.find((item) => item.step_id === "impl_be");
  const fe = manifest.coding_execution_evidence.find((item) => item.step_id === "impl_fe");
  assert(be, "impl_be evidence missing");
  assert(fe, "impl_fe evidence missing");
  assert(be.verification_checked === true, "impl_be verification not recorded");
  assert(fe.verification_checked === true, "impl_fe verification not recorded");
  assert(Number(be?.retry_summary?.attempts_used || 0) >= 1, "impl_be execution evidence missing");
  assert(Number(fe?.retry_summary?.attempts_used || 0) >= 1, "impl_fe execution evidence missing");
  assert(be.test_log_path, "impl_be test_log_path missing");
  assert(fe.test_log_path, "impl_fe test_log_path missing");
  assert(be.prompt_contract_path, "impl_be prompt_contract_path missing");
  assert(fe.prompt_contract_path, "impl_fe prompt_contract_path missing");

  const runtime = manifest?.runtime_evidence_summary && typeof manifest.runtime_evidence_summary === "object"
    ? manifest.runtime_evidence_summary
    : null;
  assert(runtime, "runtime_evidence_summary missing");
  assert(runtime.smoke_present === true, "smoke_present not recorded");
  assert(String(runtime.smoke_verdict || "") === "pass", `smoke_verdict expected pass, got ${String(runtime?.smoke_verdict || "none")}`);
  assert(Number(runtime.smoke_root_status || 0) === 200, `smoke_root_status expected 200, got ${String(runtime?.smoke_root_status || "none")}`);
  const apiStatus = Number(runtime?.smoke_api_status || 0);
  assert(apiStatus === 200 || apiStatus === 0, `smoke_api_status expected 200 or 0, got ${String(runtime?.smoke_api_status || "none")}`);
  const superpowersConfigured = Number(runtime?.superpowers_configured_steps || 0);
  const superpowersAvailable = Number(runtime?.superpowers_available_steps || 0);
  assert(superpowersConfigured >= 0, "superpowers_configured_steps invalid");
  assert(superpowersAvailable >= 0, "superpowers_available_steps invalid");
}

function parseStepResultJson(step) {
  try {
    return JSON.parse(String(step?.result_json || "{}"));
  } catch {
    return {};
  }
}

function validateTerminalStepsFallback(steps = [], artifactPaths = []) {
  const safeSteps = Array.isArray(steps) ? steps : [];
  const safePaths = Array.isArray(artifactPaths) ? artifactPaths : [];
  const requiredPaths = [
    "impl/be_changes/package.json",
    "impl/be_changes/server.js",
    "impl/fe_changes/public/index.html",
    "impl/fe_changes/public/app.js",
    "release/release_notes.md",
    "release/artifact_manifest.json",
    "release/README.md",
    "release/start.sh",
    "smoke/smoke_result.json",
    "verify/qa_report.json",
  ];
  for (const relPath of requiredPaths) {
    assert(
      safePaths.some((item) => String(item || "").replace(/\\/g, "/").endsWith(relPath)),
      `required artifact missing: ${relPath}`,
    );
  }

  const beStep = safeSteps.find((step) => String(step?.step_id || "") === "impl_be");
  const feStep = safeSteps.find((step) => String(step?.step_id || "") === "impl_fe");
  assert(String(beStep?.status || "") === "succeeded", "impl_be step did not succeed");
  assert(String(feStep?.status || "") === "succeeded", "impl_fe step did not succeed");

  const beResult = parseStepResultJson(beStep);
  const feResult = parseStepResultJson(feStep);
  const beVerified = beResult?.verification?.checked === true || beResult?.impl_validation?.checked === true;
  const feVerified = feResult?.verification?.checked === true || feResult?.impl_validation?.checked === true;
  assert(beVerified, "impl_be verification not recorded");
  assert(feVerified, "impl_fe verification not recorded");
  assert(beResult?.artifacts?.prompt_contract, "impl_be prompt_contract missing");
  assert(feResult?.artifacts?.prompt_contract, "impl_fe prompt_contract missing");
  assert(beResult?.artifacts?.test_log || beResult?.verification?.logPath, "impl_be test evidence missing");
  assert(feResult?.artifacts?.test_log || feResult?.verification?.logPath, "impl_fe test evidence missing");
}

function validateStepModelRouting(steps = []) {
  const safeSteps = Array.isArray(steps) ? steps : [];
  const releasePack = safeSteps.find((step) => String(step?.step_id || "") === "release_pack");
  assert(String(releasePack?.status || "") === "succeeded", "release_pack step did not succeed");
  const releasePackResult = parseStepResultJson(releasePack);
  const releasePackLane = String(releasePackResult?.execution_lane || "");
  const releasePackModel = String(releasePackResult?.model_used || "");
  assert(releasePackLane === "primary_minimax_lane", `release_pack execution_lane expected primary_minimax_lane, got ${releasePackLane || "none"}`);
  assert(
    /qwen-plus-2025-04-28/i.test(releasePackModel),
    `release_pack model_used expected qwen-plus-2025-04-28, got ${releasePackModel || "none"}`,
  );
}

function summarizePreviewDeployStatus(steps = [], releasePackBypass = false) {
  const safeSteps = Array.isArray(steps) ? steps : [];
  const deployPreview = safeSteps.find((step) => String(step?.step_id || "") === "deploy_preview");
  const deployStatus = String(deployPreview?.status || "");
  if (deployStatus === "succeeded") return "completed";
  if (releasePackBypass) return "queued_without_preview_worker";
  if (deployStatus) return deployStatus;
  return "unknown";
}

function resolveManifestPath({ workspaceRoot, runId, packValidation }) {
  const manifestPathHint = String(packValidation?.validation?.manifest_path || "").trim();
  const workspaceCandidates = [path.resolve(workspaceRoot), path.resolve(workspaceRoot, "..")];
  if (manifestPathHint) {
    const normalized = manifestPathHint.replace(/^\/workspace/i, "").replace(/^\/+/, "");
    for (const candidateRoot of workspaceCandidates) {
      const hintedPath = path.resolve(candidateRoot, normalized);
      if (fs.existsSync(hintedPath)) return hintedPath;
    }
  }
  const candidates = workspaceCandidates.flatMap((candidateRoot) => ([
    path.join(candidateRoot, "artifacts", "release", runId, "meta", "run_manifest.json"),
    path.join(candidateRoot, "artifacts", "release", runId, "release", "run_manifest.json"),
  ]));
  return candidates.find((item) => fs.existsSync(item)) || candidates[0];
}

function validateRunSummary(summaryText) {
  const text = String(summaryText || "");
  assert(/## Runtime Evidence/.test(text), "run_summary Runtime Evidence section missing");
  assert(/smoke_verdict:\s*pass/i.test(text), "run_summary smoke_verdict missing");
  assert(/smoke_root_status:\s*200/i.test(text), "run_summary smoke_root_status missing");
  assert(/superpowers_configured_steps:\s*\d+/i.test(text), "run_summary superpowers_configured_steps missing");
  assert(/superpowers_available_steps:\s*\d+/i.test(text), "run_summary superpowers_available_steps missing");
}

function resolveSummaryPath(manifestPath, runId = "") {
  const safeManifestPath = String(manifestPath || "").trim();
  if (safeManifestPath) {
    const candidate = path.join(path.dirname(path.dirname(safeManifestPath)), "summary", "run_summary.md");
    if (fs.existsSync(candidate)) return candidate;
  }
  if (runId) {
    const workspaceRoot = path.resolve(process.cwd());
    const workspaceCandidates = [path.resolve(workspaceRoot), path.resolve(workspaceRoot, "..")];
    const candidates = workspaceCandidates.map((candidateRoot) =>
      path.join(candidateRoot, "artifacts", "release", runId, "summary", "run_summary.md")
    );
    return candidates.find((item) => fs.existsSync(item)) || candidates[0];
  }
  return "";
}
export async function main(options = {}) {
  const baseUrl = String(options.baseUrl || arg("base-url", process.env.ORCH_BASE_URL || "http://localhost:3000")).replace(/\/+$/, "");
  const approvalToken = String(options.approvalToken || arg("approval-token", process.env.APPROVAL_TOKEN || "dev-approval-token"));
  const timeoutMs = Math.max(120000, Number(options.timeoutMs || arg("timeout-ms", "420000")));
  const workspaceRoot = path.resolve(process.cwd());
  const report = {
    generated_at: new Date().toISOString(),
    base_url: baseUrl,
    timeout_ms: timeoutMs,
    overall: "pass",
    workflow_payload: buildWorkflowPayload(),
  };

  try {
    await checkHealth(baseUrl);
    const startRes = await getJson(`${baseUrl}/workflow-runs/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(report.workflow_payload),
    });
    assert(startRes.ok, `workflow start failed: ${startRes.status}`);
    report.start = startRes.json;

    const workflowRunId = String(startRes.json?.workflow_run_id || "");
    const runId = String(startRes.json?.run_id || "");
    assert(workflowRunId, "workflow_run_id missing");
    assert(runId, "run_id missing");

    report.terminal = await pollWorkflow(baseUrl, workflowRunId, {
      runId,
      approvalToken,
      timeoutMs,
      intervalMs: 4000,
    });
    const terminalStatus = String(report.terminal?.run?.status || "");
    const validationNotice = String(report.terminal?.validation_notice || "");
    const releasePackBypass = validationNotice === "release_pack_succeeded_deploy_preview_still_queued";
    if (releasePackBypass) {
      const settled = await waitForWorkflowCompletion(baseUrl, workflowRunId, {
        timeoutMs: 30000,
        intervalMs: 3000,
      });
      if (settled?.run) {
        report.terminal = settled;
      }
    }
    const finalTerminalStatus = String(report.terminal?.run?.status || "");
    const finalValidationNotice = String(report.terminal?.validation_notice || "");
    const finalReleasePackBypass = finalValidationNotice === "release_pack_succeeded_deploy_preview_still_queued";
    assert(
      finalTerminalStatus === "succeeded" || finalReleasePackBypass,
      `workflow ended with status=${finalTerminalStatus}`,
    );

    const artifactsRes = await getJson(`${baseUrl}/runs/${encodeURIComponent(runId)}/artifacts`);
    assert(artifactsRes.ok, `artifacts query failed: ${artifactsRes.status}`);
    report.artifacts = artifactsRes.json;

    const packValidationRes = await getJson(`${baseUrl}/workflow-runs/${encodeURIComponent(workflowRunId)}/validate-pack`);
    assert(packValidationRes.ok, `validate-pack failed: ${packValidationRes.status}`);
    report.pack_validation = packValidationRes.json;
    report.preview_deploy_status = summarizePreviewDeployStatus(report.terminal?.steps || [], finalReleasePackBypass);

    const manifestPath = resolveManifestPath({
      workspaceRoot,
      runId,
      packValidation: packValidationRes.json,
    });
    if (fs.existsSync(manifestPath)) {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
      const summaryPath = resolveSummaryPath(manifestPath, runId);
      report.run_manifest_path = manifestPath.replace(/\\/g, "/");
      report.run_summary_path = summaryPath ? summaryPath.replace(/\\/g, "/") : null;
      report.coding_execution_summary = manifest.coding_execution_summary || null;
      report.runtime_evidence_summary = manifest.runtime_evidence_summary || null;
      validateManifest(manifest);
      if (summaryPath && fs.existsSync(summaryPath)) {
        validateRunSummary(fs.readFileSync(summaryPath, "utf8"));
      } else {
        throw new Error(`run summary missing: ${summaryPath || "unknown"}`);
      }
    } else {
      assert(finalReleasePackBypass, `run manifest missing: ${manifestPath}`);
      const artifactPaths = Array.isArray(artifactsRes.json?.release_files)
        ? artifactsRes.json.release_files.map((item) => String(item?.path || ""))
        : [];
      validateTerminalStepsFallback(report.terminal?.steps || [], artifactPaths);
      report.run_manifest_path = null;
      report.coding_execution_summary = "validated_via_terminal_steps_and_release_artifacts";
    }
    validateStepModelRouting(report.terminal?.steps || []);
  } catch (err) {
    report.overall = "fail";
    report.error = err.message || String(err);
  }

  const outDir = resolveOrchestratorArtifactPath("canary", "live_m9_workflow");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "live_m9_workflow_report.json");
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");

  console.log("# Live M9 Workflow Validation");
  console.log(`- report: ${outPath.replace(/\\/g, "/")}`);
  console.log(`- overall: ${report.overall}`);
  if (report.error) console.log(`- error: ${report.error}`);
  if (report.overall !== "pass") {
    throw new Error(report.error || "live M9 workflow validation failed");
  }
  return {
    reportPath: outPath.replace(/\\/g, "/"),
    report,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(`[live-m9-workflow] failed: ${err.message || String(err)}`);
    process.exit(1);
  });
}

