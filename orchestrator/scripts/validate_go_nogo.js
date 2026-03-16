import fs from "fs";
import path from "path";
import { validateArtifactPack } from "../src/artifact_pack_validator.js";
import { getExpectedWorkflowStepCount } from "../src/domain/workflow_artifact_audit.js";
import { getDefaultRegistryPath, loadRegistryOrThrow } from "../src/registry.js";
import { getDefaultWorkspaceRoot } from "./_paths.js";

function argValue(name, fallback = "") {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && process.argv[idx + 1]) return String(process.argv[idx + 1]);
  return fallback;
}

function boolEnv(name, fallback = false) {
  const raw = String(process.env[name] || "").trim().toLowerCase();
  if (!raw) return fallback;
  return ["1", "true", "yes", "on"].includes(raw);
}

function safeReadJson(p) {
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return null;
  }
}

function findRegistryPath() {
  const envPath = String(process.env.REGISTRY_PATH || "").trim();
  if (envPath) return envPath;
  return getDefaultRegistryPath();
}

function getReleaseRoot({ releaseRootArg = "", runId = "", workspaceRoot = "" }) {
  if (releaseRootArg) return path.resolve(releaseRootArg);
  if (runId) return path.resolve(workspaceRoot, "artifacts", "release", runId);
  return "";
}

function findRunByWorkflowRunId({ workflowRunId = "", workspaceRoot = "" }) {
  const releaseBase = path.resolve(workspaceRoot, "artifacts", "release");
  if (!workflowRunId || !fs.existsSync(releaseBase)) return null;
  const dirs = fs.readdirSync(releaseBase, { withFileTypes: true }).filter((d) => d.isDirectory());
  for (const d of dirs) {
    const manifestPath = path.join(releaseBase, d.name, "meta", "run_manifest.json");
    if (!fs.existsSync(manifestPath)) continue;
    const manifest = safeReadJson(manifestPath);
    if (!manifest) continue;
    if (String(manifest.workflow_run_id || "") === String(workflowRunId)) {
      return {
        runId: String(manifest.run_id || d.name),
        releaseRoot: path.join(releaseBase, d.name),
        manifest,
      };
    }
  }
  return null;
}

function printResult(result) {
  console.log(`# Go/No-Go Validation`);
  console.log(`- verdict: ${result.verdict}`);
  console.log(`- workflow_run_id: ${result.workflow_run_id}`);
  console.log(`- run_id: ${result.run_id}`);
  console.log(`- release_root: ${result.release_root}`);
  console.log(`- checks_passed: ${result.passed_checks}/${result.total_checks}`);
  console.log(`- checks_failed: ${result.failed_checks}`);
  for (const item of result.checks) {
    console.log(`  - [${item.pass ? "PASS" : "FAIL"}] ${item.name}: ${item.detail}`);
  }
  if (result.reasons.length > 0) {
    console.log(`- reasons:`);
    for (const r of result.reasons) console.log(`  - ${r}`);
  }
}

function main() {
  const workspaceRoot = path.resolve(argValue("workspace-root", getDefaultWorkspaceRoot()));
  const runIdArg = argValue("run-id", "").trim();
  const workflowRunId = argValue("workflow-run-id", "").trim();
  const releaseRootArg = argValue("release-root", "").trim();
  const strict = boolEnv("GONOGO_STRICT", true);
  const expectedStepsArg = Number(argValue("expected-steps", "0"));

  let runId = runIdArg;
  let releaseRoot = getReleaseRoot({ releaseRootArg, runId, workspaceRoot });
  let manifest = null;

  if (!releaseRoot && workflowRunId) {
    const found = findRunByWorkflowRunId({ workflowRunId, workspaceRoot });
    if (!found) {
      console.error(`[gonogo] workflow_run_id not found in release artifacts: ${workflowRunId}`);
      process.exit(1);
    }
    runId = found.runId;
    releaseRoot = found.releaseRoot;
    manifest = found.manifest;
  }

  if (!releaseRoot) {
    console.error("[gonogo] missing release root. provide --run-id, --workflow-run-id, or --release-root");
    process.exit(1);
  }

  const manifestPath = path.join(releaseRoot, "meta", "run_manifest.json");
  const summaryPath = path.join(releaseRoot, "summary", "run_summary.md");
  const canaryJsonPath = path.join(releaseRoot, "qa", "strict_canary_report.json");
  const canaryMdPath = path.join(releaseRoot, "qa", "strict_canary_report.md");

  if (!manifest) manifest = safeReadJson(manifestPath);
  if (!manifest) {
    console.error(`[gonogo] run manifest invalid or missing: ${manifestPath}`);
    process.exit(1);
  }

  const registryPath = findRegistryPath();
  if (!registryPath || !fs.existsSync(registryPath)) {
    console.error("[gonogo] registry not found");
    process.exit(1);
  }
  const registry = loadRegistryOrThrow(registryPath);

  const run = {
    workflow_run_id: String(manifest.workflow_run_id || workflowRunId || ""),
    run_id: String(manifest.run_id || runId || ""),
    workflow_id: String(manifest.workflow_id || ""),
    project_type: String(manifest.project_type || ""),
  };
  const steps = Array.isArray(manifest.steps) ? manifest.steps : [];
  const checkpoints = Array.isArray(manifest.checkpoints) ? manifest.checkpoints : [];
  const canary = safeReadJson(canaryJsonPath);
  const validator = validateArtifactPack({
    run,
    steps,
    checkpoints,
    manifestPath,
    summaryPath,
    registry,
  });

  const expectedSteps = Number.isFinite(expectedStepsArg) && expectedStepsArg > 0
    ? expectedStepsArg
    : getExpectedWorkflowStepCount(registry, run.workflow_id);
  const acceptanceStep = steps.find((s) => String(s.gate_name || "") === "acceptance") || steps.find((s) => String(s.step_id || "") === "qa_verify");

  const checks = [];
  checks.push({
    name: "artifact_pack_validator",
    pass: Boolean(validator.ok),
    detail: validator.ok ? "validator passed" : `validator failed: ${(validator.reasons || []).join("; ")}`,
  });
  checks.push({
    name: "release_pack_files",
    pass: fs.existsSync(manifestPath) && fs.existsSync(summaryPath) && fs.existsSync(canaryJsonPath) && fs.existsSync(canaryMdPath),
    detail: `manifest=${fs.existsSync(manifestPath)} summary=${fs.existsSync(summaryPath)} canary_json=${fs.existsSync(canaryJsonPath)} canary_md=${fs.existsSync(canaryMdPath)}`,
  });
  checks.push({
    name: "workflow_status",
    pass: String(manifest.status || "") === "succeeded",
    detail: `manifest.status=${String(manifest.status || "")}`,
  });
  checks.push({
    name: "step_success",
    pass: steps.length > 0 && steps.every((s) => String(s.status || "") === "succeeded"),
    detail: `succeeded=${steps.filter((s) => String(s.status || "") === "succeeded").length}/${steps.length}`,
  });
  if (expectedSteps > 0) {
    checks.push({
      name: "step_count",
      pass: steps.length === expectedSteps,
      detail: `steps=${steps.length} expected=${expectedSteps}`,
    });
  }
  checks.push({
    name: "acceptance_gate",
    pass: !!acceptanceStep && String(acceptanceStep.status || "") === "succeeded",
    detail: acceptanceStep
      ? `step=${String(acceptanceStep.step_id || "")} status=${String(acceptanceStep.status || "")}`
      : "acceptance step missing",
  });
  checks.push({
    name: "strict_canary_verdict",
    pass: !!canary && String(canary.verdict || "") === "pass",
    detail: canary ? `verdict=${String(canary.verdict || "")}` : "strict canary report missing/invalid",
  });
  checks.push({
    name: "strict_canary_missing_artifacts",
    pass: !!canary && Number(canary?.totals?.missing_artifacts_total || 0) === 0,
    detail: canary ? `missing_artifacts_total=${Number(canary?.totals?.missing_artifacts_total || 0)}` : "strict canary report missing/invalid",
  });

  const reasons = [];
  for (const c of checks) {
    if (!c.pass) reasons.push(`${c.name}: ${c.detail}`);
  }
  if (!strict) {
    // In non-strict mode, allow pass if only step-count check fails.
    const onlyStepCountFail = reasons.every((r) => r.startsWith("step_count:"));
    if (onlyStepCountFail) reasons.length = 0;
  }

  const result = {
    verdict: reasons.length === 0 ? "GO" : "NO_GO",
    workflow_run_id: run.workflow_run_id,
    run_id: run.run_id,
    release_root: releaseRoot.replace(/\\/g, "/"),
    total_checks: checks.length,
    passed_checks: checks.filter((c) => c.pass).length,
    failed_checks: checks.filter((c) => !c.pass).length,
    checks,
    reasons,
    strict,
    generated_at: new Date().toISOString(),
  };

  const outPath = path.join(releaseRoot, "qa", "go_no_go_result.json");
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2), "utf8");
  printResult(result);
  console.log(`- output_json: ${outPath.replace(/\\/g, "/")}`);
  if (result.verdict !== "GO") process.exit(1);
}

main();
