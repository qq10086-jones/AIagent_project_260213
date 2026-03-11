import fs from "fs";
import path from "path";

import {
  ORCHESTRATOR_ROOT,
  REPO_ROOT,
  resolveRepoPath,
} from "./_paths.js";

function normalizePath(filePath) {
  return String(filePath || "").replace(/\\/g, "/");
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function exists(filePath) {
  return fs.existsSync(filePath);
}

function sameJson(leftPath, rightPath) {
  if (!exists(leftPath) || !exists(rightPath)) return false;
  return JSON.stringify(readJson(leftPath)) === JSON.stringify(readJson(rightPath));
}

function main() {
  const composePath = resolveRepoPath("infra", "docker-compose.yml");
  const composeText = readText(composePath);
  const resolvedWorkspaceRoot = path.resolve(process.env.WORKSPACE_ROOT || REPO_ROOT);

  const report = {
    generated_at: new Date().toISOString(),
    overall: "pass",
    startup_profiles: {
      local_manual: {
        cwd_expected: normalizePath(ORCHESTRATOR_ROOT),
        workspace_root_resolved: normalizePath(resolvedWorkspaceRoot),
        governance_config_root: normalizePath(resolveRepoPath("configs")),
        orchestrator_config_root: normalizePath(path.join(ORCHESTRATOR_ROOT, "configs")),
      },
      docker_compose: {
        workspace_root: "/workspace",
        app_config_root: "/app/configs",
      },
    },
    required_files: [],
    compose_mount_checks: [],
    governance_mirror_checks: [],
  };

  const requiredFiles = [
    { id: "runtime_defaults", path: resolveRepoPath("configs", "runtime", "runtime_defaults.json") },
    { id: "llm_providers", path: path.join(ORCHESTRATOR_ROOT, "configs", "llm_providers.json") },
    { id: "llm_role_policy", path: path.join(ORCHESTRATOR_ROOT, "configs", "llm_role_policy.json") },
    { id: "context_budget_policy", path: path.join(ORCHESTRATOR_ROOT, "configs", "context_budget_policy.json") },
    { id: "production_parallel_rollout_root", path: resolveRepoPath("configs", "production_parallel_rollout.json") },
    { id: "parallel_exposure_policy_root", path: resolveRepoPath("configs", "parallel_exposure_policy.json") },
    { id: "m7_exposure_cohorts_root", path: resolveRepoPath("configs", "m7_exposure_cohorts.json") },
  ];

  report.required_files = requiredFiles.map((item) => ({
    ...item,
    path: normalizePath(item.path),
    exists: exists(item.path),
  }));

  const composeChecks = [
    { id: "workspace_mount", pattern: "../..:/workspace", expected: false },
    { id: "workspace_mount_exact", pattern: "- ..:/workspace", expected: true },
    { id: "runtime_defaults_mount", pattern: "../configs/runtime:/app/configs/runtime:ro", expected: true },
    { id: "llm_providers_mount", pattern: "../orchestrator/configs/llm_providers.json:/app/configs/llm_providers.json:ro", expected: true },
    { id: "llm_role_policy_mount", pattern: "../orchestrator/configs/llm_role_policy.json:/app/configs/llm_role_policy.json:ro", expected: true },
    { id: "context_budget_policy_mount", pattern: "../orchestrator/configs/context_budget_policy.json:/app/configs/context_budget_policy.json:ro", expected: true },
    { id: "rollout_mount_root", pattern: "../configs/production_parallel_rollout.json:/app/configs/production_parallel_rollout.json:ro", expected: true },
    { id: "parallel_policy_mount_root", pattern: "../configs/parallel_exposure_policy.json:/app/configs/parallel_exposure_policy.json:ro", expected: true },
    { id: "m7_cohort_mount_root", pattern: "../configs/m7_exposure_cohorts.json:/app/configs/m7_exposure_cohorts.json:ro", expected: true },
  ];

  report.compose_mount_checks = composeChecks.map((item) => {
    const present = composeText.includes(item.pattern);
    return {
      id: item.id,
      pattern: item.pattern,
      ok: item.expected ? present : !present,
    };
  });

  const governanceMirrors = [
    {
      id: "production_parallel_rollout",
      root_path: resolveRepoPath("configs", "production_parallel_rollout.json"),
      orchestrator_path: path.join(ORCHESTRATOR_ROOT, "configs", "production_parallel_rollout.json"),
    },
    {
      id: "parallel_exposure_policy",
      root_path: resolveRepoPath("configs", "parallel_exposure_policy.json"),
      orchestrator_path: path.join(ORCHESTRATOR_ROOT, "configs", "parallel_exposure_policy.json"),
    },
    {
      id: "m7_exposure_cohorts",
      root_path: resolveRepoPath("configs", "m7_exposure_cohorts.json"),
      orchestrator_path: path.join(ORCHESTRATOR_ROOT, "configs", "m7_exposure_cohorts.json"),
    },
  ];

  report.governance_mirror_checks = governanceMirrors.map((item) => ({
    id: item.id,
    root_path: normalizePath(item.root_path),
    orchestrator_path: normalizePath(item.orchestrator_path),
    root_exists: exists(item.root_path),
    orchestrator_exists: exists(item.orchestrator_path),
    in_sync: sameJson(item.root_path, item.orchestrator_path),
  }));

  const missingRequired = report.required_files.filter((item) => !item.exists);
  const failedCompose = report.compose_mount_checks.filter((item) => !item.ok);

  if (missingRequired.length > 0 || failedCompose.length > 0) {
    report.overall = "fail";
  }

  const outDir = path.join(ORCHESTRATOR_ROOT, "artifacts", "validation", "runtime_boot_sources");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, "runtime_boot_sources_report.json");
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), "utf8");

  console.log("# Runtime Boot Sources");
  console.log(`- report: ${normalizePath(outPath)}`);
  console.log(`- overall: ${report.overall}`);
  console.log(`- missing_required: ${missingRequired.length}`);
  console.log(`- failed_compose_checks: ${failedCompose.length}`);
  console.log(`- governance_mirror_drift: ${report.governance_mirror_checks.filter((item) => !item.in_sync).length}`);

  if (report.overall !== "pass") {
    throw new Error("runtime boot source validation failed");
  }

  return {
    reportPath: normalizePath(outPath),
    report,
  };
}

try {
  main();
} catch (err) {
  console.error(err?.message || err);
  process.exit(1);
}
