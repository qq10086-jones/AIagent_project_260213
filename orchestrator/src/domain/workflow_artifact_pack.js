/**
 * workflow_artifact_pack.js
 *
 * Artifact pack generation, validation, and MinIO archival.
 * Extracted from workflow_engine.js as part of WS-11-04 decomposition.
 */

import fs from "fs";
import path from "path";
import { validateArtifactPack } from "../artifact_pack_validator.js";
import { buildFinalResultPackage, validateFinalResultPackage } from "../final_result_packager.js";
import {
  getWorkflowRunById,
  listWorkflowCheckpoints,
} from "../data/workflow_repository.js";
import {
  buildReleasePackPaths,
  buildStepArtifactsFromCheckpoints,
  buildGoNoGoResult,
  buildStrictCanaryMarkdown,
  buildStrictCanaryReport,
  ensureDir,
  getExpectedWorkflowStepCount,
  inferProjectArtifactCoverage,
  writeJsonFile,
} from "./workflow_artifact_audit.js";
import { parseJsonSafe } from "./workflow_runner.js";

/**
 * @param {{ pool, workspaceRoot, registry, archiveReleasePackToMinio,
 *           indexReleasePackToDb, minioBucket, recordEvent, getSteps }} deps
 */
export function createArtifactPackService({
  pool,
  workspaceRoot,
  registry,
  archiveReleasePackToMinio,
  indexReleasePackToDb,
  minioBucket,
  recordEvent,
  getSteps,
}) {
  function collectContextBudgetReports(steps = []) {
    const reports = [];
    for (const step of steps || []) {
      const result = parseJsonSafe(step?.result_json, {});
      const reportPath = String(result?.context_budget_report_path || "").trim().replace(/\\/g, "/");
      const report = result?.context_budget_report && typeof result.context_budget_report === "object"
        ? result.context_budget_report
        : null;
      reports.push({
        step_index: Number(step?.step_index),
        step_id: String(step?.step_id || ""),
        role_name: String(step?.role_name || ""),
        status: report?.status || null,
        report_path: reportPath,
        threshold_source: report?.threshold_source || null,
      });
    }
    const statusCounts = { ok: 0, warning: 0, overflow_risk: 0, missing: 0 };
    for (const item of reports) {
      if (!item.report_path || !item.status) statusCounts.missing += 1;
      else if (Object.prototype.hasOwnProperty.call(statusCounts, item.status)) statusCounts[item.status] += 1;
    }
    return {
      reports,
      summary: {
        total_steps: reports.length,
        ok: statusCounts.ok,
        warning: statusCounts.warning,
        overflow_risk: statusCounts.overflow_risk,
        missing: statusCounts.missing,
      },
    };
  }

  function collectContextArtifacts(steps = []) {
    return (steps || []).map((step) => {
      const result = parseJsonSafe(step?.result_json, {});
      return {
        step_index: Number(step?.step_index),
        step_id: String(step?.step_id || ""),
        context_packet_path: String(result?.context_packet_path || "").trim().replace(/\\/g, "/") || null,
        repo_map_path: String(result?.repo_map_path || "").trim().replace(/\\/g, "/") || null,
      };
    });
  }

  function collectCodingExecutionEvidence(steps = []) {
    const items = (steps || []).map((step) => {
      const result = parseJsonSafe(step?.result_json, {});
      const diagnostics = result?.diagnostics && typeof result.diagnostics === "object"
        ? result.diagnostics
        : {};
      const retrySummary = diagnostics?.retry_summary && typeof diagnostics.retry_summary === "object"
        ? diagnostics.retry_summary
        : null;
      const finalFailureSummary = diagnostics?.final_failure_summary && typeof diagnostics.final_failure_summary === "object"
        ? diagnostics.final_failure_summary
        : null;
      const failureMemory = diagnostics?.coding_failure_memory && typeof diagnostics.coding_failure_memory === "object"
        ? diagnostics.coding_failure_memory
        : null;
      return {
        step_index: Number(step?.step_index),
        step_id: String(step?.step_id || ""),
        test_result: String(result?.test_result || "").trim() || null,
        test_log_path: String(result?.artifacts?.test_log || "").trim().replace(/\\/g, "/") || null,
        verification_checked: Boolean(diagnostics?.verification?.checked),
        verification_ok: diagnostics?.verification?.checked ? Boolean(diagnostics?.verification?.ok) : null,
        verification_command: String(diagnostics?.verification?.command || "").trim() || null,
        retry_summary: retrySummary,
        final_failure_summary: finalFailureSummary,
        prompt_contract_path: String(result?.artifacts?.prompt_contract || "").trim().replace(/\\/g, "/") || null,
        coding_failure_memory_latest_path: String(failureMemory?.latest_path || "").trim().replace(/\\/g, "/") || null,
        coding_failure_memory_jsonl_path: String(failureMemory?.jsonl_path || "").trim().replace(/\\/g, "/") || null,
      };
    });
    const summary = {
      total_steps: items.length,
      verification_checked: items.filter((item) => item.verification_checked).length,
      verification_passed: items.filter((item) => item.verification_ok === true).length,
      verification_failed: items.filter((item) => item.verification_ok === false).length,
      test_logs_present: items.filter((item) => item.test_log_path).length,
      retry_enabled_steps: items.filter((item) => Number(item?.retry_summary?.max_attempts || 1) > 1).length,
      retry_attempted_steps: items.filter((item) => Number(item?.retry_summary?.attempts_used || 1) > 1).length,
      final_failures_present: items.filter((item) => item.final_failure_summary).length,
      failure_memory_entries: items.filter((item) => item.coding_failure_memory_latest_path || item.coding_failure_memory_jsonl_path).length,
    };
    return { items, summary };
  }

  async function generateArtifactPack(run) {
    const steps = await getSteps(run.workflow_run_id);
    const checkpoints = await listWorkflowCheckpoints(pool, run.workflow_run_id);
    const reasons = [];

    if (!Array.isArray(steps) || steps.length === 0) reasons.push("no workflow steps found");
    if (steps.some((s) => String(s.status) !== "succeeded")) reasons.push("not all steps are succeeded");
    if (checkpoints.length === 0) reasons.push("no checkpoints generated");
    if (checkpoints.length < steps.length) reasons.push("checkpoint count lower than step count");

    const required = registry.project_types?.[run.project_type]?.required_artifacts || [];
    const coverage = inferProjectArtifactCoverage(steps);
    for (const req of required) {
      if (!coverage[req]) reasons.push(`required artifact missing: ${req}`);
    }

    const stepArtifacts = buildStepArtifactsFromCheckpoints(steps, checkpoints, parseJsonSafe);
    const contextBudget = collectContextBudgetReports(steps);
    const contextArtifacts = collectContextArtifacts(steps);
    const codingExecutionEvidence = collectCodingExecutionEvidence(steps);
    const releasePaths = buildReleasePackPaths(workspaceRoot, run);
    const { manifest_path: manifestPath, summary_path: summaryPath,
            strict_canary_json_path: canaryJsonPath, strict_canary_report_path: canaryMdPath,
            go_no_go_result_path: goNoGoJsonPath } = releasePaths;

    const canaryReport = buildStrictCanaryReport({ run, steps });
    const canaryJsonRel = path.relative(workspaceRoot, canaryJsonPath).replace(/\\/g, "/");
    const canaryMdRel = path.relative(workspaceRoot, canaryMdPath).replace(/\\/g, "/");
    if (String(canaryReport.verdict) !== "pass") reasons.push("strict canary report failed");

    const manifest = {
      workflow_run_id: run.workflow_run_id,
      run_id: run.run_id,
      workflow_id: run.workflow_id,
      project_type: run.project_type,
      status: reasons.length === 0 ? "succeeded" : "failed",
      generated_at: new Date().toISOString(),
      required_artifacts: required,
      artifact_coverage: coverage,
      step_artifacts: stepArtifacts,
      context_budget_reports: contextBudget.reports,
      context_budget_summary: contextBudget.summary,
      context_artifacts: contextArtifacts,
      coding_execution_evidence: codingExecutionEvidence.items,
      coding_execution_summary: codingExecutionEvidence.summary,
      steps: steps.map((s) => ({
        step_index: Number(s.step_index),
        step_id: s.step_id,
        role_name: s.role_name,
        tool_name: s.tool_name,
        gate_name: s.gate_name,
        task_id: s.task_id,
        status: s.status,
        checkpoint_id: s.checkpoint_id,
        error_code: s.error_code || null,
      })),
      checkpoints: checkpoints.map((c) => ({
        checkpoint_id: c.checkpoint_id,
        step_index: Number(c.step_index),
        step_id: c.step_id,
        task_id: c.task_id,
        workspace_hash: c.workspace_hash,
      })),
      strict_canary: {
        verdict: canaryReport.verdict,
        report_json: canaryJsonRel,
        report_md: canaryMdRel,
        failed_steps: Number(canaryReport?.totals?.failed_steps || 0),
        missing_artifacts_total: Number(canaryReport?.totals?.missing_artifacts_total || 0),
      },
      reasons,
    };

    try {
      writeJsonFile(canaryJsonPath, canaryReport);
      ensureDir(path.dirname(canaryMdPath));
      fs.writeFileSync(canaryMdPath, buildStrictCanaryMarkdown(canaryReport, canaryJsonRel), "utf8");
      writeJsonFile(manifestPath, manifest);
      ensureDir(path.dirname(summaryPath));
      const summaryLines = [
        `# Run Summary`, ``,
        `- run_id: ${run.run_id}`,
        `- workflow_run_id: ${run.workflow_run_id}`,
        `- workflow_id: ${run.workflow_id}`,
        `- project_type: ${run.project_type}`,
        `- status: ${manifest.status}`,
        `- strict_canary_verdict: ${String(canaryReport.verdict || "").toUpperCase()}`,
        `- generated_at: ${manifest.generated_at}`, ``,
        `## Context Budget`,
        `- total_steps: ${Number(contextBudget.summary.total_steps || 0)}`,
        `- ok: ${Number(contextBudget.summary.ok || 0)}`,
        `- warning: ${Number(contextBudget.summary.warning || 0)}`,
        `- overflow_risk: ${Number(contextBudget.summary.overflow_risk || 0)}`,
        `- missing: ${Number(contextBudget.summary.missing || 0)}`, ``,
        `## Context Artifacts`,
        `- context_packets: ${contextArtifacts.filter((item) => item.context_packet_path).length}`,
        `- repo_maps: ${contextArtifacts.filter((item) => item.repo_map_path).length}`, ``,
        `## Coding Execution Evidence`,
        `- verification_checked: ${Number(codingExecutionEvidence.summary.verification_checked || 0)}`,
        `- verification_passed: ${Number(codingExecutionEvidence.summary.verification_passed || 0)}`,
        `- verification_failed: ${Number(codingExecutionEvidence.summary.verification_failed || 0)}`,
        `- retry_enabled_steps: ${Number(codingExecutionEvidence.summary.retry_enabled_steps || 0)}`,
        `- retry_attempted_steps: ${Number(codingExecutionEvidence.summary.retry_attempted_steps || 0)}`,
        `- failure_memory_entries: ${Number(codingExecutionEvidence.summary.failure_memory_entries || 0)}`, ``,
        `## Steps`,
        ...manifest.steps.map((s) => `- [${s.status === "succeeded" ? "OK" : "FAIL"}] ${s.step_index}:${s.step_id} (${s.tool_name})`),
      ];
      fs.writeFileSync(summaryPath, summaryLines.join("\n"), "utf8");
    } catch (err) {
      reasons.push(`artifact write failed: ${err.message}`);
    }

    const validator = validateArtifactPack({ run, steps, checkpoints, manifestPath, summaryPath, registry });
    try {
      const q = validator?.quality_summary || {};
      const qLines = [
        "", "## Artifact Quality",
        `- contract_found: ${Boolean(q.contract_found)}`,
        `- required: ${Number(q.required || 0)}`,
        `- present: ${Number(q.present || 0)}`,
        `- missing: ${Number(q.missing || 0)}`,
        `- invalid: ${Number(q.invalid || 0)}`,
      ];
      fs.appendFileSync(summaryPath, `${qLines.join("\n")}\n`, "utf8");
    } catch { /* ignore: summary write error is non-fatal */ }

    const expectedSteps = getExpectedWorkflowStepCount(registry, run.workflow_id);
    const goNoGo = buildGoNoGoResult({ run, manifest, steps, validator, canaryReport, expectedSteps, strict: true });
    writeJsonFile(goNoGoJsonPath, goNoGo);

    const finalResultPackage = buildFinalResultPackage({
      workflowRunId: run.workflow_run_id,
      runId: run.run_id,
      status: manifest.status,
      summaryPath,
      manifestPath,
      goNoGoResultPath: goNoGoJsonPath,
      strictCanaryReportPath: canaryMdPath,
      strictCanaryJsonPath: canaryJsonPath,
      goNoGoVerdict: goNoGo.verdict,
      strictCanaryVerdict: canaryReport.verdict,
    });
    const finalResultChecked = validateFinalResultPackage(finalResultPackage);
    if (!finalResultChecked.ok) {
      reasons.push(`final result package invalid: ${finalResultChecked.errors.join("; ")}`);
    }

    const minioArchived = await archiveReleasePackToMinio({
      run, manifestPath, summaryPath,
      extraPaths: [
        canaryJsonPath,
        canaryMdPath,
        goNoGoJsonPath,
        ...contextBudget.reports
          .map((item) => (item.report_path ? path.resolve(workspaceRoot, item.report_path) : ""))
          .filter(Boolean),
        ...contextArtifacts.flatMap((item) => [item.context_packet_path, item.repo_map_path])
          .filter(Boolean)
          .map((rel) => path.resolve(workspaceRoot, rel)),
        ...codingExecutionEvidence.items
          .flatMap((item) => [item.test_log_path, item.prompt_contract_path, item.coding_failure_memory_latest_path, item.coding_failure_memory_jsonl_path])
          .filter(Boolean)
          .map((rel) => path.resolve(workspaceRoot, rel)),
      ],
    });
    await indexReleasePackToDb({
      run, manifestPath, summaryPath,
      extraPaths: [
        canaryJsonPath,
        canaryMdPath,
        goNoGoJsonPath,
        ...contextBudget.reports
          .map((item) => (item.report_path ? path.resolve(workspaceRoot, item.report_path) : ""))
          .filter(Boolean),
        ...contextArtifacts.flatMap((item) => [item.context_packet_path, item.repo_map_path])
          .filter(Boolean)
          .map((rel) => path.resolve(workspaceRoot, rel)),
        ...codingExecutionEvidence.items
          .flatMap((item) => [item.test_log_path, item.prompt_contract_path, item.coding_failure_memory_latest_path, item.coding_failure_memory_jsonl_path])
          .filter(Boolean)
          .map((rel) => path.resolve(workspaceRoot, rel)),
      ],
      stepArtifacts, minioArchived,
    });

    const allReasons = [...reasons, ...(validator.reasons || [])];
    if (String(goNoGo.verdict || "") !== "GO") allReasons.push("go/no-go checklist failed");

    const result = {
      run_manifest_path: manifestPath,
      summary_path: summaryPath,
      go_no_go_result_path: goNoGoJsonPath,
      go_no_go_verdict: goNoGo.verdict,
      strict_canary_report_path: canaryMdPath,
      strict_canary_json_path: canaryJsonPath,
      strict_canary_verdict: canaryReport.verdict,
      final_result_package: finalResultPackage,
      validator,
    };
    if (allReasons.length > 0) {
      return { ok: false, error: "artifact pack incomplete", reasons: [...new Set(allReasons)], ...result };
    }
    return { ok: true, ...result };
  }

  async function validateRunArtifactPack(workflow_run_id) {
    const run = await getWorkflowRunById(pool, workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    const steps = await getSteps(workflow_run_id);
    const checkpoints = await listWorkflowCheckpoints(pool, workflow_run_id);
    const releasePaths = buildReleasePackPaths(workspaceRoot, run);
    return validateArtifactPack({
      run, steps,
      checkpoints,
      manifestPath: releasePaths.manifest_path,
      summaryPath: releasePaths.summary_path,
      registry,
    });
  }

  async function archiveRunArtifactPack(workflow_run_id) {
    const run = await getWorkflowRunById(pool, workflow_run_id);
    if (!run) {
      const err = new Error(`workflow_run '${workflow_run_id}' not found`);
      err.code = "WORKFLOW_RUN_NOT_FOUND";
      throw err;
    }
    const releasePaths = buildReleasePackPaths(workspaceRoot, run);
    const { manifest_path: manifestPath, summary_path: summaryPath,
            strict_canary_json_path: canaryJsonPath, strict_canary_report_path: canaryMdPath,
            go_no_go_result_path: goNoGoJsonPath } = releasePaths;

    if (!fs.existsSync(manifestPath) || !fs.existsSync(summaryPath)) {
      const err = new Error("ARTIFACT_INCOMPLETE: release pack files missing");
      err.code = "ARTIFACT_INCOMPLETE";
      throw err;
    }
    const steps = await getSteps(workflow_run_id);
    const checkpoints = await listWorkflowCheckpoints(pool, workflow_run_id);
    const stepArtifacts = buildStepArtifactsFromCheckpoints(steps, checkpoints, parseJsonSafe);
    const extraPaths = [canaryJsonPath, canaryMdPath, goNoGoJsonPath].filter((p) => fs.existsSync(p));
    const contextBudgetPaths = steps
      .map((step) => parseJsonSafe(step?.result_json, {}))
      .map((result) => String(result?.context_budget_report_path || "").trim())
      .filter(Boolean)
      .map((rel) => path.resolve(workspaceRoot, rel))
      .filter((p) => fs.existsSync(p));
    const contextArtifactPaths = steps
      .map((step) => parseJsonSafe(step?.result_json, {}))
      .flatMap((result) => [String(result?.context_packet_path || "").trim(), String(result?.repo_map_path || "").trim()])
      .filter(Boolean)
      .map((rel) => path.resolve(workspaceRoot, rel))
      .filter((p) => fs.existsSync(p));
    const codingEvidencePaths = steps
      .map((step) => parseJsonSafe(step?.result_json, {}))
      .flatMap((result) => {
        const diagnostics = result?.diagnostics && typeof result.diagnostics === "object" ? result.diagnostics : {};
        const memory = diagnostics?.coding_failure_memory && typeof diagnostics.coding_failure_memory === "object"
          ? diagnostics.coding_failure_memory
          : {};
        return [
          String(result?.artifacts?.test_log || "").trim(),
          String(memory?.latest_path || "").trim(),
          String(memory?.jsonl_path || "").trim(),
        ];
      })
      .filter(Boolean)
      .map((rel) => path.resolve(workspaceRoot, rel))
      .filter((p) => fs.existsSync(p));
    const archived = await archiveReleasePackToMinio({ run, manifestPath, summaryPath, extraPaths: [...extraPaths, ...contextBudgetPaths, ...contextArtifactPaths, ...codingEvidencePaths] });
    await indexReleasePackToDb({ run, manifestPath, summaryPath, extraPaths: [...extraPaths, ...contextBudgetPaths, ...contextArtifactPaths, ...codingEvidencePaths], stepArtifacts, minioArchived: archived });
    await recordEvent(workflow_run_id, "artifact.pack.minio.archived", {
      workflow_run_id, count: archived.length, bucket: minioBucket,
    });
    return { ok: true, count: archived.length, objects: archived };
  }

  return { generateArtifactPack, validateRunArtifactPack, archiveRunArtifactPack };
}
