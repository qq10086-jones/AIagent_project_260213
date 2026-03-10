/**
 * workflow_step_artifacts.js
 *
 * Extracted from workflow_engine.js (WS-33-02).
 *
 * Provides per-step artifact helpers:
 *  - writeContextBudgetReport: writes a context budget JSON artifact to disk
 *  - applyStructuredPatchIfPresent: applies a be/fe patch bundle if found
 */

import fs from "fs";
import path from "path";

export function createWorkflowStepArtifactHelpers({ workspaceRoot, contextBudgetService, patchBundleService }) {
  function buildContextBudgetArtifactPath(payload) {
    const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
    return path.resolve(workspaceRoot, relRoot, "metrics", `context_budget_${String(payload?.step_id || "step")}.json`);
  }

  function buildContextPacketArtifactPath(payload) {
    const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
    return path.resolve(workspaceRoot, relRoot, "context", `context_packet_${String(payload?.step_id || "step")}.json`);
  }

  function buildRepoMapArtifactPath(payload) {
    const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
    return path.resolve(workspaceRoot, relRoot, "context", `repo_map_${String(payload?.step_id || "step")}.json`);
  }

  function writeContextBudgetReport(payload) {
    const reportPath = buildContextBudgetArtifactPath(payload);
    const report = contextBudgetService.buildReport({
      stepId: payload?.step_id,
      role: payload?.role,
      prompt: payload?.task_prompt || payload?.prompt || "",
      injectedContext: Array.isArray(payload?.target_file_context)
        ? payload.target_file_context.map((item) => `${item.path}\n${item.content}`).join("\n")
        : "",
      artifactPaths: [],
      runId: payload?.run_id,
      workflowRunId: payload?.workflow_run_id,
    });
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
    return {
      report,
      report_path: path.relative(workspaceRoot, reportPath).replace(/\\/g, "/"),
    };
  }

  function writeContextArtifacts(payload) {
    const result = {
      context_packet_path: null,
      repo_map_path: null,
    };
    if (payload?.context_packet && typeof payload.context_packet === "object") {
      const packetPath = buildContextPacketArtifactPath(payload);
      fs.mkdirSync(path.dirname(packetPath), { recursive: true });
      fs.writeFileSync(packetPath, JSON.stringify(payload.context_packet, null, 2), "utf8");
      result.context_packet_path = path.relative(workspaceRoot, packetPath).replace(/\\/g, "/");
    }
    if (payload?.repo_map && typeof payload.repo_map === "object") {
      const repoMapPath = buildRepoMapArtifactPath(payload);
      fs.mkdirSync(path.dirname(repoMapPath), { recursive: true });
      fs.writeFileSync(repoMapPath, JSON.stringify(payload.repo_map, null, 2), "utf8");
      result.repo_map_path = path.relative(workspaceRoot, repoMapPath).replace(/\\/g, "/");
    }
    return result;
  }

  function applyStructuredPatchIfPresent(payload) {
    const relRoot = String(payload?.artifact_root || "").trim().replace(/\\/g, "/");
    const stepId = String(payload?.step_id || "");
    const patchFileName = stepId === "impl_be" ? "be_patch_bundle.json" : stepId === "impl_fe" ? "fe_patch_bundle.json" : "";
    if (!patchFileName || !relRoot) return null;
    const bundlePath = path.resolve(workspaceRoot, relRoot, "impl", patchFileName);
    if (!fs.existsSync(bundlePath)) return null;
    const result = patchBundleService.applyPatchBundleFromFile(bundlePath);
    return {
      bundle_path: path.relative(workspaceRoot, bundlePath).replace(/\\/g, "/"),
      mode: String(result?.bundle?.mode || result?.mode || ""),
      written_files: Array.isArray(result?.written_files) ? result.written_files : [],
      operation_count: Number(result?.operation_count || 0),
    };
  }

  return { writeContextBudgetReport, writeContextArtifacts, applyStructuredPatchIfPresent };
}
