import fs from "fs";
import path from "path";

import { buildAttemptedFixes } from "./retry_policy.js";
import { buildTaskContractMetadata, deriveFailureAttribution } from "./task_contract.js";
import { redactSensitiveText } from "./verification_runner.js";

export function summarizeTerminalText(value, maxLines = 50, maxChars = 4000) {
  const text = redactSensitiveText(String(value || "").trim());
  if (!text) return "";
  const lines = text.split(/\r?\n/);
  const tail = lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
  return tail.slice(-maxChars);
}

export function persistCodingFailureMemory({
  workspaceRoot,
  runId,
  taskId,
  stepId,
  providerRequested,
  providerUsed,
  targetPaths = [],
  failedPhase,
  terminalOutcome,
  errorCode,
  error,
  commandUsed = null,
  verificationCommand = "",
  stderrText = "",
  stdoutText = "",
  attemptedFixes = [],
  filesChanged = [],
  artifactPaths = {},
  taskClass = null,
  betaTemplateId = null,
  contextEnvelope = null,
  lineage = null,
  contextResolution = null,
}) {
  try {
    const runDir = path.join(workspaceRoot, "artifacts", "runs", runId || "default");
    const memoryDir = path.join(runDir, "memory");
    fs.mkdirSync(memoryDir, { recursive: true });
    const entry = {
      failure_id: `cf_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`,
      recorded_at: new Date().toISOString(),
      run_id: String(runId || ""),
      task_id: String(taskId || ""),
      step_id: String(stepId || ""),
      provider_requested: String(providerRequested || ""),
      provider_used: providerUsed ? String(providerUsed) : null,
      failed_phase: String(failedPhase || "delegate"),
      terminal_outcome: String(terminalOutcome || "failed"),
      error_code: String(errorCode || ""),
      error: redactSensitiveText(String(error || "")),
      command_used: commandUsed ? String(commandUsed) : null,
      verification_command: verificationCommand ? String(verificationCommand) : null,
      stderr_summary: summarizeTerminalText(stderrText),
      stdout_summary: summarizeTerminalText(stdoutText, 20, 2000),
      target_paths: Array.isArray(targetPaths) ? targetPaths : [],
      files_changed: Array.isArray(filesChanged) ? filesChanged : [],
      attempted_fixes: Array.isArray(attemptedFixes) ? attemptedFixes : [],
      artifacts: artifactPaths || {},
      failure_attribution: deriveFailureAttribution({ phase: failedPhase, errorCode }),
      task_contract: buildTaskContractMetadata({
        taskClass,
        betaTemplateId,
        contextEnvelope,
      }),
      lineage: lineage && typeof lineage === "object" ? {
        parent_task_id: String(lineage.parent_task_id || ""),
        parent_run_id: String(lineage.parent_run_id || ""),
        refinement_round: Number(lineage.refinement_round || 0),
        refinement_instruction: String(lineage.refinement_instruction || "").slice(0, 500),
      } : null,
      // Phase 2 (P2-3): context_resolution cohort data
      context_resolution: contextResolution && typeof contextResolution === "object" ? {
        method: String(contextResolution.method || "none"),
        files_provided: Number(contextResolution.files_provided || 0),
        token_usage: Number(contextResolution.token_usage || 0),
        confidence: Number.isFinite(contextResolution.confidence) ? contextResolution.confidence : null,
        missing_context: Array.isArray(contextResolution.missing_context) ? contextResolution.missing_context : [],
      } : null,
    };
    const jsonlPath = path.join(memoryDir, "coding_failures.jsonl");
    fs.appendFileSync(jsonlPath, `${JSON.stringify(entry)}\n`, "utf8");
    const latestPath = path.join(memoryDir, "coding_failure_latest.json");
    fs.writeFileSync(latestPath, JSON.stringify(entry, null, 2), "utf8");
    return {
      jsonl_path: path.relative(workspaceRoot, jsonlPath).replace(/\\/g, "/"),
      latest_path: path.relative(workspaceRoot, latestPath).replace(/\\/g, "/"),
      failure_id: entry.failure_id,
    };
  } catch (err) {
    return {
      error: `coding failure memory write failed: ${String(err?.message || err || "")}`,
    };
  }
}

export function buildDelegateFailureSummary({
  workspaceRoot,
  runId,
  taskId,
  stepId,
  providerRequested,
  providerUsed,
  targetPaths,
  finalGitSummary,
  stdoutPath,
  stderrPath,
  staticCheck,
  verification,
  adapterResult,
  verificationCommand,
  promptContract,
  redactedStdout,
  redactedStderr,
  startedAt,
  phase,
  summaryText,
  testResult,
  errorCode,
  error,
  taskClass = null,
  betaTemplateId = null,
  contextEnvelope = null,
  lineage = null,
  contextResolution = null,
}) {
  const failureMemory = persistCodingFailureMemory({
    workspaceRoot,
    runId,
    taskId,
    stepId,
    providerRequested,
    providerUsed,
    targetPaths,
    failedPhase: phase,
    terminalOutcome: "failed",
    errorCode,
    error,
    commandUsed: adapterResult?.command_used || null,
    verificationCommand,
    stderrText: redactedStderr,
    stdoutText: redactedStdout,
    attemptedFixes: buildAttemptedFixes({ adapterResult, staticCheck, verification }),
    filesChanged: Array.isArray(finalGitSummary?.filesChanged) ? finalGitSummary.filesChanged : [],
    artifactPaths: {
      diff_bundle: finalGitSummary?.diffPath || null,
      test_log: verification?.logPath || staticCheck?.logPath || null,
      prompt_contract: promptContract?.path || null,
      raw_stdout: stdoutPath,
      raw_stderr: stderrPath,
    },
    taskClass,
    betaTemplateId,
    contextEnvelope,
    lineage,
    contextResolution,
  });
  const taskContract = buildTaskContractMetadata({
    taskClass,
    betaTemplateId,
    contextEnvelope,
  });
  const failureAttribution = deriveFailureAttribution({ phase, errorCode });
  return {
    ok: false,
    provider_used: providerUsed || null,
    model_used: adapterResult?.model_used || null,
    summary: summaryText,
    files_changed: Array.isArray(finalGitSummary?.filesChanged) ? finalGitSummary.filesChanged : [],
    diff_stats: finalGitSummary?.diffStats || { added: 0, deleted: 0, files: 0 },
    test_result: testResult,
    git: finalGitSummary?.git || { base_ref: "HEAD", branch: "main", commit_sha: null, dirty: false },
    rollback_performed: false,
    artifacts: {
      diff_bundle: finalGitSummary?.diffPath || null,
      patch_file: null,
      test_log: verification?.logPath || staticCheck?.logPath || null,
      prompt_contract: promptContract?.path || null,
      raw_stdout: stdoutPath,
      raw_stderr: stderrPath,
    },
    diagnostics: {
      ...(adapterResult?.diagnostics || {}),
      provider_requested: providerRequested,
      target_paths: targetPaths,
      error_code: errorCode,
      failed_phase: phase,
      failure_attribution: failureAttribution,
      task_contract: taskContract,
      prompt_contract: promptContract?.diagnostics || null,
      static_check: staticCheck || { checked: false, ok: true, commands: [], logPath: null },
      verification: verification || { checked: false, ok: true, command: "", logPath: null },
      coding_failure_memory: failureMemory,
      lineage: lineage || null,
      context_resolution: contextResolution || null,
    },
    error,
    command_used: adapterResult?.command_used || null,
    command_source: adapterResult?.command_source || "unknown",
    started_at: startedAt,
    finished_at: new Date().toISOString(),
  };
}
