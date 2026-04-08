import crypto from "crypto";
import fs from "fs";
import path from "path";
import { truncateContextForTokenBudget } from "./task_contract.js";

export function buildAutoFixPrompt({ originalPrompt, previousFailure, attemptIndex }) {
  const lines = [
    String(originalPrompt || "").trim(),
    "",
    "[Auto-Fix Retry]",
    `Attempt: ${attemptIndex}`,
    "Repair objective: fix the previously failed coding attempt without expanding scope.",
  ];
  if (previousFailure?.phase) lines.push(`Previous failed phase: ${String(previousFailure.phase)}`);
  if (previousFailure?.error_code) lines.push(`Previous error code: ${String(previousFailure.error_code)}`);
  if (previousFailure?.error) lines.push(`Previous error: ${String(previousFailure.error)}`);
  if (previousFailure?.command_used) lines.push(`Previous command used: ${String(previousFailure.command_used)}`);
  lines.push("Instructions: preserve target_paths, fix the reported failure, and return only the minimum necessary change set.");
  return lines.join("\n");
}

export function buildExecutionPromptContract({
  stepId,
  targetPaths = [],
  expectedArtifacts = [],
  verificationCommand = "",
  verificationPlan = [],
  executionAdapterPacket = null,
  contextPacket = null,
  repoMap = null,
}) {
  const safeTargetPaths = Array.isArray(targetPaths) ? targetPaths.filter(Boolean) : [];
  const safeArtifacts = Array.isArray(expectedArtifacts) ? expectedArtifacts.filter(Boolean) : [];
  const inputArtifacts = Array.isArray(executionAdapterPacket?.input_artifacts)
    ? executionAdapterPacket.input_artifacts.filter(Boolean)
    : [];
  const lines = [
    "[Execution Contract v1]",
    `- step_id: ${String(stepId || "unknown")}`,
    `- target_paths: ${safeTargetPaths.length > 0 ? safeTargetPaths.join(", ") : "none"}`,
    `- required_outputs: ${safeArtifacts.length > 0 ? safeArtifacts.join(", ") : "none"}`,
    `- input_artifacts: ${inputArtifacts.length > 0 ? inputArtifacts.join(", ") : "none"}`,
    `- verification_command: ${String(verificationCommand || "").trim() || "none"}`,
    `- verification_plan: ${Array.isArray(verificationPlan) && verificationPlan.length > 0 ? verificationPlan.map((item) => `${String(item.tier || "custom")}:${String(item.command || "unresolved")}`).join(", ") : "none"}`,
    `- execution_mode: ${String(executionAdapterPacket?.execution_mode || "default")}`,
    "Rules:",
    "1. Only modify files under target_paths.",
    "2. Produce the minimum patch needed to satisfy the task.",
    "3. Do not edit infrastructure, CI, git metadata, or unrelated roots.",
    "4. Prefer deterministic file outputs over conversational prose.",
    "5. If a required artifact is requested, generate it at the exact relative path.",
  ];
  if (contextPacket && typeof contextPacket === "object") {
    const entrypoints = Array.isArray(contextPacket.entrypoints) ? contextPacket.entrypoints.filter(Boolean).slice(0, 6) : [];
    const relatedTests = Array.isArray(contextPacket.related_tests) ? contextPacket.related_tests.filter(Boolean).slice(0, 6) : [];
    if (entrypoints.length > 0) lines.push(`- entrypoints: ${entrypoints.join(", ")}`);
    if (relatedTests.length > 0) lines.push(`- related_tests: ${relatedTests.join(", ")}`);
  }
  if (repoMap && typeof repoMap === "object") {
    const candidateFiles = Array.isArray(repoMap.candidate_files) ? repoMap.candidate_files.filter(Boolean).slice(0, 8) : [];
    if (candidateFiles.length > 0) lines.push(`- candidate_files: ${candidateFiles.join(", ")}`);
  }
  return lines.join("\n");
}

export function writePromptContractArtifact({
  workspaceRoot,
  taskDir,
  attemptIndex,
  stepId,
  basePrompt,
  targetPaths,
  expectedArtifacts,
  verificationCommand,
  verificationPlan,
  executionAdapterPacket,
  contextPacket,
  repoMap,
  contextEnvelope = null,
}) {
  // Token budget truncation (Phase 0, P0-3)
  let effectiveContextPacket = contextPacket;
  let effectiveRepoMap = repoMap;
  let contextEnvelopeTruncated = false;
  let contextTokenUsage = null;
  const maxTokens = Number(contextEnvelope?.max_tokens || 0);
  if (maxTokens > 0 && (contextPacket || repoMap)) {
    const result = truncateContextForTokenBudget({
      contextPacket: contextPacket || {},
      repoMap: repoMap || {},
      maxTokens,
    });
    effectiveContextPacket = result.contextPacket;
    effectiveRepoMap = result.repoMap;
    contextEnvelopeTruncated = result.truncated;
    contextTokenUsage = result.tokenUsage;
  }

  const contractBlock = buildExecutionPromptContract({
    stepId,
    targetPaths,
    expectedArtifacts,
    verificationCommand,
    verificationPlan,
    executionAdapterPacket,
    contextPacket: effectiveContextPacket,
    repoMap: effectiveRepoMap,
  });
  const prompt = `${String(basePrompt || "").trim()}\n\n${contractBlock}\n`;
  const diagnostics = {
    contract_version: "execution_contract_v1",
    target_paths: Array.isArray(targetPaths) ? targetPaths : [],
    required_outputs: Array.isArray(expectedArtifacts) ? expectedArtifacts : [],
    verification_command: String(verificationCommand || "").trim() || null,
    verification_plan: Array.isArray(verificationPlan) ? verificationPlan : [],
    execution_mode: String(executionAdapterPacket?.execution_mode || "default"),
    prompt_sha1: crypto.createHash("sha1").update(prompt).digest("hex"),
    attempt: attemptIndex,
    context_envelope_truncated: contextEnvelopeTruncated,
    context_token_usage: contextTokenUsage,
  };
  let relPath = null;
  try {
    const outPath = path.join(taskDir, `prompt_contract_attempt${attemptIndex}_${Date.now()}.json`);
    fs.writeFileSync(outPath, JSON.stringify({
      generated_at: new Date().toISOString(),
      prompt,
      ...diagnostics,
    }, null, 2), "utf8");
    relPath = path.relative(workspaceRoot, outPath).replace(/\\/g, "/");
  } catch (writeErr) {
    console.warn("[prompt_contract] Failed to write prompt contract artifact:", writeErr.message);
  }
  return {
    prompt,
    path: relPath,
    diagnostics,
  };
}
