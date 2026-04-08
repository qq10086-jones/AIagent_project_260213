/**
 * refinement_context_builder.js — Lineage context builder for refinement re-entry.
 * Phase 1.5 (P1.5-1): Builds augmented prompt context for tasks that inherit
 * lineage from parent tasks, enabling iterative fix loops without context loss.
 */
import { readScopedFile } from "./native_file_tools.js";

const REFINEMENT_MAX_ROUNDS = 5;

/**
 * Validate lineage object shape and constraints.
 * @param {Object|null} lineage
 * @returns {{ ok: boolean, error?: string }}
 */
export function validateLineage(lineage) {
  if (!lineage || typeof lineage !== "object") {
    return { ok: false, error: "lineage is null or not an object" };
  }
  const parentTaskId = String(lineage.parent_task_id || "").trim();
  if (!parentTaskId) {
    return { ok: false, error: "parent_task_id is required" };
  }
  const round = Number(lineage.refinement_round);
  if (!Number.isFinite(round) || round < 1) {
    return { ok: false, error: "refinement_round must be a positive integer" };
  }
  if (round > REFINEMENT_MAX_ROUNDS) {
    return { ok: false, error: `refinement_round ${round} exceeds max ${REFINEMENT_MAX_ROUNDS}` };
  }
  return { ok: true };
}

/**
 * Build refinement context block to prepend to the task prompt.
 * Reads current state of target files and constructs a structured context injection.
 *
 * @param {{ lineage: Object, workspaceRoot: string, targetPaths: string[], maxContextBytes?: number }} opts
 * @returns {{ contextBlock: string, diagnostics: Object }}
 */
export function buildRefinementContext({ lineage, workspaceRoot, targetPaths = [], maxContextBytes = 16384 }) {
  const validation = validateLineage(lineage);
  if (!validation.ok) {
    return {
      contextBlock: "",
      diagnostics: { error: validation.error, files_included: 0, total_bytes: 0 },
    };
  }

  const round = Math.trunc(Number(lineage.refinement_round));
  const parentTaskId = String(lineage.parent_task_id || "").trim();
  const parentRunId = String(lineage.parent_run_id || "").trim() || "unknown";
  const instruction = String(lineage.refinement_instruction || "").trim() || "(no instruction provided)";

  const lines = [
    `[Refinement Re-entry Round ${round}]`,
    `Parent task: ${parentTaskId} (run: ${parentRunId})`,
    `Instruction: ${instruction}`,
    "",
  ];

  // Collect current state of target files
  const filesIncluded = [];
  let totalBytes = 0;

  if (Array.isArray(targetPaths) && targetPaths.length > 0) {
    lines.push("[Current State of Target Files]");

    for (const tp of targetPaths) {
      if (totalBytes >= maxContextBytes) break;

      const result = readScopedFile({
        workspaceRoot,
        relPath: tp,
        allowedTargetPaths: targetPaths,
        maxBytes: Math.min(8192, maxContextBytes - totalBytes),
      });

      if (result.ok && result.content) {
        lines.push(`--- ${tp} ---`);
        lines.push(result.content);
        lines.push("");
        totalBytes += result.bytes_read || result.content.length;
        filesIncluded.push(tp);
      }
    }
  }

  // Inherited context from parent (if provided)
  if (lineage.inherited_context?.files_changed && Array.isArray(lineage.inherited_context.files_changed)) {
    const parentFiles = lineage.inherited_context.files_changed;
    for (const filePath of parentFiles) {
      if (totalBytes >= maxContextBytes) break;
      if (filesIncluded.includes(filePath)) continue;

      const result = readScopedFile({
        workspaceRoot,
        relPath: filePath,
        allowedTargetPaths: targetPaths,
        maxBytes: Math.min(4096, maxContextBytes - totalBytes),
      });

      if (result.ok && result.content) {
        lines.push(`--- ${filePath} (parent changed) ---`);
        lines.push(result.content);
        lines.push("");
        totalBytes += result.bytes_read || result.content.length;
        filesIncluded.push(filePath);
      }
    }
  }

  lines.push("Rules:");
  lines.push("1. Fix ONLY what the refinement instruction specifies.");
  lines.push("2. Do NOT expand scope beyond inherited target_paths.");
  lines.push("3. Do NOT revert changes from the parent task unless explicitly instructed.");
  lines.push(`4. This is round ${round} of max ${REFINEMENT_MAX_ROUNDS}.`);

  return {
    contextBlock: lines.join("\n"),
    diagnostics: {
      round,
      parent_task_id: parentTaskId,
      parent_run_id: parentRunId,
      files_included: filesIncluded.length,
      total_bytes: totalBytes,
    },
  };
}
