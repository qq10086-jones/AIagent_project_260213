/**
 * workflow_refinement_service.js — Phase 1.5-4: Orchestrator-side refinement re-entry.
 *
 * Determines which workflow steps to skip for refinement tasks based on task_class
 * classification. Returns a list of step IDs that should be auto-succeeded.
 *
 * Design contract (v3.2 Section 3.5.4):
 * - fe_modify: skip pm_spec, arch_design, impl_be, release_pack, deploy_preview
 * - be_create / be_modify: skip pm_spec, arch_design, impl_fe (unless handoff has interface deps)
 * - bug_fix: skip pm_spec, arch_design
 * - default: skip nothing (safe fallback)
 */

const REFINEMENT_MAX_ROUNDS = 5;

// 步骤跳过规则表 (3.5.4)
const SKIP_RULES = {
  fe_modify: new Set(["pm_spec", "arch_design", "impl_be", "release_pack", "deploy_preview"]),
  fe_create: new Set(["pm_spec", "arch_design", "impl_be", "release_pack", "deploy_preview"]),
  be_modify: new Set(["pm_spec", "arch_design", "impl_fe", "release_pack", "deploy_preview"]),
  be_create: new Set(["pm_spec", "arch_design", "impl_fe", "release_pack", "deploy_preview"]),
  bug_fix: new Set(["pm_spec", "arch_design", "release_pack", "deploy_preview"]),
  full_stack: new Set(["pm_spec", "arch_design", "release_pack", "deploy_preview"]),
};

/**
 * Normalize and validate a lineage object from run input.
 * @param {Object|null} lineage
 * @returns {{ ok: boolean, lineage: Object|null, error?: string }}
 */
export function normalizeRefinementLineage(lineage) {
  if (!lineage || typeof lineage !== "object") {
    return { ok: false, lineage: null, error: "lineage is null or not an object" };
  }
  const parentTaskId = String(lineage.parent_task_id || "").trim();
  if (!parentTaskId) {
    return { ok: false, lineage: null, error: "parent_task_id is required" };
  }
  const round = Number(lineage.refinement_round);
  if (!Number.isFinite(round) || round < 1) {
    return { ok: false, lineage: null, error: "refinement_round must be a positive integer" };
  }
  if (round > REFINEMENT_MAX_ROUNDS) {
    return { ok: false, lineage: null, error: `refinement_round ${round} exceeds max ${REFINEMENT_MAX_ROUNDS}` };
  }
  return {
    ok: true,
    lineage: {
      parent_task_id: parentTaskId,
      parent_run_id: String(lineage.parent_run_id || "").trim() || null,
      refinement_round: Math.trunc(round),
      refinement_instruction: String(lineage.refinement_instruction || "").trim() || null,
      refinement_task_class: String(lineage.refinement_task_class || "").trim().toLowerCase() || null,
      inherited_context: lineage.inherited_context && typeof lineage.inherited_context === "object"
        ? lineage.inherited_context
        : null,
    },
  };
}

/**
 * Determine which steps to skip for a refinement task.
 * @param {{ lineage: Object, steps: Array, handoffHasInterfaceDeps?: boolean }} opts
 * @returns {{ skipStepIds: string[], reason: string }}
 */
export function resolveRefinementSkips({ lineage, steps = [], handoffHasInterfaceDeps = false }) {
  if (!lineage || !lineage.refinement_task_class) {
    return { skipStepIds: [], reason: "no_refinement_task_class" };
  }

  const taskClass = lineage.refinement_task_class;
  const skipSet = SKIP_RULES[taskClass];
  if (!skipSet) {
    return { skipStepIds: [], reason: `unknown_task_class:${taskClass}` };
  }

  const validStepIds = new Set(steps.map((s) => String(s?.id || "")));
  let skipStepIds = Array.from(skipSet).filter((id) => validStepIds.has(id));

  // be_create/be_modify: 如果 handoff 有接口依赖，不能跳过 impl_fe
  if (["be_create", "be_modify"].includes(taskClass) && handoffHasInterfaceDeps) {
    skipStepIds = skipStepIds.filter((id) => id !== "impl_fe");
  }

  return { skipStepIds, reason: `refinement_skip:${taskClass}` };
}

/**
 * Check if a be_to_fe.json handoff has non-empty interface dependencies.
 * Used to decide if impl_fe must remain for BE refinements.
 * @param {Object|null} beToFeHandoff
 * @returns {boolean}
 */
export function hasNonEmptyHandoffDeps(beToFeHandoff) {
  if (!beToFeHandoff || typeof beToFeHandoff !== "object") return false;
  const contracts = beToFeHandoff.api_contracts;
  if (Array.isArray(contracts) && contracts.length > 0) return true;
  const types = beToFeHandoff.shared_types;
  if (Array.isArray(types) && types.length > 0) return true;
  return false;
}
