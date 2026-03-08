import { normalizeStepStatus } from "./workflow_state.js";

function stepKey(step, index) {
  return String(step?.id || `step_${index}`);
}

function normalizeDependsOn(step, index, steps) {
  if (Array.isArray(step?.depends_on)) {
    return step.depends_on.map((item) => String(item || "").trim()).filter(Boolean);
  }
  if (index === 0) return [];
  return [stepKey(steps[index - 1], index - 1)];
}

export function buildDagPlan(steps = []) {
  const byStepId = new Map();
  steps.forEach((step, index) => {
    byStepId.set(stepKey(step, index), index);
  });

  return steps.map((step, index) => {
    const dependencyIds = normalizeDependsOn(step, index, steps);
    const dependencyIndexes = dependencyIds.map((id) => {
      if (!byStepId.has(id)) {
        const err = new Error(`step '${stepKey(step, index)}' depends on unknown step '${id}'`);
        err.code = "WORKFLOW_DEPENDENCY_UNKNOWN";
        throw err;
      }
      return byStepId.get(id);
    });
    return {
      step_id: stepKey(step, index),
      step_index: index,
      dependency_ids: dependencyIds,
      dependency_indexes: dependencyIndexes,
      dependency_signature: dependencyIndexes.join(","),
    };
  });
}

function statusMap(stepRows = []) {
  const map = new Map();
  for (const row of stepRows) {
    map.set(Number(row.step_index), normalizeStepStatus(row.status));
  }
  return map;
}

export function listReadyStepIndexes({ stepRows = [], dagPlan = [] }) {
  const statuses = statusMap(stepRows);
  return dagPlan
    .filter((meta) => {
      const current = statuses.get(meta.step_index) || "pending";
      if (current !== "pending") return false;
      return meta.dependency_indexes.every((index) => (statuses.get(index) || "pending") === "succeeded");
    })
    .map((meta) => meta.step_index);
}

function collectParallelGroups(dagPlan = []) {
  const groups = new Map();
  for (const meta of dagPlan) {
    const key = meta.dependency_signature;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(meta);
  }
  return Array.from(groups.values()).filter((group) => group.length > 1);
}

export function summarizeDagProgress({ stepRows = [], dagPlan = [] }) {
  const statuses = statusMap(stepRows);
  const readyStepIndexes = listReadyStepIndexes({ stepRows, dagPlan });
  const hasActiveWork = Array.from(statuses.values()).some((status) =>
    ["queued", "waiting_approval", "running"].includes(status)
  );
  const anyFailed = Array.from(statuses.values()).some((status) => status === "failed");
  const allSucceeded =
    stepRows.length > 0 && stepRows.every((row) => normalizeStepStatus(row.status) === "succeeded");

  let mixedFailureGroup = null;
  let allFailedParallelGroup = null;

  for (const group of collectParallelGroups(dagPlan)) {
    const groupStatuses = group.map((meta) => statuses.get(meta.step_index) || "pending");
    const terminal = groupStatuses.every((status) => ["succeeded", "failed"].includes(status));
    if (!terminal) continue;
    if (groupStatuses.some((status) => status === "succeeded") && groupStatuses.some((status) => status === "failed")) {
      mixedFailureGroup = group;
      break;
    }
    if (groupStatuses.every((status) => status === "failed")) {
      allFailedParallelGroup = group;
    }
  }

  return {
    readyStepIndexes,
    hasActiveWork,
    allSucceeded,
    anyFailed,
    mixedFailureGroup,
    allFailedParallelGroup,
  };
}
