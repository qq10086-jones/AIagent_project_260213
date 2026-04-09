/**
 * project_plan_contract.js
 *
 * Schema validation and DAG acyclicity checks for project_plan.json.
 * Rules C-01 through C-10 as defined in v3.3 design doc.
 */

import fs from "fs";
import path from "path";

// 标准 artifact 路径白名单 (来自 step_artifact_contract.js 定义)
const ALLOWED_ARTIFACT_PATHS = new Set([
  "handoff/pm_to_architect.json",
  "handoff/architect_to_impl.json",
  "handoff/be_to_fe.json",
  "handoff/impl_to_qa.json",
  "plan/spec.md",
  "plan/arch.md",
  "plan/interfaces.md",
  "plan/workplan.json",
  "plan/workplan.md",
  "plan/acceptance.json",
  "plan/milestones.md",
  "risk/risk_report.json",
  "impl/be_notes.md",
  "impl/fe_notes.md",
]);

const RUN_KEY_RE = /^R-\d{2,3}$/;

/**
 * 从 worker_coding_task_classes.json 加载已注册 task_class 枚举。
 * 支持传入已解析对象或文件路径。
 */
export function loadTaskClasses(taskClassesOrPath) {
  if (Array.isArray(taskClassesOrPath)) return new Set(taskClassesOrPath);
  if (taskClassesOrPath && typeof taskClassesOrPath === "object" && Array.isArray(taskClassesOrPath.task_classes)) {
    return new Set(taskClassesOrPath.task_classes);
  }
  if (typeof taskClassesOrPath === "string") {
    try {
      const raw = JSON.parse(fs.readFileSync(taskClassesOrPath, "utf8"));
      return new Set(Array.isArray(raw.task_classes) ? raw.task_classes : []);
    } catch { /* fall through */ }
  }
  // 默认枚举 (fallback)
  return new Set(["fe_create", "fe_modify", "be_create", "bug_fix", "artifact_completion"]);
}

/**
 * Topological sort — 返回排序后的 run_keys 或 null (有环)。
 */
export function topoSort(dependencyGraph) {
  const nodes = Object.keys(dependencyGraph);
  const inDegree = {};
  const adj = {};
  for (const n of nodes) {
    inDegree[n] = 0;
    adj[n] = [];
  }
  for (const [node, deps] of Object.entries(dependencyGraph)) {
    for (const dep of deps) {
      if (!adj[dep]) { adj[dep] = []; inDegree[dep] = 0; }
      adj[dep].push(node);
      inDegree[node] = (inDegree[node] || 0) + 1;
    }
  }
  const queue = Object.keys(inDegree).filter((k) => inDegree[k] === 0);
  const sorted = [];
  while (queue.length > 0) {
    const cur = queue.shift();
    sorted.push(cur);
    for (const next of (adj[cur] || [])) {
      inDegree[next]--;
      if (inDegree[next] === 0) queue.push(next);
    }
  }
  const allNodes = new Set([...nodes, ...Object.values(dependencyGraph).flat()]);
  return sorted.length === allNodes.size ? sorted : null;
}

/**
 * 计算执行波次 (waves) — 每一波可并行执行。
 */
export function computeWaves(dependencyGraph) {
  const sorted = topoSort(dependencyGraph);
  if (!sorted) return null;

  const waveMap = {};
  for (const key of sorted) {
    const deps = dependencyGraph[key] || [];
    const depWaves = deps.map((d) => waveMap[d] ?? -1);
    waveMap[key] = depWaves.length > 0 ? Math.max(...depWaves) + 1 : 0;
  }
  const waves = [];
  for (const [key, waveIdx] of Object.entries(waveMap)) {
    if (!waves[waveIdx]) waves[waveIdx] = [];
    waves[waveIdx].push(key);
  }
  return waves;
}

/**
 * 验证 project_plan.json 是否合规。
 *
 * @param {object} plan — 待验证的 plan 对象
 * @param {object} opts
 * @param {Set|Array|object|string} opts.taskClasses — 已注册 task_class (Set/Array/object/path)
 * @param {number} opts.maxRuns — 最大 run 数 (default 12)
 * @returns {{ ok: boolean, errors: string[], warnings: string[], sorted: string[]|null, waves: string[][]|null }}
 */
export function validateProjectPlan(plan, { taskClasses, maxRuns = 12 } = {}) {
  const errors = [];
  const warnings = [];
  const registeredClasses = loadTaskClasses(taskClasses);

  if (!plan || typeof plan !== "object") {
    return { ok: false, errors: ["plan is not an object"], warnings, sorted: null, waves: null };
  }

  const runs = Array.isArray(plan.runs) ? plan.runs : [];

  // C-01: runs 数量
  if (runs.length < 2) {
    errors.push(`C-01: runs.length=${runs.length}, minimum is 2 (use single-run workflow for smaller tasks)`);
  }
  if (runs.length > maxRuns) {
    errors.push(`C-01: runs.length=${runs.length} exceeds max_runs=${maxRuns}`);
  }

  const runKeys = new Set();
  const allTargetPaths = new Map(); // path → run_key

  for (const run of runs) {
    const key = String(run.run_key || "");

    // C-03: run_key 唯一 + 格式
    if (!RUN_KEY_RE.test(key)) {
      errors.push(`C-03: run_key '${key}' does not match R-\\d{2,3} format`);
    }
    if (runKeys.has(key)) {
      errors.push(`C-03: duplicate run_key '${key}'`);
    }
    runKeys.add(key);

    // C-02: task_class 枚举
    if (!registeredClasses.has(String(run.task_class || ""))) {
      errors.push(`C-02: run '${key}' has unregistered task_class '${run.task_class}'. Valid: [${[...registeredClasses].join(", ")}]`);
    }

    // C-07: acceptance_criteria 非空
    const ac = Array.isArray(run.acceptance_criteria) ? run.acceptance_criteria : [];
    if (ac.length < 1) {
      errors.push(`C-07: run '${key}' has no acceptance_criteria (minimum 1)`);
    }

    // C-08: prompt 长度
    const promptLen = String(run.prompt || "").length;
    if (promptLen < 50) {
      errors.push(`C-08: run '${key}' prompt is too short (${promptLen} chars, minimum 50)`);
    }

    // C-06: target_paths 跨 run 不重叠
    const paths = Array.isArray(run.target_paths) ? run.target_paths : [];
    for (const tp of paths) {
      const normalized = String(tp || "").replace(/\\/g, "/").replace(/\/+$/, "");
      if (!normalized) continue;
      for (const [existing, owner] of allTargetPaths) {
        if (owner === key) continue;
        if (normalized.startsWith(existing) || existing.startsWith(normalized)) {
          errors.push(`C-06: target_path '${normalized}' in run '${key}' overlaps with '${existing}' in run '${owner}'`);
        }
      }
      allTargetPaths.set(normalized, key);
    }

    // C-09: shared_context.artifacts 白名单
    const artifacts = run.shared_context?.artifacts || [];
    for (const art of artifacts) {
      if (!ALLOWED_ARTIFACT_PATHS.has(String(art || ""))) {
        warnings.push(`C-09: run '${key}' references non-standard artifact '${art}' (stripped)`);
      }
    }
  }

  // Build dependency_graph from runs (authoritative source)
  const depGraph = {};
  for (const run of runs) {
    const key = String(run.run_key || "");
    depGraph[key] = Array.isArray(run.depends_on) ? [...run.depends_on] : [];
  }

  // C-05: depends_on 引用存在
  for (const [key, deps] of Object.entries(depGraph)) {
    for (const dep of deps) {
      if (!runKeys.has(dep)) {
        errors.push(`C-05: run '${key}' depends_on '${dep}' which does not exist`);
      }
    }
  }

  // C-04: DAG 无环
  const sorted = topoSort(depGraph);
  if (!sorted) {
    errors.push("C-04: dependency_graph contains a cycle");
  }

  const waves = sorted ? computeWaves(depGraph) : null;

  // C-10: 孤立模块检测
  const modules = Array.isArray(plan.modules) ? plan.modules : [];
  const referencedModules = new Set(runs.map((r) => String(r.module_id || "")));
  for (const mod of modules) {
    if (!referencedModules.has(String(mod.module_id || ""))) {
      warnings.push(`C-10: module '${mod.module_id}' is not referenced by any run`);
    }
  }

  return {
    ok: errors.length === 0,
    errors,
    warnings,
    sorted,
    waves,
    dependency_graph: depGraph,
  };
}
