/**
 * project_executor.js
 *
 * v3.3 Phase B+C — Multi-run orchestrator for project plans.
 * Takes a validated project_plan.json and executes its runs in topological order,
 * respecting dependency constraints and max_parallel_runs concurrency limits.
 *
 * Phase C additions:
 *   - P-C1: persistProjectSummary — write project_summary.json to artifact dir
 *   - P-C3: checkpoint save/load — persist run states for restart
 */

import fs from "fs";
import path from "path";
import { computeWaves } from "./project_plan_contract.js";

/**
 * Run 状态机:
 *   PENDING → SCHEDULED → RUNNING → COMPLETED | FAILED
 *                                → SKIPPED (if upstream failed + stop_dependents)
 */
const RUN_STATUS = {
  PENDING: "PENDING",
  SCHEDULED: "SCHEDULED",
  RUNNING: "RUNNING",
  COMPLETED: "COMPLETED",
  FAILED: "FAILED",
  SKIPPED: "SKIPPED",
};

/**
 * Project 状态机:
 *   CREATED → VALIDATED → CONFIRMED → RUNNING → COMPLETED | PARTIAL_FAILURE → REPORTED
 */
const PROJECT_STATUS = {
  CREATED: "CREATED",
  VALIDATED: "VALIDATED",
  CONFIRMED: "CONFIRMED",
  RUNNING: "RUNNING",
  COMPLETED: "COMPLETED",
  PARTIAL_FAILURE: "PARTIAL_FAILURE",
  REPORTED: "REPORTED",
};

/**
 * 从上游 run 的 artifact 目录提取 shared_context 文件。
 *
 * @param {object} opts
 * @param {string[]} opts.fromRuns — 上游 run_keys
 * @param {string[]} opts.artifacts — 要提取的标准 artifact 路径
 * @param {Map<string,object>} opts.runResults — run_key → { artifact_dir, ... }
 * @returns {object[]} 提取到的 artifact 内容列表
 */
function extractSharedContext({ fromRuns = [], artifacts = [], runResults }) {
  const extracted = [];
  for (const upstreamKey of fromRuns) {
    const upstream = runResults.get(upstreamKey);
    if (!upstream?.artifact_dir) continue;
    for (const artPath of artifacts) {
      const fullPath = path.join(upstream.artifact_dir, artPath);
      try {
        if (fs.existsSync(fullPath)) {
          const content = fs.readFileSync(fullPath, "utf8");
          extracted.push({
            from_run: upstreamKey,
            artifact_path: artPath,
            content,
          });
        } else {
          console.warn(`[project_executor] artifact not found: ${fullPath} (from ${upstreamKey}), skipping`);
        }
      } catch (err) {
        console.warn(`[project_executor] failed to read artifact ${fullPath}: ${err.message}`);
      }
    }
  }
  return extracted;
}

/**
 * 生成项目汇总报告。
 */
function buildProjectSummary({ plan, runStates, runResults, startTime }) {
  const endTime = new Date();
  const runs = plan.runs || [];
  const runSummaries = runs.map((run) => {
    const state = runStates.get(run.run_key) || RUN_STATUS.PENDING;
    const result = runResults.get(run.run_key);
    return {
      run_key: run.run_key,
      title: run.title,
      task_class: run.task_class,
      status: state,
      workflow_run_id: result?.workflow_run_id || null,
      duration_ms: result?.duration_ms || null,
      error: result?.error || null,
    };
  });

  const completedCount = runSummaries.filter((r) => r.status === RUN_STATUS.COMPLETED).length;
  const failedCount = runSummaries.filter((r) => r.status === RUN_STATUS.FAILED).length;
  const skippedCount = runSummaries.filter((r) => r.status === RUN_STATUS.SKIPPED).length;

  return {
    project_id: plan.project_id,
    project_title: plan.project_title,
    status: failedCount > 0 ? PROJECT_STATUS.PARTIAL_FAILURE : PROJECT_STATUS.COMPLETED,
    started_at: startTime.toISOString(),
    completed_at: endTime.toISOString(),
    duration_ms: endTime - startTime,
    total_runs: runs.length,
    completed: completedCount,
    failed: failedCount,
    skipped: skippedCount,
    acceptance_rate: runs.length > 0 ? completedCount / runs.length : 0,
    runs: runSummaries,
    risk_report: {
      has_failures: failedCount > 0,
      failure_runs: runSummaries.filter((r) => r.status === RUN_STATUS.FAILED).map((r) => r.run_key),
      skipped_runs: runSummaries.filter((r) => r.status === RUN_STATUS.SKIPPED).map((r) => r.run_key),
    },
  };
}

/**
 * 创建 project executor 实例。
 *
 * @param {object} opts
 * @param {object} opts.workflowEngine — createWorkflowEngine() 实例 (需要 startWorkflowRun)
 * @param {function} opts.recordEvent — 事件记录函数
 * @param {object} [opts.runtimeConfig] — runtime 配置
 * @returns {{ executeProjectPlan: function }}
 */
export function createProjectExecutor({
  workflowEngine,
  recordEvent,
  runtimeConfig = {},
  artifactDir = null,
} = {}) {
  if (!workflowEngine?.startWorkflowRun) {
    throw new Error("workflowEngine.startWorkflowRun is required");
  }
  if (typeof recordEvent !== "function") {
    throw new Error("recordEvent is required");
  }

  /**
   * 执行一个 project plan。
   *
   * @param {object} plan — 已验证的 project_plan.json
   * @param {object} [context] — 请求上下文
   * @param {object} [opts] — 执行选项
   * @param {boolean} [opts.resume=false] — 从 checkpoint 恢复
   * @returns {Promise<object>} project_summary.json
   */
  async function executeProjectPlan(plan, context = {}, { resume = false } = {}) {
    const startTime = new Date();
    const runs = plan.runs || [];
    const depGraph = plan.dependency_graph || {};
    const strategy = plan.execution_strategy || {};
    const maxParallel = Math.max(1, strategy.max_parallel_runs ?? runtimeConfig.project_planner_max_parallel_runs ?? 2);
    const failurePolicy = strategy.failure_policy ?? runtimeConfig.project_planner_failure_policy ?? "stop_dependents";

    // 状态追踪
    const runStates = new Map();
    const runResults = new Map();
    for (const run of runs) {
      runStates.set(run.run_key, RUN_STATUS.PENDING);
    }

    // P-C3: 从 checkpoint 恢复
    let restoredCount = 0;
    if (resume && artifactDir) {
      const checkpoint = loadCheckpoint(artifactDir);
      if (checkpoint && checkpoint.project_id === plan.project_id) {
        restoredCount = restoreFromCheckpoint(checkpoint, runStates, runResults);
        console.log(`[project_executor] restored ${restoredCount} completed runs from checkpoint`);
      }
    }

    // 计算执行波次
    const waves = computeWaves(depGraph);
    if (!waves) {
      throw new Error("Cannot compute execution waves — dependency graph has cycle");
    }

    await recordEvent(plan.project_id, "project.started", {
      project_id: plan.project_id,
      total_runs: runs.length,
      total_waves: waves.length,
      max_parallel: maxParallel,
      failure_policy: failurePolicy,
    });

    // Run 查找表
    const runByKey = new Map();
    for (const run of runs) {
      runByKey.set(run.run_key, run);
    }

    // 按波次执行
    for (let waveIdx = 0; waveIdx < waves.length; waveIdx++) {
      const wave = waves[waveIdx];

      // 过滤掉被跳过或已完成的 runs
      const executableRuns = wave.filter((key) => {
        const s = runStates.get(key);
        return s !== RUN_STATUS.SKIPPED && s !== RUN_STATUS.COMPLETED;
      });

      if (executableRuns.length === 0) continue;

      await recordEvent(plan.project_id, "project.wave.started", {
        project_id: plan.project_id,
        wave_index: waveIdx,
        run_keys: executableRuns,
      });

      // 按 maxParallel 分批执行
      for (let batchStart = 0; batchStart < executableRuns.length; batchStart += maxParallel) {
        const batch = executableRuns.slice(batchStart, batchStart + maxParallel);

        const batchPromises = batch.map(async (runKey) => {
          const run = runByKey.get(runKey);
          if (!run) return;

          // 检查是否已完成（checkpoint 恢复）或已被标记为 SKIPPED
          const currentState = runStates.get(runKey);
          if (currentState === RUN_STATUS.SKIPPED || currentState === RUN_STATUS.COMPLETED) return;

          runStates.set(runKey, RUN_STATUS.SCHEDULED);

          // 提取上游 shared_context
          const sharedContext = run.shared_context || {};
          const upstreamArtifacts = extractSharedContext({
            fromRuns: sharedContext.from_runs || [],
            artifacts: sharedContext.artifacts || [],
            runResults,
          });

          const projectContext = {
            project_id: plan.project_id,
            run_key: runKey,
            upstream_artifacts: upstreamArtifacts,
            project_constraints: plan.project_constraints,
          };

          runStates.set(runKey, RUN_STATUS.RUNNING);
          const runStart = Date.now();

          try {
            const result = await workflowEngine.startWorkflowRun({
              workflow_id: "coding_team_v0",
              project_type: plan.project_constraints?.project_type || "generic_coding_task",
              run_id: `${plan.project_id}__${runKey}`,
              input: {
                goal: run.prompt,
                provider: context.provider || runtimeConfig.coder_provider_default || "opencode",
                model: context.model || runtimeConfig.coder_model_default || "minimax-coding-plan/MiniMax-M2.7",
                task_envelope: context.task_envelope || null,
                project_context: projectContext,
                fast_mode: Boolean(context.fast_mode),
              },
              context,
            });

            runStates.set(runKey, RUN_STATUS.COMPLETED);
            runResults.set(runKey, {
              workflow_run_id: result.workflow_run_id,
              artifact_dir: result.artifact_dir || null,
              duration_ms: Date.now() - runStart,
            });

            await recordEvent(plan.project_id, "project.run.completed", {
              project_id: plan.project_id,
              run_key: runKey,
              workflow_run_id: result.workflow_run_id,
              duration_ms: Date.now() - runStart,
            });

            // P-C3: checkpoint after each completed run
            saveCheckpoint({ projectId: plan.project_id, plan, runStates, runResults, artifactDir });
          } catch (err) {
            runStates.set(runKey, RUN_STATUS.FAILED);
            runResults.set(runKey, {
              workflow_run_id: null,
              artifact_dir: null,
              duration_ms: Date.now() - runStart,
              error: err.message,
            });

            await recordEvent(plan.project_id, "project.run.failed", {
              project_id: plan.project_id,
              run_key: runKey,
              error: err.message,
              duration_ms: Date.now() - runStart,
            });

            // P-C3: checkpoint after failure too
            saveCheckpoint({ projectId: plan.project_id, plan, runStates, runResults, artifactDir });

            // 失败策略处理
            if (failurePolicy === "stop_all") {
              // 标记所有未执行的 runs 为 SKIPPED
              for (const [key, state] of runStates) {
                if (state === RUN_STATUS.PENDING) {
                  runStates.set(key, RUN_STATUS.SKIPPED);
                }
              }
            } else if (failurePolicy === "stop_dependents") {
              // 标记所有依赖此 run 的下游 runs 为 SKIPPED
              skipDependents(runKey, depGraph, runStates, runByKey);
            }
            // "continue_all" — 不做任何额外处理
          }
        });

        await Promise.all(batchPromises);
      }

      await recordEvent(plan.project_id, "project.wave.completed", {
        project_id: plan.project_id,
        wave_index: waveIdx,
        run_states: Object.fromEntries(runStates),
      });
    }

    // 生成汇总报告
    const summary = buildProjectSummary({ plan, runStates, runResults, startTime });

    // P-C1: 持久化 summary
    const summaryPath = persistProjectSummary(summary, artifactDir);
    if (summaryPath) {
      summary._summary_path = summaryPath;
      // 状态 → REPORTED (仅当 summary 成功持久化)
      if (summary.status === PROJECT_STATUS.COMPLETED) {
        summary.status = PROJECT_STATUS.REPORTED;
      }
    }
    if (restoredCount > 0) summary._restored_from_checkpoint = restoredCount;

    await recordEvent(plan.project_id, "project.completed", {
      project_id: plan.project_id,
      status: summary.status,
      duration_ms: summary.duration_ms,
      completed: summary.completed,
      failed: summary.failed,
      skipped: summary.skipped,
      summary_path: summaryPath,
    });

    return summary;
  }

  return { executeProjectPlan };
}

/**
 * 递归标记依赖失败 run 的下游 runs 为 SKIPPED。
 */
function skipDependents(failedKey, depGraph, runStates, runByKey) {
  for (const [key, deps] of Object.entries(depGraph)) {
    if (deps.includes(failedKey) && runStates.get(key) === RUN_STATUS.PENDING) {
      runStates.set(key, RUN_STATUS.SKIPPED);
      // 递归跳过下游
      skipDependents(key, depGraph, runStates, runByKey);
    }
  }
}

// ---------------------------------------------------------------------------
// P-C1: Summary persistence
// ---------------------------------------------------------------------------

/**
 * 持久化 project_summary.json 到 artifact 目录。
 */
function persistProjectSummary(summary, artifactDir) {
  if (!artifactDir) return null;
  try {
    fs.mkdirSync(artifactDir, { recursive: true });
    const filePath = path.join(artifactDir, "project_summary.json");
    fs.writeFileSync(filePath, JSON.stringify(summary, null, 2), "utf8");
    return filePath;
  } catch (err) {
    console.warn(`[project_executor] failed to persist summary: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// P-C3: Checkpoint save / load
// ---------------------------------------------------------------------------

/**
 * 保存 checkpoint — 持久化当前 run 状态和结果。
 */
function saveCheckpoint({ projectId, plan, runStates, runResults, artifactDir }) {
  if (!artifactDir) return null;
  try {
    fs.mkdirSync(artifactDir, { recursive: true });
    const checkpoint = {
      project_id: projectId,
      saved_at: new Date().toISOString(),
      run_states: Object.fromEntries(runStates),
      run_results: Object.fromEntries(
        [...runResults].map(([k, v]) => [k, { ...v }]),
      ),
      plan_snapshot: {
        project_id: plan.project_id,
        runs: (plan.runs || []).map((r) => r.run_key),
        dependency_graph: plan.dependency_graph,
        execution_strategy: plan.execution_strategy,
      },
    };
    const filePath = path.join(artifactDir, "project_checkpoint.json");
    fs.writeFileSync(filePath, JSON.stringify(checkpoint, null, 2), "utf8");
    return filePath;
  } catch (err) {
    console.warn(`[project_executor] failed to save checkpoint: ${err.message}`);
    return null;
  }
}

/**
 * 加载 checkpoint — 恢复之前的 run 状态。
 * 返回 null 如果无有效 checkpoint。
 */
function loadCheckpoint(artifactDir) {
  if (!artifactDir) return null;
  const filePath = path.join(artifactDir, "project_checkpoint.json");
  try {
    if (!fs.existsSync(filePath)) return null;
    const raw = JSON.parse(fs.readFileSync(filePath, "utf8"));
    if (!raw.run_states || typeof raw.run_states !== "object") return null;
    return raw;
  } catch {
    return null;
  }
}

/**
 * 从 checkpoint 恢复 runStates 和 runResults。
 * 已 COMPLETED 的 run 保持完成状态（跳过执行）。
 * FAILED / RUNNING / SCHEDULED 的 run 重置为 PENDING（重新执行）。
 */
function restoreFromCheckpoint(checkpoint, runStates, runResults) {
  let restored = 0;
  for (const [key, state] of Object.entries(checkpoint.run_states)) {
    if (state === RUN_STATUS.COMPLETED) {
      runStates.set(key, RUN_STATUS.COMPLETED);
      if (checkpoint.run_results?.[key]) {
        runResults.set(key, checkpoint.run_results[key]);
      }
      restored++;
    }
    // FAILED/RUNNING/SCHEDULED → 重置为 PENDING 让它重跑
  }
  return restored;
}

// 导出常量和内部函数用于测试
export {
  RUN_STATUS, PROJECT_STATUS,
  extractSharedContext, buildProjectSummary, skipDependents,
  persistProjectSummary, saveCheckpoint, loadCheckpoint, restoreFromCheckpoint,
};
