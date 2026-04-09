/**
 * project_planner.js
 *
 * v3.3 Project Planner — LLM-driven product-level requirement decomposition.
 * Takes a raw user requirement and decomposes it into multiple workflow runs,
 * each executable by the existing coding_team_v0 pipeline.
 */

import { qwenChat } from "../nlp/router.js";
import { inferCodingProjectType } from "./coding_project_type.js";
import { validateProjectPlan, topoSort, computeWaves } from "./project_plan_contract.js";

/**
 * 从 LLM 响应中提取 JSON 对象。
 * 支持 ```json ... ``` fence 和裸 JSON。
 */
function extractJson(text) {
  const raw = String(text || "").trim();

  // 尝试 ```json ... ``` fence
  const fenceMatch = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
  if (fenceMatch) {
    try { return JSON.parse(fenceMatch[1].trim()); } catch { /* fall through */ }
  }

  // 尝试第一个 { ... } 块
  const braceStart = raw.indexOf("{");
  const braceEnd = raw.lastIndexOf("}");
  if (braceStart >= 0 && braceEnd > braceStart) {
    try { return JSON.parse(raw.slice(braceStart, braceEnd + 1)); } catch { /* fall through */ }
  }

  return null;
}

/**
 * 生成项目 slug (用于 workspace 路径)。
 */
function generateProjectSlug(rawInput) {
  const slug = String(rawInput || "")
    .replace(/[^\w\u4e00-\u9fff\s-]/g, "")
    .trim()
    .slice(0, 40)
    .replace(/\s+/g, "-")
    .toLowerCase();
  return slug || "project";
}

/**
 * 构建 decomposition prompt。
 * 所有变量运行时注入 — 不硬编码枚举或路径。
 */
function buildDecompositionPrompt({ rawInput, taskClasses, projectType, workspaceRoot }) {
  const taskClassList = Array.isArray(taskClasses) ? taskClasses : [...taskClasses];

  return `<role>
你是 Nexus 项目规划器。你的职责是将产品级需求拆解为多个可独立执行的功能模块（Run）。
每个 Run 将由一条完整的 coding_team_v0 流水线执行（PM→架构→实现→测试→QA→发布）。
</role>

<task>
将以下需求拆解为多个独立的 Runs。
需求: ${rawInput}
</task>

<constraints>
已注册 task_class: ${JSON.stringify(taskClassList)}
项目类型: ${projectType}
目标 workspace: ${workspaceRoot}
</constraints>

<rules>
1. 粒度控制: 每个 Run 必须是一个 LLM 在单次调用中能完成的功能单元
   - 好的粒度: "用户认证 API（注册+登录+JWT验证）" — 1个模块, 3-5个文件
   - 坏的粒度: "整个后端" — 太大; "加一行配置" — 太小
2. task_class 必须从已注册枚举中选择: ${taskClassList.join(" | ")}
3. depends_on 声明因果依赖:
   - 后端 API 必须先于调用它的前端页面
   - 共享数据模型必须先于使用它的业务逻辑
   - 不相关的模块之间不要加假依赖
4. target_paths 在不同 Run 之间不得重叠，每个 run 用独立子目录
5. 每个 Run 包含 2-5 条具体的验收标准（AC），可验证、可自动化
6. 总 Run 数量: 3-12 个
7. shared_context.artifacts 只能引用标准 artifact 路径:
   handoff/pm_to_architect.json, handoff/architect_to_impl.json,
   handoff/be_to_fe.json, handoff/impl_to_qa.json,
   plan/interfaces.md, plan/spec.md, plan/arch.md, plan/workplan.json
8. prompt 字段: 写给 PM 角色看的需求描述 (200-500字)，包含:
   - 要实现的功能
   - 关键数据结构/接口
   - 与其他模块的边界约束
9. run_key 格式: R-01, R-02, ..., R-12
</rules>

<output_format>
返回 JSON，严格符合以下结构 (不要包含注释):
{
  "modules": [{ "module_id": "mod-xxx", "title": "...", "description": "..." }],
  "runs": [{
    "run_key": "R-01",
    "module_id": "mod-xxx",
    "task_class": "${taskClassList[0] || "be_create"}",
    "title": "简短标题 (<80字符)",
    "prompt": "给 PM 的详细需求描述 (200-500字)...",
    "target_paths": ["${workspaceRoot}子路径/"],
    "depends_on": [],
    "shared_context": { "from_runs": [], "artifacts": [] },
    "estimated_complexity": "medium",
    "acceptance_criteria": ["AC-R01-1: 具体可验证条件"]
  }],
  "tech_stack_hints": { "backend": "...", "frontend": "...", "test": "..." }
}
</output_format>`;
}

/**
 * 构建 retry prompt — 包含上次失败的错误信息。
 */
function buildRetryPrompt(originalPrompt, errors) {
  return `${originalPrompt}

<retry_feedback>
上次拆解结果未通过验证，请修正以下问题:
${errors.map((e, i) => `${i + 1}. ${e}`).join("\n")}
</retry_feedback>`;
}

/**
 * 构建 single-run fallback plan。
 * 当 LLM 拆解失败时，退化为现有的单 run 行为。
 */
function buildFallbackPlan({ rawInput, projectType, workspaceRoot }) {
  return {
    project_id: `proj-fallback-${Date.now()}`,
    project_title: String(rawInput || "").slice(0, 100),
    created_at: new Date().toISOString(),
    decomposition_model: "fallback",
    decomposition_confidence: 0,
    modules: [],
    runs: [
      {
        run_key: "R-01",
        module_id: "mod-fallback",
        task_class: "be_create",
        title: String(rawInput || "").slice(0, 80),
        prompt: String(rawInput || ""),
        target_paths: [`${workspaceRoot}`],
        depends_on: [],
        shared_context: { from_runs: [], artifacts: [] },
        estimated_complexity: "complex",
        acceptance_criteria: ["AC-R01-1: 功能按需求实现"],
      },
    ],
    dependency_graph: { "R-01": [] },
    execution_strategy: { max_parallel_runs: 1, failure_policy: "stop_all", retry_failed_runs: false, max_retries_per_run: 0 },
    project_constraints: { project_type: projectType, workspace_root: workspaceRoot, tech_stack_hints: null },
    _fallback: true,
  };
}

/**
 * 将 LLM 输出补全为完整的 project_plan.json。
 */
function assemblePlan({ llmOutput, rawInput, projectType, workspaceRoot, model, config }) {
  const slug = generateProjectSlug(rawInput);
  const runs = Array.isArray(llmOutput.runs) ? llmOutput.runs : [];

  // 从 runs 构建 dependency_graph
  const depGraph = {};
  for (const run of runs) {
    depGraph[String(run.run_key || "")] = Array.isArray(run.depends_on) ? [...run.depends_on] : [];
  }

  return {
    project_id: `proj-${Date.now()}-${slug}`,
    project_title: String(rawInput || "").slice(0, 120),
    created_at: new Date().toISOString(),
    decomposition_model: model,
    decomposition_confidence: llmOutput.confidence ?? 0.8,
    modules: Array.isArray(llmOutput.modules) ? llmOutput.modules : [],
    runs,
    dependency_graph: depGraph,
    execution_strategy: {
      max_parallel_runs: config.max_parallel_runs ?? 2,
      failure_policy: config.failure_policy ?? "stop_dependents",
      retry_failed_runs: true,
      max_retries_per_run: 1,
    },
    project_constraints: {
      project_type: projectType,
      workspace_root: workspaceRoot,
      tech_stack_hints: llmOutput.tech_stack_hints || null,
    },
  };
}

/**
 * 核心拆解函数。
 *
 * @param {object} opts
 * @param {string} opts.raw_input — 用户原始需求文本
 * @param {string} [opts.project_type] — 项目类型 (可选, 自动推断)
 * @param {object} [opts.classifier_result] — brain_router_classifier 结果
 * @param {object} [opts.registry] — capability_registry.json 对象
 * @param {Set|Array|object} [opts.task_classes] — 已注册 task_class
 * @param {object} [opts.config] — runtime config (project_planner_* flags)
 * @returns {Promise<object>} project_plan.json
 */
export async function decompose({
  raw_input,
  project_type,
  classifier_result,
  registry,
  task_classes,
  config = {},
} = {}) {
  const rawInput = String(raw_input || "").trim();
  if (!rawInput) {
    return buildFallbackPlan({ rawInput, projectType: "generic_coding_task", workspaceRoot: "workspace/sandbox/project/" });
  }

  const projectType = project_type || inferCodingProjectType(rawInput);
  const slug = generateProjectSlug(rawInput);
  const workspaceRoot = `workspace/sandbox/${slug}/`;

  const taskClassSet = task_classes instanceof Set ? task_classes
    : Array.isArray(task_classes) ? new Set(task_classes)
    : (task_classes?.task_classes ? new Set(task_classes.task_classes) : new Set(["fe_create", "fe_modify", "be_create", "bug_fix", "artifact_completion"]));

  const maxRuns = config.project_planner_max_runs ?? 12;
  const maxParallelRuns = config.project_planner_max_parallel_runs ?? 2;
  const failurePolicy = config.project_planner_failure_policy ?? "stop_dependents";

  const prompt = buildDecompositionPrompt({
    rawInput,
    taskClasses: taskClassSet,
    projectType,
    workspaceRoot,
  });

  // 第一次 LLM 调用
  let llmResult = await callLlm(prompt);
  let llmRaw = llmResult?.raw;
  let llmOutput = llmResult?.parsed;
  let model = llmResult?.model || "unknown";

  if (!llmOutput) {
    console.warn("[project_planner] LLM returned no parseable JSON, using fallback");
    return buildFallbackPlan({ rawInput, projectType, workspaceRoot });
  }

  // 验证
  let plan = assemblePlan({
    llmOutput,
    rawInput,
    projectType,
    workspaceRoot,
    model,
    config: { max_parallel_runs: maxParallelRuns, failure_policy: failurePolicy },
  });

  let validation = validateProjectPlan(plan, { taskClasses: taskClassSet, maxRuns });

  // 验证失败 → 重试 1 次 (带错误反馈)
  if (!validation.ok && validation.errors.length > 0) {
    console.warn("[project_planner] First attempt failed validation, retrying with feedback");
    const retryPrompt = buildRetryPrompt(prompt, validation.errors);
    const retryResult = await callLlm(retryPrompt);
    if (retryResult?.parsed) {
      plan = assemblePlan({
        llmOutput: retryResult.parsed,
        rawInput,
        projectType,
        workspaceRoot,
        model: retryResult.model || model,
        config: { max_parallel_runs: maxParallelRuns, failure_policy: failurePolicy },
      });
      validation = validateProjectPlan(plan, { taskClasses: taskClassSet, maxRuns });
      llmRaw = retryResult.raw;
    }
  }

  if (!validation.ok) {
    console.warn("[project_planner] Retry also failed validation, using fallback");
    const fallback = buildFallbackPlan({ rawInput, projectType, workspaceRoot });
    fallback._validation_errors = validation.errors;
    fallback._llm_raw = llmRaw;
    return fallback;
  }

  // 附加验证结果
  plan.sorted = validation.sorted;
  plan.waves = validation.waves;
  plan._warnings = validation.warnings;
  plan._llm_raw = llmRaw;

  return plan;
}

/**
 * 内部 LLM 调用封装。
 */
async function callLlm(systemContent) {
  try {
    const messages = [
      { role: "system", content: systemContent },
      { role: "user", content: "请根据上述需求进行项目拆解，返回 JSON。" },
    ];
    const result = await qwenChat(messages, 90000);
    if (!result?.choices?.[0]?.message?.content) {
      return { raw: null, parsed: null, model: null };
    }
    const raw = result.choices[0].message.content;
    const parsed = extractJson(raw);
    return { raw, parsed, model: result.model || null };
  } catch (err) {
    console.error("[project_planner] LLM call failed:", err.message);
    return { raw: null, parsed: null, model: null };
  }
}

// 导出内部函数用于测试
export { extractJson, generateProjectSlug, buildDecompositionPrompt, buildFallbackPlan, assemblePlan };
