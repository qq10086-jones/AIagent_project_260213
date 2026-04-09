/**
 * project_planner.js
 *
 * v3.3 Project Planner — LLM-driven product-level requirement decomposition.
 * Takes a raw user requirement and decomposes it into multiple workflow runs,
 * each executable by the existing coding_team_v0 pipeline.
 *
 * Supports two LLM backends:
 *   - MiniMax (cloud) via qwenChat()
 *   - Ollama (local) via callLocalOllamaChat() — default for planner
 */

import { qwenChat } from "../nlp/router.js";
import { callLocalOllamaChat } from "./local_llm_client.js";
import { inferCodingProjectType } from "./coding_project_type.js";
import { validateProjectPlan, topoSort, computeWaves } from "./project_plan_contract.js";

// ---------------------------------------------------------------------------
// extractJson — robust JSON extraction from LLM output
// ---------------------------------------------------------------------------

/**
 * 从 LLM 响应中提取 JSON 对象。
 * 处理: <think> 标签、```json fence、裸 JSON、截断 JSON 修复。
 */
function extractJson(text) {
  let raw = String(text || "").trim();

  // 1. 移除 <think>...</think> 标签（MiniMax-M2.7 thinking chain）
  raw = raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  // 移除未闭合的 <think>... (截断的 thinking)
  raw = raw.replace(/<think>[\s\S]*$/gi, "").trim();

  // 2. 尝试 ```json ... ``` fence
  const fenceMatch = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
  if (fenceMatch) {
    const fenceContent = fenceMatch[1].trim();
    try { return JSON.parse(fenceContent); } catch {
      // fence 内 JSON 可能被截断，尝试修复
      const fixed = tryFixTruncatedJson(fenceContent);
      if (fixed) return fixed;
    }
  }

  // 3. 尝试第一个 { ... } 块
  const braceStart = raw.indexOf("{");
  const braceEnd = raw.lastIndexOf("}");
  if (braceStart >= 0 && braceEnd > braceStart) {
    const candidate = raw.slice(braceStart, braceEnd + 1);
    try { return JSON.parse(candidate); } catch {
      const fixed = tryFixTruncatedJson(candidate);
      if (fixed) return fixed;
    }
  }

  // 4. 最后尝试: 从 { 开始到末尾，修复截断
  if (braceStart >= 0) {
    const tail = raw.slice(braceStart);
    const fixed = tryFixTruncatedJson(tail);
    if (fixed) return fixed;
  }

  return null;
}

/**
 * 尝试修复被截断的 JSON。
 * 策略: 补齐缺失的 ] 和 }，移除尾部不完整的字段。
 */
function tryFixTruncatedJson(text) {
  let s = String(text || "").trim();
  if (!s.startsWith("{")) return null;

  // 移除尾部不完整的键值对 (如 "key": "val 或 "key": [1,2
  s = s.replace(/,\s*"[^"]*"?\s*:\s*("([^"\\]|\\.)*)?$/g, "");
  s = s.replace(/,\s*$/g, "");

  // 计算未闭合的括号
  let braces = 0;
  let brackets = 0;
  let inString = false;
  let escape = false;
  for (const ch of s) {
    if (escape) { escape = false; continue; }
    if (ch === "\\") { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === "{") braces++;
    if (ch === "}") braces--;
    if (ch === "[") brackets++;
    if (ch === "]") brackets--;
  }

  // 如果在字符串内，先闭合字符串
  if (inString) s += '"';

  // 补齐括号
  while (brackets > 0) { s += "]"; brackets--; }
  while (braces > 0) { s += "}"; braces--; }

  try { return JSON.parse(s); } catch { return null; }
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

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
 * 精简的 decomposition prompt。
 */
function buildDecompositionPrompt({ rawInput, taskClasses, projectType, workspaceRoot }) {
  const taskClassList = Array.isArray(taskClasses) ? taskClasses : [...taskClasses];

  return `你是项目拆解器。将产品需求拆为多个独立 Run，每个 Run 由 coding_team_v0 流水线执行。

需求: ${rawInput}
task_class 枚举: ${taskClassList.join(" | ")}
项目类型: ${projectType}
workspace: ${workspaceRoot}

规则:
1. 每 Run 粒度 = 1个 LLM 单次能完成的功能（1模块 3-5文件）
2. task_class 只能从枚举中选
3. depends_on 声明因果依赖（后端先于前端，共享模型先于业务逻辑）
4. target_paths 不同 Run 间不重叠
5. 每 Run 含 2-5 条可验证的 acceptance_criteria
6. 总 Run 数: 2-12
7. run_key 格式: R-01, R-02, ...
8. prompt 字段写给 PM 的详细描述（至少 50 字）

不要输出思考过程。直接返回 JSON（不要注释，不要 markdown 解释）:
{"modules":[{"module_id":"mod-xxx","title":"...","description":"..."}],"runs":[{"run_key":"R-01","module_id":"mod-xxx","task_class":"${taskClassList[0] || "be_create"}","title":"标题(<80字)","prompt":"给PM的详细需求(至少50字)，描述功能、数据结构、边界...","target_paths":["${workspaceRoot}backend/"],"depends_on":[],"shared_context":{"from_runs":[],"artifacts":[]},"estimated_complexity":"medium","acceptance_criteria":["AC-R01-1: 具体可验证条件","AC-R01-2: 第二条"]}]}`;
}

function buildRetryPrompt(originalPrompt, errors) {
  return `${originalPrompt}

上次结果未通过验证，请修正:
${errors.map((e, i) => `${i + 1}. ${e}`).join("\n")}`;
}

// ---------------------------------------------------------------------------
// Fallback & assembly
// ---------------------------------------------------------------------------

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

function assemblePlan({ llmOutput, rawInput, projectType, workspaceRoot, model, config }) {
  const slug = generateProjectSlug(rawInput);
  const runs = Array.isArray(llmOutput.runs) ? llmOutput.runs : [];

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

// ---------------------------------------------------------------------------
// Core decompose function
// ---------------------------------------------------------------------------

/**
 * @param {object} opts
 * @param {string} opts.raw_input — 用户原始需求文本
 * @param {string} [opts.project_type]
 * @param {object} [opts.classifier_result]
 * @param {object} [opts.registry]
 * @param {Set|Array|object} [opts.task_classes]
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
  let llmResult = await callLlm(prompt, config);
  let llmRaw = llmResult?.raw;
  let llmOutput = llmResult?.parsed;
  let model = llmResult?.model || "unknown";

  if (!llmOutput) {
    console.warn("[project_planner] LLM returned no parseable JSON, using fallback");
    const fb = buildFallbackPlan({ rawInput, projectType, workspaceRoot });
    fb._llm_raw = llmRaw;
    return fb;
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
    const retryResult = await callLlm(retryPrompt, config);
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

  plan.sorted = validation.sorted;
  plan.waves = validation.waves;
  plan._warnings = validation.warnings;
  plan._llm_raw = llmRaw;

  return plan;
}

// ---------------------------------------------------------------------------
// LLM call — supports Ollama (local) and MiniMax (cloud)
// ---------------------------------------------------------------------------

/**
 * 调用 LLM 进行拆解。优先使用 Ollama 本地模型（无超时限制），
 * 失败时 fallback 到 MiniMax cloud。
 */
async function callLlm(systemContent, config = {}) {
  const provider = config.project_planner_llm_provider || process.env.PLANNER_LLM_PROVIDER || "auto";
  const ollamaModel = config.project_planner_ollama_model || process.env.PLANNER_OLLAMA_MODEL || "gemma4:26b";
  const ollamaBase = process.env.OLLAMA_BASE_URL || "http://localhost:11434";

  // auto: 先试 Ollama，失败 fallback 到 MiniMax
  // ollama: 只用 Ollama
  // minimax: 只用 MiniMax
  if (provider === "ollama" || provider === "auto") {
    const ollamaResult = await callLlmOllama(systemContent, ollamaModel, ollamaBase);
    if (ollamaResult?.parsed) return ollamaResult;
    if (provider === "ollama") return ollamaResult; // 不 fallback
    console.warn("[project_planner] Ollama failed, falling back to MiniMax");
  }

  return callLlmMinimax(systemContent);
}

async function callLlmOllama(systemContent, model, baseUrl) {
  try {
    const messages = [
      { role: "system", content: systemContent },
      { role: "user", content: "请根据上述需求进行项目拆解，直接返回 JSON。" },
    ];
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 min for local
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        think: false,
        stream: false,
        options: { num_ctx: 32768, num_predict: 32768 },
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      console.warn(`[project_planner] Ollama error: ${data?.error || response.status}`);
      return { raw: null, parsed: null, model: null };
    }
    const raw = String(data.message?.content || "");
    const parsed = extractJson(raw);
    return { raw, parsed, model: `ollama/${model}` };
  } catch (err) {
    console.warn(`[project_planner] Ollama call failed: ${err.message}`);
    return { raw: null, parsed: null, model: null };
  }
}

async function callLlmMinimax(systemContent) {
  try {
    const messages = [
      { role: "system", content: systemContent },
      { role: "user", content: "请根据上述需求进行项目拆解，直接返回 JSON。" },
    ];
    const result = await qwenChat(messages, 300000, { max_tokens: 8192 });
    if (!result?.choices?.[0]?.message?.content) {
      return { raw: null, parsed: null, model: null };
    }
    const raw = result.choices[0].message.content;
    const parsed = extractJson(raw);
    return { raw, parsed, model: result.model || null };
  } catch (err) {
    console.error("[project_planner] MiniMax call failed:", err.message);
    return { raw: null, parsed: null, model: null };
  }
}

// 导出内部函数用于测试
export { extractJson, tryFixTruncatedJson, generateProjectSlug, buildDecompositionPrompt, buildFallbackPlan, assemblePlan };
