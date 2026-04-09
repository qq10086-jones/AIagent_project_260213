#!/usr/bin/env node
/**
 * test_planner.js — 真实 LLM 端到端拆解验证脚本
 *
 * 用法:
 *   node scripts/test_planner.js "做一套客诉管理系统"
 *   node scripts/test_planner.js --all          # 跑小/中/大三个内置场景
 *   node scripts/test_planner.js                # 默认跑 --all
 *
 * 需要环境变量:
 *   MINIMAX_API_KEY 或 QWEN_API_KEY
 */

import path from "path";
import { fileURLToPath } from "url";
import { decompose } from "../src/vnext/project_planner.js";
import { validateProjectPlan, loadTaskClasses } from "../src/vnext/project_plan_contract.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TASK_CLASSES_PATH = path.resolve(__dirname, "../../configs/registry/worker_coding_task_classes.json");

const BUILTIN_SCENARIOS = [
  {
    label: "小型 (2-6 runs)",
    input: "做一个待办事项应用，支持添加、完成、删除、筛选待办",
    expectedRunRange: [2, 8],
  },
  {
    label: "中型 (4-9 runs)",
    input: "做一套客诉管理系统，包含客诉提交、状态流转、处理记录、统计报表，前后端分离",
    expectedRunRange: [3, 12],
  },
  {
    label: "大型 (6-12 runs)",
    input: "做一个多租户 SaaS 电商平台，包含用户认证 RBAC、商品管理、购物车、订单、支付集成、库存管理、租户后台、数据报表，前后端分离",
    expectedRunRange: [4, 12],
  },
];

function printSeparator() {
  console.log("─".repeat(72));
}

function printPlanSummary(plan) {
  console.log(`  project_id     : ${plan.project_id}`);
  console.log(`  project_title  : ${plan.project_title}`);
  console.log(`  model          : ${plan.decomposition_model}`);
  console.log(`  confidence     : ${plan.decomposition_confidence}`);
  console.log(`  runs           : ${plan.runs?.length || 0}`);
  console.log(`  modules        : ${plan.modules?.length || 0}`);
  console.log(`  fallback       : ${plan._fallback || false}`);
  if (plan.waves) {
    console.log(`  waves          : ${plan.waves.length} [${plan.waves.map((w) => w.join(",")).join(" | ")}]`);
  }
  if (plan._warnings?.length > 0) {
    console.log(`  warnings       : ${plan._warnings.length}`);
    plan._warnings.forEach((w) => console.log(`    - ${w}`));
  }
  console.log();
  if (plan.runs) {
    for (const run of plan.runs) {
      const deps = run.depends_on?.length > 0 ? `depends: [${run.depends_on.join(", ")}]` : "depends: []";
      console.log(`  ${run.run_key} [${run.task_class}] ${run.title}  ${deps}`);
      if (run.acceptance_criteria?.length > 0) {
        run.acceptance_criteria.forEach((ac) => console.log(`    - ${ac}`));
      }
    }
  }
}

async function runScenario({ label, input, expectedRunRange, taskClasses }) {
  printSeparator();
  console.log(`SCENARIO: ${label}`);
  console.log(`INPUT: "${input}"`);
  console.log();

  const startMs = Date.now();
  const plan = await decompose({
    raw_input: input,
    task_classes: taskClasses,
    config: {
      project_planner_max_runs: 12,
      project_planner_max_parallel_runs: 2,
      project_planner_failure_policy: "stop_dependents",
      project_planner_llm_provider: process.env.PLANNER_LLM_PROVIDER || "auto",
      project_planner_ollama_model: process.env.PLANNER_OLLAMA_MODEL || "gemma4:26b",
    },
  });
  const elapsedMs = Date.now() - startMs;

  console.log(`  elapsed        : ${elapsedMs}ms`);
  printPlanSummary(plan);
  console.log();

  // 验证
  const validation = validateProjectPlan(plan, { taskClasses, maxRuns: 12 });
  const runCount = plan.runs?.length || 0;
  const isFallback = plan._fallback === true;

  const checks = [];

  // Check 1: contract validation
  if (isFallback) {
    checks.push({ name: "contract_valid", pass: true, detail: "fallback plan (skipped)" });
    // Show validation errors that caused fallback
    if (plan._validation_errors?.length > 0) {
      console.log("  [DIAG] fallback validation errors:");
      plan._validation_errors.forEach((e) => console.log(`    - ${e}`));
    }
    if (plan._llm_raw) {
      console.log(`  [DIAG] LLM raw output (first 500 chars): ${String(plan._llm_raw).slice(0, 500)}`);
    }
  } else if (validation.ok) {
    checks.push({ name: "contract_valid", pass: true, detail: `${validation.errors.length} errors` });
  } else {
    checks.push({ name: "contract_valid", pass: false, detail: validation.errors.join("; ") });
  }

  // Check 2: run count in expected range
  if (expectedRunRange && !isFallback) {
    const [min, max] = expectedRunRange;
    const inRange = runCount >= min && runCount <= max;
    checks.push({
      name: "run_count_range",
      pass: inRange,
      detail: `${runCount} runs (expected ${min}-${max})`,
    });
  }

  // Check 3: DAG acyclic (sorted exists)
  if (!isFallback) {
    checks.push({
      name: "dag_acyclic",
      pass: Array.isArray(plan.sorted) && plan.sorted.length === runCount,
      detail: plan.sorted ? `sorted: ${plan.sorted.length} nodes` : "MISSING",
    });
  }

  // Check 4: no fallback (LLM actually produced a plan)
  checks.push({
    name: "llm_produced_plan",
    pass: !isFallback,
    detail: isFallback ? "FALLBACK — LLM failed to produce valid plan" : "OK",
  });

  // Check 5: all task_classes registered
  if (!isFallback) {
    const invalidClasses = (plan.runs || [])
      .filter((r) => !taskClasses.has(r.task_class))
      .map((r) => `${r.run_key}:${r.task_class}`);
    checks.push({
      name: "task_classes_valid",
      pass: invalidClasses.length === 0,
      detail: invalidClasses.length > 0 ? `invalid: ${invalidClasses.join(", ")}` : "OK",
    });
  }

  // Print results
  let allPass = true;
  for (const c of checks) {
    const icon = c.pass ? "PASS" : "FAIL";
    if (!c.pass) allPass = false;
    console.log(`  [${icon}] ${c.name}: ${c.detail}`);
  }
  console.log();

  return { label, allPass, checks, plan, elapsedMs, isFallback };
}

async function main() {
  const args = process.argv.slice(2);
  const taskClasses = loadTaskClasses(TASK_CLASSES_PATH);

  // Check LLM key (not needed for Ollama)
  const provider = process.env.PLANNER_LLM_PROVIDER || "auto";
  if (provider !== "ollama" && !process.env.MINIMAX_API_KEY && !process.env.QWEN_API_KEY) {
    console.error("ERROR: MINIMAX_API_KEY or QWEN_API_KEY required (or set PLANNER_LLM_PROVIDER=ollama)");
    process.exit(1);
  }

  let scenarios;
  if (args.length === 0 || args[0] === "--all") {
    scenarios = BUILTIN_SCENARIOS;
  } else {
    scenarios = [{ label: "custom", input: args.join(" "), expectedRunRange: [2, 12] }];
  }

  console.log(`\ntest_planner.js — ${scenarios.length} scenario(s)\n`);

  const results = [];
  for (const scenario of scenarios) {
    const result = await runScenario({ ...scenario, taskClasses });
    results.push(result);
  }

  // Summary
  printSeparator();
  console.log("SUMMARY");
  console.log();
  let totalPass = 0;
  let totalFail = 0;
  for (const r of results) {
    const icon = r.allPass ? "PASS" : "FAIL";
    const runCount = r.plan.runs?.length || 0;
    console.log(`  [${icon}] ${r.label}: ${runCount} runs, ${r.elapsedMs}ms${r.isFallback ? " (FALLBACK)" : ""}`);
    if (r.allPass) totalPass++;
    else totalFail++;
  }
  console.log();
  console.log(`  Total: ${totalPass} pass, ${totalFail} fail`);
  printSeparator();

  if (totalFail > 0) process.exit(1);
}

main().catch((err) => {
  console.error("test_planner.js fatal error:", err);
  process.exit(1);
});
