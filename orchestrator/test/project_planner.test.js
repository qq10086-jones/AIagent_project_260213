import { describe, it, mock } from "node:test";
import assert from "node:assert/strict";
import {
  extractJson,
  generateProjectSlug,
  buildDecompositionPrompt,
  buildFallbackPlan,
  assemblePlan,
} from "../src/vnext/project_planner.js";

describe("extractJson", () => {
  it("extracts from ```json fence", () => {
    const text = 'Some text\n```json\n{"runs": [1,2]}\n```\nMore text';
    const result = extractJson(text);
    assert.deepEqual(result, { runs: [1, 2] });
  });

  it("extracts from bare JSON", () => {
    const text = 'Here is the plan: {"modules": [], "runs": []}';
    const result = extractJson(text);
    assert.deepEqual(result, { modules: [], runs: [] });
  });

  it("returns null for non-JSON", () => {
    assert.equal(extractJson("no json here"), null);
  });

  it("handles nested braces", () => {
    const text = '{"a": {"b": 1}, "c": [{"d": 2}]}';
    const result = extractJson(text);
    assert.deepEqual(result, { a: { b: 1 }, c: [{ d: 2 }] });
  });
});

describe("generateProjectSlug", () => {
  it("generates slug from Chinese input", () => {
    const slug = generateProjectSlug("做一套客诉管理系统");
    assert.ok(slug.length > 0);
    assert.ok(!slug.includes(" "));
  });

  it("generates slug from English input", () => {
    const slug = generateProjectSlug("Build a customer complaint system");
    assert.equal(slug, "build-a-customer-complaint-system");
  });

  it("truncates long input", () => {
    const slug = generateProjectSlug("a".repeat(100));
    assert.ok(slug.length <= 40);
  });

  it("falls back to 'project' on empty", () => {
    assert.equal(generateProjectSlug(""), "project");
  });
});

describe("buildDecompositionPrompt", () => {
  it("injects task_classes dynamically", () => {
    const prompt = buildDecompositionPrompt({
      rawInput: "Build a dashboard",
      taskClasses: new Set(["fe_create", "be_create"]),
      projectType: "generic_app",
      workspaceRoot: "workspace/sandbox/dashboard/",
    });
    assert.ok(prompt.includes("fe_create"));
    assert.ok(prompt.includes("be_create"));
    assert.ok(prompt.includes("generic_app"));
    assert.ok(prompt.includes("workspace/sandbox/dashboard/"));
    assert.ok(prompt.includes("Build a dashboard"));
  });

  it("does not hardcode specific project references", () => {
    const prompt = buildDecompositionPrompt({
      rawInput: "任意需求",
      taskClasses: ["be_create"],
      projectType: "coding_task",
      workspaceRoot: "workspace/sandbox/test/",
    });
    // Prompt should be generic, not mentioning specific systems
    assert.ok(!prompt.includes("客诉"));
    assert.ok(!prompt.includes("complaint"));
  });
});

describe("buildFallbackPlan", () => {
  it("returns a single-run plan with _fallback flag", () => {
    const plan = buildFallbackPlan({
      rawInput: "do something complex",
      projectType: "generic_app",
      workspaceRoot: "workspace/sandbox/test/",
    });
    assert.equal(plan._fallback, true);
    assert.equal(plan.runs.length, 1);
    assert.equal(plan.runs[0].run_key, "R-01");
    assert.equal(plan.runs[0].task_class, "be_create");
    assert.ok(plan.project_id.startsWith("proj-fallback"));
  });
});

describe("assemblePlan", () => {
  it("builds complete plan from LLM output", () => {
    const llmOutput = {
      modules: [{ module_id: "mod-api", title: "API", description: "REST API" }],
      runs: [
        {
          run_key: "R-01", module_id: "mod-api", task_class: "be_create",
          title: "Create API", prompt: "Implement a RESTful API with user authentication, CRUD operations, and input validation for the core data model",
          target_paths: ["workspace/sandbox/app/api/"],
          depends_on: [], shared_context: { from_runs: [], artifacts: [] },
          estimated_complexity: "medium",
          acceptance_criteria: ["AC-R01-1: GET /api/users returns 200"],
        },
        {
          run_key: "R-02", module_id: "mod-api", task_class: "fe_create",
          title: "Create UI", prompt: "Build a responsive frontend with user list, detail view, and authentication forms that consume the backend API endpoints",
          target_paths: ["workspace/sandbox/app/ui/"],
          depends_on: ["R-01"], shared_context: { from_runs: ["R-01"], artifacts: ["plan/interfaces.md"] },
          estimated_complexity: "medium",
          acceptance_criteria: ["AC-R02-1: User list renders"],
        },
      ],
      tech_stack_hints: { backend: "Express", frontend: "React" },
    };

    const plan = assemblePlan({
      llmOutput,
      rawInput: "Build a user management app",
      projectType: "generic_app",
      workspaceRoot: "workspace/sandbox/app/",
      model: "MiniMax-M2.7",
      config: { max_parallel_runs: 2, failure_policy: "stop_dependents" },
    });

    assert.ok(plan.project_id.startsWith("proj-"));
    assert.equal(plan.runs.length, 2);
    assert.equal(plan.decomposition_model, "MiniMax-M2.7");
    assert.deepEqual(plan.dependency_graph, { "R-01": [], "R-02": ["R-01"] });
    assert.equal(plan.execution_strategy.max_parallel_runs, 2);
    assert.equal(plan.execution_strategy.failure_policy, "stop_dependents");
    assert.equal(plan.project_constraints.project_type, "generic_app");
    assert.deepEqual(plan.project_constraints.tech_stack_hints, { backend: "Express", frontend: "React" });
  });
});
