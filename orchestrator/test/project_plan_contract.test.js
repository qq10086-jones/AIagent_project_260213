import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { validateProjectPlan, topoSort, computeWaves, loadTaskClasses } from "../src/vnext/project_plan_contract.js";

// 最小合法 plan 工厂
function makePlan(overrides = {}) {
  const base = {
    modules: [{ module_id: "mod-a", title: "Module A", description: "desc" }],
    runs: [
      {
        run_key: "R-01", module_id: "mod-a", task_class: "be_create",
        title: "Backend API", prompt: "Create a RESTful API with CRUD operations for the core data model including validation and error handling",
        target_paths: ["workspace/sandbox/proj/backend/"],
        depends_on: [], shared_context: { from_runs: [], artifacts: [] },
        estimated_complexity: "medium",
        acceptance_criteria: ["AC-R01-1: POST /api/items returns 201"],
      },
      {
        run_key: "R-02", module_id: "mod-a", task_class: "fe_create",
        title: "Frontend Page", prompt: "Create a frontend page that displays items from the API with list view, detail view, and create form components",
        target_paths: ["workspace/sandbox/proj/frontend/"],
        depends_on: ["R-01"], shared_context: { from_runs: ["R-01"], artifacts: ["handoff/be_to_fe.json"] },
        estimated_complexity: "medium",
        acceptance_criteria: ["AC-R02-1: List page renders items"],
      },
    ],
  };
  return { ...base, ...overrides };
}

const TASK_CLASSES = ["fe_create", "fe_modify", "be_create", "bug_fix", "artifact_completion"];

describe("validateProjectPlan", () => {
  it("accepts a valid minimal 2-run plan", () => {
    const result = validateProjectPlan(makePlan(), { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, true, `errors: ${result.errors.join("; ")}`);
    assert.equal(result.errors.length, 0);
    assert.ok(result.sorted);
    assert.deepEqual(result.sorted, ["R-01", "R-02"]);
    assert.ok(result.waves);
    assert.equal(result.waves.length, 2);
  });

  it("accepts custom task_class passed as Set", () => {
    const plan = makePlan();
    plan.runs[0].task_class = "custom_deploy";
    plan.runs[1].task_class = "custom_deploy";
    const result = validateProjectPlan(plan, { taskClasses: new Set(["custom_deploy"]) });
    assert.equal(result.ok, true, `errors: ${result.errors.join("; ")}`);
  });

  it("C-01: rejects plan with fewer than 2 runs", () => {
    const plan = makePlan({ runs: [makePlan().runs[0]] });
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-01")));
  });

  it("C-01: rejects plan exceeding max_runs", () => {
    const runs = [];
    for (let i = 1; i <= 5; i++) {
      runs.push({
        ...makePlan().runs[0],
        run_key: `R-${String(i).padStart(2, "0")}`,
        target_paths: [`workspace/sandbox/proj/mod${i}/`],
        depends_on: [],
      });
    }
    const result = validateProjectPlan({ ...makePlan(), runs }, { taskClasses: TASK_CLASSES, maxRuns: 3 });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-01") && e.includes("max_runs")));
  });

  it("C-02: rejects unregistered task_class", () => {
    const plan = makePlan();
    plan.runs[0].task_class = "deploy_production";
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-02") && e.includes("deploy_production")));
  });

  it("C-03: rejects duplicate run_key", () => {
    const plan = makePlan();
    plan.runs[1].run_key = "R-01";
    plan.runs[1].target_paths = ["workspace/sandbox/proj/dup/"];
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-03") && e.includes("duplicate")));
  });

  it("C-03: rejects malformed run_key", () => {
    const plan = makePlan();
    plan.runs[0].run_key = "step1";
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-03") && e.includes("format")));
  });

  it("C-04: rejects cyclic dependency graph", () => {
    const plan = makePlan();
    plan.runs[0].depends_on = ["R-02"];
    plan.runs[1].depends_on = ["R-01"];
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-04") && e.includes("cycle")));
  });

  it("C-05: rejects depends_on referencing non-existent run", () => {
    const plan = makePlan();
    plan.runs[1].depends_on = ["R-99"];
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-05") && e.includes("R-99")));
  });

  it("C-06: rejects overlapping target_paths", () => {
    const plan = makePlan();
    plan.runs[1].target_paths = ["workspace/sandbox/proj/backend/sub/"];
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-06") && e.includes("overlaps")));
  });

  it("C-07: rejects run with no acceptance_criteria", () => {
    const plan = makePlan();
    plan.runs[0].acceptance_criteria = [];
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-07")));
  });

  it("C-08: rejects run with prompt too short", () => {
    const plan = makePlan();
    plan.runs[0].prompt = "do it";
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-08")));
  });

  it("C-09: warns on non-standard artifact paths and strips them", () => {
    const plan = makePlan();
    plan.runs[1].shared_context.artifacts = ["handoff/be_to_fe.json", "custom/my_file.txt"];
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.ok(result.warnings.some((w) => w.includes("C-09") && w.includes("custom/my_file.txt")));
    // Non-standard artifact should be stripped, only whitelisted ones remain
    assert.deepEqual(plan.runs[1].shared_context.artifacts, ["handoff/be_to_fe.json"]);
  });

  it("C-09: rejects path traversal in artifacts", () => {
    const plan = makePlan();
    plan.runs[0].shared_context = { from_runs: [], artifacts: ["../../etc/passwd"] };
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.equal(result.ok, false);
    assert.ok(result.errors.some((e) => e.includes("C-09") && e.includes("path traversal")));
  });

  it("C-10: warns on orphan module", () => {
    const plan = makePlan();
    plan.modules.push({ module_id: "mod-orphan", title: "Orphan", description: "unused" });
    const result = validateProjectPlan(plan, { taskClasses: TASK_CLASSES });
    assert.ok(result.warnings.some((w) => w.includes("C-10") && w.includes("mod-orphan")));
  });
});

describe("topoSort", () => {
  it("sorts linear chain correctly", () => {
    const sorted = topoSort({ "R-01": [], "R-02": ["R-01"], "R-03": ["R-02"] });
    assert.deepEqual(sorted, ["R-01", "R-02", "R-03"]);
  });

  it("returns null for cycle", () => {
    const sorted = topoSort({ "R-01": ["R-02"], "R-02": ["R-01"] });
    assert.equal(sorted, null);
  });

  it("handles diamond dependency", () => {
    const sorted = topoSort({ "R-01": [], "R-02": ["R-01"], "R-03": ["R-01"], "R-04": ["R-02", "R-03"] });
    assert.ok(sorted);
    assert.ok(sorted.indexOf("R-01") < sorted.indexOf("R-02"));
    assert.ok(sorted.indexOf("R-01") < sorted.indexOf("R-03"));
    assert.ok(sorted.indexOf("R-02") < sorted.indexOf("R-04"));
    assert.ok(sorted.indexOf("R-03") < sorted.indexOf("R-04"));
  });
});

describe("computeWaves", () => {
  it("computes correct waves for parallel branches", () => {
    const waves = computeWaves({ "R-01": [], "R-04": [], "R-02": ["R-01"], "R-05": ["R-04"], "R-07": ["R-02", "R-05"] });
    assert.ok(waves);
    // Wave 0: R-01, R-04 (no deps)
    assert.ok(waves[0].includes("R-01"));
    assert.ok(waves[0].includes("R-04"));
    // Wave 1: R-02, R-05
    assert.ok(waves[1].includes("R-02"));
    assert.ok(waves[1].includes("R-05"));
    // Wave 2: R-07
    assert.deepEqual(waves[2], ["R-07"]);
  });
});

describe("loadTaskClasses", () => {
  it("loads from Set (passthrough)", () => {
    const input = new Set(["custom_class", "another"]);
    const s = loadTaskClasses(input);
    assert.ok(s.has("custom_class"));
    assert.ok(s.has("another"));
    assert.equal(s, input); // same reference — no copy
  });

  it("loads from array", () => {
    const s = loadTaskClasses(["a", "b"]);
    assert.ok(s.has("a"));
    assert.ok(s.has("b"));
  });

  it("loads from object with task_classes field", () => {
    const s = loadTaskClasses({ task_classes: ["x", "y"] });
    assert.ok(s.has("x"));
  });

  it("returns default on invalid input", () => {
    const s = loadTaskClasses(undefined);
    assert.ok(s.has("be_create"));
  });
});
