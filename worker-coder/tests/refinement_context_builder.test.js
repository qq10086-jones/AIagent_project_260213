import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import { validateLineage, buildRefinementContext } from "../refinement_context_builder.js";

function makeTmpWorkspace() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "rc-test-"));
  fs.mkdirSync(path.join(root, "workspace", "src"), { recursive: true });
  fs.writeFileSync(path.join(root, "workspace", "src", "app.js"), 'console.log("hello");', "utf8");
  fs.writeFileSync(path.join(root, "workspace", "src", "utils.js"), "export const add = (a, b) => a + b;", "utf8");
  return root;
}

describe("validateLineage", () => {
  it("accepts valid lineage", () => {
    const result = validateLineage({
      parent_task_id: "task_abc123",
      refinement_round: 1,
      refinement_instruction: "fix the bug",
    });
    assert.equal(result.ok, true);
  });

  it("rejects null lineage", () => {
    assert.equal(validateLineage(null).ok, false);
  });

  it("rejects missing parent_task_id", () => {
    const result = validateLineage({ refinement_round: 1 });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("parent_task_id"));
  });

  it("rejects round > 5", () => {
    const result = validateLineage({ parent_task_id: "task_1", refinement_round: 6 });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("exceeds max"));
  });

  it("rejects round < 1", () => {
    const result = validateLineage({ parent_task_id: "task_1", refinement_round: 0 });
    assert.equal(result.ok, false);
  });
});

describe("buildRefinementContext", () => {
  it("builds context block with target files", () => {
    const root = makeTmpWorkspace();
    const result = buildRefinementContext({
      lineage: {
        parent_task_id: "task_abc",
        parent_run_id: "run_xyz",
        refinement_round: 1,
        refinement_instruction: "change console.log to console.error",
      },
      workspaceRoot: root,
      targetPaths: ["workspace/src/app.js"],
    });

    assert.ok(result.contextBlock.includes("[Refinement Re-entry Round 1]"));
    assert.ok(result.contextBlock.includes("task_abc"));
    assert.ok(result.contextBlock.includes("console.log"));
    assert.ok(result.contextBlock.includes("change console.log to console.error"));
    assert.ok(result.contextBlock.includes("Rules:"));
    assert.equal(result.diagnostics.round, 1);
    assert.equal(result.diagnostics.files_included, 1);
  });

  it("returns empty context for invalid lineage", () => {
    const root = makeTmpWorkspace();
    const result = buildRefinementContext({
      lineage: null,
      workspaceRoot: root,
      targetPaths: ["workspace/src"],
    });
    assert.equal(result.contextBlock, "");
    assert.ok(result.diagnostics.error);
  });

  it("respects maxContextBytes", () => {
    const root = makeTmpWorkspace();
    const result = buildRefinementContext({
      lineage: {
        parent_task_id: "task_1",
        refinement_round: 2,
        refinement_instruction: "fix",
      },
      workspaceRoot: root,
      targetPaths: ["workspace/src/app.js", "workspace/src/utils.js"],
      maxContextBytes: 10,
    });
    // Should still produce a context block but with limited file content
    assert.ok(result.contextBlock.includes("[Refinement Re-entry Round 2]"));
    assert.ok(result.diagnostics.total_bytes <= 10 || result.diagnostics.files_included <= 1);
  });

  it("includes inherited_context files", () => {
    const root = makeTmpWorkspace();
    const result = buildRefinementContext({
      lineage: {
        parent_task_id: "task_1",
        refinement_round: 1,
        refinement_instruction: "add error handling",
        inherited_context: {
          files_changed: ["workspace/src/utils.js"],
        },
      },
      workspaceRoot: root,
      targetPaths: ["workspace/src"],
    });
    assert.ok(result.contextBlock.includes("utils.js"));
    assert.ok(result.contextBlock.includes("parent changed"));
  });
});
