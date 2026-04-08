import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import { canFix, applySurgicalPatches } from "../surgical_patch.js";

function makeTmpWorkspace() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "sp-test-"));
  fs.mkdirSync(path.join(root, "workspace", "src"), { recursive: true });
  return root;
}

describe("canFix", () => {
  it("returns fixable=false for empty records", () => {
    assert.deepEqual(canFix({ records: [] }), { fixable: false, candidates: [] });
  });

  it("returns fixable=false for passing records", () => {
    const result = canFix({ records: [{ file: "a.json", kind: "json_parse", ok: true, exit_code: 0, stderr: "" }] });
    assert.equal(result.fixable, false);
  });

  it("detects json trailing comma", () => {
    const result = canFix({ records: [{
      file: "a.json", kind: "json_parse", ok: false, exit_code: 1,
      stderr: "Expected double-quoted property name in JSON at position 42 (line 3 column 1)",
    }] });
    assert.equal(result.fixable, true);
    assert.equal(result.candidates.length, 1);
  });

  it("detects js unclosed bracket", () => {
    const result = canFix({ records: [{
      file: "a.js", kind: "node_syntax", ok: false, exit_code: 1,
      stderr: "SyntaxError: Unexpected end of input",
    }] });
    assert.equal(result.fixable, true);
    assert.equal(result.candidates[0].fixType, "js_unclosed_bracket");
  });

  it("detects python indentation error", () => {
    const result = canFix({ records: [{
      file: "a.py", kind: "py_compile", ok: false, exit_code: 1,
      stderr: "IndentationError: unexpected indent",
    }] });
    assert.equal(result.fixable, true);
    assert.equal(result.candidates[0].fixType, "py_indentation");
  });

  it("returns fixable=false for unknown errors", () => {
    const result = canFix({ records: [{
      file: "a.js", kind: "node_syntax", ok: false, exit_code: 1,
      stderr: "SyntaxError: Identifier 'foo' has already been declared",
    }] });
    assert.equal(result.fixable, false);
  });
});

describe("applySurgicalPatches", () => {
  it("fixes JSON trailing comma", () => {
    const root = makeTmpWorkspace();
    const jsonPath = path.join(root, "workspace", "src", "data.json");
    fs.writeFileSync(jsonPath, '{"a": 1, "b": 2,}', "utf8");

    const result = applySurgicalPatches({
      executionWorkspaceRoot: root,
      records: [{
        file: "workspace/src/data.json", kind: "json_parse", ok: false, exit_code: 1,
        stderr: "Expected double-quoted property name in JSON at position 18",
      }],
      allowedTargetPaths: ["workspace/src"],
      workspaceRoot: root,
    });

    assert.equal(result.attempted, true);
    assert.equal(result.success, true);
    assert.equal(result.patches_applied.length, 1);
    // Verify the file is now valid JSON
    const fixed = fs.readFileSync(jsonPath, "utf8");
    assert.doesNotThrow(() => JSON.parse(fixed));
  });

  it("fixes JS unclosed bracket", () => {
    const root = makeTmpWorkspace();
    const jsPath = path.join(root, "workspace", "src", "app.js");
    fs.writeFileSync(jsPath, 'function foo() {\n  console.log("hi");\n', "utf8");

    const result = applySurgicalPatches({
      executionWorkspaceRoot: root,
      records: [{
        file: "workspace/src/app.js", kind: "node_syntax", ok: false, exit_code: 1,
        stderr: "SyntaxError: Unexpected end of input",
      }],
      allowedTargetPaths: ["workspace/src"],
      workspaceRoot: root,
    });

    assert.equal(result.attempted, true);
    assert.equal(result.success, true);
    const fixed = fs.readFileSync(jsPath, "utf8");
    assert.ok(fixed.includes("}"));
  });

  it("fixes JSON missing comma between values", () => {
    const root = makeTmpWorkspace();
    const jsonPath = path.join(root, "workspace", "src", "data.json");
    fs.writeFileSync(jsonPath, '{\n  "a": 1\n  "b": 2\n}', "utf8");

    const result = applySurgicalPatches({
      executionWorkspaceRoot: root,
      records: [{
        file: "workspace/src/data.json", kind: "json_parse", ok: false, exit_code: 1,
        stderr: "Expected ',' or '}' after property value in JSON at position 12",
      }],
      allowedTargetPaths: ["workspace/src"],
      workspaceRoot: root,
    });

    assert.equal(result.attempted, true);
    if (result.success) {
      const fixed = fs.readFileSync(jsonPath, "utf8");
      assert.doesNotThrow(() => JSON.parse(fixed));
    }
  });

  it("rejects file outside scope", () => {
    const root = makeTmpWorkspace();
    fs.mkdirSync(path.join(root, "workspace", "other"), { recursive: true });
    fs.writeFileSync(path.join(root, "workspace", "other", "x.json"), '{"a":1,}', "utf8");

    const result = applySurgicalPatches({
      executionWorkspaceRoot: root,
      records: [{
        file: "workspace/other/x.json", kind: "json_parse", ok: false, exit_code: 1,
        stderr: "trailing comma",
      }],
      allowedTargetPaths: ["workspace/src"],
      workspaceRoot: root,
    });

    assert.equal(result.attempted, true);
    assert.equal(result.patches_applied.length, 0);
  });

  it("no-op when all records ok", () => {
    const root = makeTmpWorkspace();
    const result = applySurgicalPatches({
      executionWorkspaceRoot: root,
      records: [{ file: "workspace/src/a.js", kind: "node_syntax", ok: true, exit_code: 0, stderr: "" }],
      allowedTargetPaths: ["workspace/src"],
      workspaceRoot: root,
    });

    assert.equal(result.attempted, false);
  });

  it("fixes python mixed indentation", () => {
    const root = makeTmpWorkspace();
    const pyPath = path.join(root, "workspace", "src", "app.py");
    fs.writeFileSync(pyPath, "def foo():\n\t    print('hi')\n", "utf8");

    const result = applySurgicalPatches({
      executionWorkspaceRoot: root,
      records: [{
        file: "workspace/src/app.py", kind: "py_compile", ok: false, exit_code: 1,
        stderr: "IndentationError: unexpected indent",
      }],
      allowedTargetPaths: ["workspace/src"],
      workspaceRoot: root,
    });

    assert.equal(result.attempted, true);
    assert.equal(result.success, true);
    const fixed = fs.readFileSync(pyPath, "utf8");
    assert.ok(!fixed.includes("\t"));
  });
});
