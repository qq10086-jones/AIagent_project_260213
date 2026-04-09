import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import { resolveContext, buildContextBlock } from "../context_resolver.js";

let tmpDir;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ctx_resolver_test_"));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

function writeFile(relPath, content) {
  const abs = path.join(tmpDir, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

describe("resolveContext — JS imports", () => {
  it("discovers 1-depth import dependency", () => {
    writeFile("src/app.js", `import { foo } from './utils.js';\nconsole.log(foo);`);
    writeFile("src/utils.js", `export const foo = 1;`);

    const result = resolveContext({
      task_class: "be_modify",
      target_paths: ["src/app.js"],
      max_files: 10,
      max_tokens: 5000,
      dependency_depth: 1,
    }, tmpDir);

    assert.equal(result.status, "complete");
    assert.ok(result.files.length >= 2, `expected ≥2 files, got ${result.files.length}`);
    const paths = result.files.map((f) => f.path);
    assert.ok(paths.includes("src/app.js"));
    assert.ok(paths.includes("src/utils.js"));
    assert.equal(result.resolution_method, "import_graph");
    assert.ok(result.confidence > 0);
  });

  it("discovers require() style imports", () => {
    writeFile("lib/index.js", `const helper = require('./helper');\nmodule.exports = helper;`);
    writeFile("lib/helper.js", `module.exports = {};`);

    const result = resolveContext({
      target_paths: ["lib/index.js"],
      dependency_depth: 1,
    }, tmpDir);

    const paths = result.files.map((f) => f.path);
    assert.ok(paths.includes("lib/helper.js"));
  });

  it("respects max_files limit", () => {
    writeFile("a.js", `import './b.js'; import './c.js'; import './d.js';`);
    writeFile("b.js", "export default 1;");
    writeFile("c.js", "export default 2;");
    writeFile("d.js", "export default 3;");

    const result = resolveContext({
      target_paths: ["a.js"],
      max_files: 2,
      dependency_depth: 1,
    }, tmpDir);

    assert.ok(result.files.length <= 2);
    assert.equal(result.status, "partial");
  });

  it("respects dependency_depth=0 (no crawling)", () => {
    writeFile("src/main.js", `import './deep.js';`);
    writeFile("src/deep.js", `export const x = 1;`);

    const result = resolveContext({
      target_paths: ["src/main.js"],
      dependency_depth: 0,
    }, tmpDir);

    assert.equal(result.files.length, 1);
    assert.equal(result.files[0].path, "src/main.js");
  });

  it("resolves index files", () => {
    writeFile("src/app.js", `import './components';`);
    writeFile("src/components/index.js", `export const Button = 'btn';`);

    const result = resolveContext({
      target_paths: ["src/app.js"],
      dependency_depth: 1,
    }, tmpDir);

    const paths = result.files.map((f) => f.path);
    assert.ok(paths.some((p) => p.includes("components")));
  });

  it("skips node_modules imports", () => {
    writeFile("app.js", `import express from 'express';\nimport { foo } from './local.js';`);
    writeFile("local.js", `export const foo = 1;`);

    const result = resolveContext({
      target_paths: ["app.js"],
      dependency_depth: 1,
    }, tmpDir);

    const paths = result.files.map((f) => f.path);
    assert.ok(!paths.some((p) => p.includes("express")));
    assert.ok(paths.includes("local.js"));
  });
});

describe("resolveContext — Python imports", () => {
  it("discovers from...import dependency", () => {
    writeFile("main.py", `from utils import helper\nhelper()`);
    writeFile("utils.py", `def helper(): pass`);

    const result = resolveContext({
      target_paths: ["main.py"],
      dependency_depth: 1,
    }, tmpDir);

    const paths = result.files.map((f) => f.path);
    assert.ok(paths.includes("main.py"));
    assert.ok(paths.includes("utils.py"));
  });

  it("discovers import statement", () => {
    writeFile("app.py", `import config\nprint(config.KEY)`);
    writeFile("config.py", `KEY = "value"`);

    const result = resolveContext({
      target_paths: ["app.py"],
      dependency_depth: 1,
    }, tmpDir);

    const paths = result.files.map((f) => f.path);
    assert.ok(paths.includes("config.py"));
  });
});

describe("resolveContext — directory target", () => {
  it("scans directory for source files", () => {
    writeFile("src/a.js", "const x = 1;");
    writeFile("src/b.ts", "const y = 2;");
    writeFile("src/readme.md", "# readme");

    const result = resolveContext({
      target_paths: ["src"],
      dependency_depth: 0,
    }, tmpDir);

    const paths = result.files.map((f) => f.path);
    assert.ok(paths.includes("src/a.js"));
    assert.ok(paths.includes("src/b.ts"));
    // .md files are not source — should not be included
    assert.ok(!paths.includes("src/readme.md"));
  });
});

describe("resolveContext — missing paths", () => {
  it("records missing paths", () => {
    const result = resolveContext({
      target_paths: ["nonexistent/file.js"],
      dependency_depth: 0,
    }, tmpDir);

    assert.ok(result.missing_context.includes("nonexistent/file.js"));
  });
});

describe("resolveContext — token budget", () => {
  it("returns partial when token budget exceeded", () => {
    // Create files that exceed a small token budget
    writeFile("big.js", "x".repeat(2000));
    writeFile("small.js", "y");

    const result = resolveContext({
      target_paths: ["big.js", "small.js"],
      max_tokens: 100,
      dependency_depth: 0,
    }, tmpDir);

    // big.js alone is ~500 tokens (2000/4), which exceeds 100
    // First file is always included, then it stops
    assert.ok(result.files.length <= 2);
  });
});

describe("buildContextBlock", () => {
  it("builds injectable context from response", () => {
    writeFile("src/app.js", "const x = 1;");
    const response = {
      files: [{ path: "src/app.js", role: "target", token_estimate: 5 }],
    };
    const block = buildContextBlock(response, tmpDir);
    assert.ok(block.includes("[Auto-Resolved Context"));
    assert.ok(block.includes("src/app.js"));
    assert.ok(block.includes("const x = 1;"));
  });

  it("returns empty for null response", () => {
    assert.equal(buildContextBlock(null, tmpDir), "");
    assert.equal(buildContextBlock({ files: [] }, tmpDir), "");
  });
});
