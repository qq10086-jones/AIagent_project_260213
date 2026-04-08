import { describe, it } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";

import { readWorkspaceFile, listWorkspaceDir, readScopedFile } from "../native_file_tools.js";

function makeTmpWorkspace() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nft-test-"));
  fs.mkdirSync(path.join(root, "workspace", "src"), { recursive: true });
  fs.writeFileSync(path.join(root, "workspace", "src", "app.js"), "console.log('hello');", "utf8");
  fs.writeFileSync(path.join(root, "workspace", "src", "secret.js"), 'const token = "sk-AAAAAABBBBBBCCCCCCDDDDDD";', "utf8");
  fs.mkdirSync(path.join(root, ".git"), { recursive: true });
  fs.writeFileSync(path.join(root, ".git", "config"), "secret-git-config", "utf8");
  return root;
}

describe("readWorkspaceFile", () => {
  it("reads a file successfully", () => {
    const root = makeTmpWorkspace();
    const result = readWorkspaceFile({ workspaceRoot: root, relPath: "workspace/src/app.js" });
    assert.equal(result.ok, true);
    assert.ok(result.content.includes("console.log"));
    assert.equal(result.truncated, false);
  });

  it("blocks path traversal", () => {
    const root = makeTmpWorkspace();
    const result = readWorkspaceFile({ workspaceRoot: root, relPath: "../../etc/passwd" });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("path traversal") || result.error.includes("protected") || result.error.includes("not found"));
  });

  it("blocks protected root (.git)", () => {
    const root = makeTmpWorkspace();
    const result = readWorkspaceFile({ workspaceRoot: root, relPath: ".git/config" });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("protected"));
  });

  it("redacts sensitive content", () => {
    const root = makeTmpWorkspace();
    const result = readWorkspaceFile({ workspaceRoot: root, relPath: "workspace/src/secret.js" });
    assert.equal(result.ok, true);
    assert.ok(result.content.includes("[REDACTED_OPENAI_KEY]"));
    assert.ok(!result.content.includes("sk-AAAAAABBBBBBCCCCCCDDDDDD"));
  });

  it("truncates to maxBytes", () => {
    const root = makeTmpWorkspace();
    const result = readWorkspaceFile({ workspaceRoot: root, relPath: "workspace/src/app.js", maxBytes: 5 });
    assert.equal(result.ok, true);
    assert.equal(result.truncated, true);
    assert.ok(result.content.length <= 5);
  });

  it("returns error for nonexistent file", () => {
    const root = makeTmpWorkspace();
    const result = readWorkspaceFile({ workspaceRoot: root, relPath: "workspace/nope.js" });
    assert.equal(result.ok, false);
  });
});

describe("listWorkspaceDir", () => {
  it("lists directory entries", () => {
    const root = makeTmpWorkspace();
    const result = listWorkspaceDir({ workspaceRoot: root, relPath: "workspace/src" });
    assert.equal(result.ok, true);
    assert.ok(result.entries.length >= 2);
    const names = result.entries.map((e) => e.name);
    assert.ok(names.includes("app.js"));
  });

  it("blocks protected root", () => {
    const root = makeTmpWorkspace();
    const result = listWorkspaceDir({ workspaceRoot: root, relPath: ".git" });
    assert.equal(result.ok, false);
  });

  it("respects maxEntries", () => {
    const root = makeTmpWorkspace();
    const result = listWorkspaceDir({ workspaceRoot: root, relPath: "workspace/src", maxEntries: 1 });
    assert.equal(result.ok, true);
    assert.equal(result.entries.length, 1);
    assert.equal(result.truncated, true);
  });
});

describe("readScopedFile", () => {
  it("reads within scope", () => {
    const root = makeTmpWorkspace();
    const result = readScopedFile({
      workspaceRoot: root,
      relPath: "workspace/src/app.js",
      allowedTargetPaths: ["workspace/src"],
    });
    assert.equal(result.ok, true);
  });

  it("rejects out of scope", () => {
    const root = makeTmpWorkspace();
    const result = readScopedFile({
      workspaceRoot: root,
      relPath: "workspace/src/app.js",
      allowedTargetPaths: ["workspace/other"],
    });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("out of scope"));
  });

  it("rejects empty target_paths", () => {
    const root = makeTmpWorkspace();
    const result = readScopedFile({
      workspaceRoot: root,
      relPath: "workspace/src/app.js",
      allowedTargetPaths: [],
    });
    assert.equal(result.ok, false);
  });
});
