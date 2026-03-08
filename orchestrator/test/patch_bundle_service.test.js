import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import { createPatchBundleService } from "../src/domain/patch_bundle_service.js";

function makeWorkspace() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "ocn-patch-bundle-"));
}

function writeFile(workspaceRoot, relativePath, content) {
  const targetPath = path.join(workspaceRoot, relativePath);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, content, "utf8");
}

function readFile(workspaceRoot, relativePath) {
  return fs.readFileSync(path.join(workspaceRoot, relativePath), "utf8");
}

test("applyPatchBundle applies insert, replace, delete, and create_file deterministically", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(
    workspaceRoot,
    "src/server.js",
    [
      "const express = require('express');",
      "app.use(express.json());",
      "const oldBlockStart = true;",
      "const legacy = 'on';",
      "const oldBlockEnd = true;",
      "const removeStart = true;",
      "const removeMe = true;",
      "const removeEnd = true;",
      "",
    ].join("\n")
  );

  const service = createPatchBundleService({ workspaceRoot });
  const result = service.applyPatchBundle({
    bundle_id: "bundle-1",
    step_id: "impl_be",
    mode: "structured_patch",
    summary: "Patch existing backend file and add helper.",
    operations: [
      {
        type: "insert_after_anchor",
        target_file: "src/server.js",
        anchor: "app.use(express.json());",
        content: "\napp.use(requestLogger);",
      },
      {
        type: "replace_range",
        target_file: "src/server.js",
        anchor_start: "const oldBlockStart = true;",
        anchor_end: "const oldBlockEnd = true;",
        content: "const replacement = 'ok';",
      },
      {
        type: "delete_range",
        target_file: "src/server.js",
        anchor_start: "const removeStart = true;",
        anchor_end: "const removeEnd = true;",
      },
      {
        type: "create_file",
        target_file: "src/request_logger.js",
        file_content: "export function requestLogger(_req, _res, next) { next(); }\n",
      },
    ],
  });

  assert.equal(result.ok, true);
  assert.equal(result.operation_count, 4);
  assert.deepEqual(result.written_files.sort(), ["src/request_logger.js", "src/server.js"]);
  assert.match(readFile(workspaceRoot, "src/server.js"), /app\.use\(requestLogger\);/);
  assert.match(readFile(workspaceRoot, "src/server.js"), /const replacement = 'ok';/);
  assert.doesNotMatch(readFile(workspaceRoot, "src/server.js"), /removeMe/);
  assert.match(readFile(workspaceRoot, "src/request_logger.js"), /requestLogger/);
});

test("applyPatchBundle resolves later anchors against prior same-file operations", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(
    workspaceRoot,
    "src/app.js",
    ["const boot = true;", "const marker = 'A';", "module.exports = {};", ""].join("\n")
  );

  const service = createPatchBundleService({ workspaceRoot });
  service.applyPatchBundle({
    bundle_id: "bundle-2",
    step_id: "impl_fe",
    mode: "structured_patch",
    summary: "Same file multi-op patch.",
    operations: [
      {
        type: "insert_after_anchor",
        target_file: "src/app.js",
        anchor: "const marker = 'A';",
        content: "\nconst markerB = 'B';",
      },
      {
        type: "insert_after_anchor",
        target_file: "src/app.js",
        anchor: "const markerB = 'B';",
        content: "\nconst markerC = 'C';",
      },
    ],
  });

  const content = readFile(workspaceRoot, "src/app.js");
  assert.match(content, /const markerB = 'B';/);
  assert.match(content, /const markerC = 'C';/);
});

test("applyPatchBundle returns typed error with operation index on anchor failure", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(workspaceRoot, "src/app.js", "const stable = true;\n");

  const service = createPatchBundleService({ workspaceRoot });
  assert.throws(
    () =>
      service.applyPatchBundle({
        bundle_id: "bundle-3",
        step_id: "impl_fe",
        mode: "structured_patch",
        summary: "Missing anchor should fail.",
        operations: [
          {
            type: "insert_after_anchor",
            target_file: "src/app.js",
            anchor: "const missing = true;",
            content: "\nconst x = 1;",
          },
        ],
      }),
    (err) => err?.code === "PATCH_ANCHOR_NOT_FOUND" && err?.operation_index === 0
  );
});

test("applyPatchBundle returns typed error when prior operation invalidates later anchor", () => {
  const workspaceRoot = makeWorkspace();
  writeFile(
    workspaceRoot,
    "src/app.js",
    ["const start = true;", "const old = true;", "const end = true;", ""].join("\n")
  );

  const service = createPatchBundleService({ workspaceRoot });
  assert.throws(
    () =>
      service.applyPatchBundle({
        bundle_id: "bundle-4",
        step_id: "impl_fe",
        mode: "structured_patch",
        summary: "Later anchor should fail after earlier replacement.",
        operations: [
          {
            type: "replace_range",
            target_file: "src/app.js",
            anchor_start: "const start = true;",
            anchor_end: "const end = true;",
            content: "const newBlock = true;",
          },
          {
            type: "insert_after_anchor",
            target_file: "src/app.js",
            anchor: "const old = true;",
            content: "\nconst shouldFail = true;",
          },
        ],
      }),
    (err) => err?.code === "PATCH_ANCHOR_NOT_FOUND" && err?.operation_index === 1
  );
});

test("applyPatchBundle rejects path traversal outside workspace root", () => {
  const workspaceRoot = makeWorkspace();
  const service = createPatchBundleService({ workspaceRoot });

  assert.throws(
    () =>
      service.applyPatchBundle({
        bundle_id: "bundle-5",
        step_id: "impl_be",
        mode: "structured_patch",
        summary: "Traversal should fail.",
        operations: [
          {
            type: "create_file",
            target_file: "../escape.js",
            file_content: "console.log('bad');\n",
          },
        ],
      }),
    (err) => err?.code === "PATCH_PATH_TRAVERSAL" && err?.operation_index === 0
  );
});
