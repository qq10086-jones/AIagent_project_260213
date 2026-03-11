import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { runPatchAutoCommit } from "../git_side_effects.js";

function makeTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "git-side-effects-"));
}

async function testUsesStructuredGitArgsAndWritesArtifact() {
  const root = makeTempDir();
  const taskDir = path.join(root, "artifacts");
  fs.mkdirSync(taskDir, { recursive: true });
  fs.mkdirSync(path.join(root, ".git"));

  const calls = [];
  const runner = async (command, args, cwd) => {
    calls.push([command, args, cwd]);
    if (args.join(" ") === "diff --name-only --diff-filter=U") {
      return { ok: true, stdout: "", stderr: "", exitCode: 0 };
    }
    if (args.join(" ") === "add -- src/app \"quote\".js") {
      return { ok: true, stdout: "", stderr: "", exitCode: 0 };
    }
    if (args.join(" ") === "diff --cached --quiet -- src/app \"quote\".js") {
      return { ok: false, stdout: "", stderr: "", exitCode: 1 };
    }
    if (args[0] === "commit") {
      return { ok: true, stdout: "[main abc123] ok", stderr: "", exitCode: 0 };
    }
    throw new Error(`Unexpected args: ${args.join(" ")}`);
  };

  const result = await runPatchAutoCommit({
    workspaceRoot: root,
    filePath: 'src/app "quote".js',
    taskId: 'task-"7"',
    taskDir,
    execFileImpl: runner,
  });

  assert.equal(result.status, "committed");
  assert.equal(result.committed, true);
  assert.equal(calls[1][0], "git");
  assert.deepEqual(calls[1][1], ["add", "--", 'src/app "quote".js']);
  assert.deepEqual(calls[3][1], ["commit", "-m", 'coding_agent: task task-"7" - updated src/app "quote".js']);
  assert.ok(result.logPath);
  const payload = JSON.parse(fs.readFileSync(result.logPath, "utf8"));
  assert.equal(payload.status, "committed");
}

async function testSkipsWhenMergeConflictsExist() {
  const root = makeTempDir();
  const taskDir = path.join(root, "artifacts");
  fs.mkdirSync(taskDir, { recursive: true });
  fs.mkdirSync(path.join(root, ".git"));

  const calls = [];
  const runner = async (command, args, cwd) => {
    calls.push([command, args, cwd]);
    return { ok: true, stdout: "src/conflicted.js\n", stderr: "", exitCode: 0 };
  };

  const result = await runPatchAutoCommit({
    workspaceRoot: root,
    filePath: "src/app.js",
    taskId: "task-2",
    taskDir,
    execFileImpl: runner,
  });

  assert.equal(result.status, "skipped");
  assert.equal(result.reason, "unresolved_merge_conflicts");
  assert.equal(calls.length, 1);
}

async function main() {
  await testUsesStructuredGitArgsAndWritesArtifact();
  await testSkipsWhenMergeConflictsExist();
  console.log("git_side_effects.test.js: all tests passed");
}

main().catch((err) => {
  console.error("git_side_effects.test.js: failed");
  console.error(err);
  process.exit(1);
});
