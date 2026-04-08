import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { runStaticChecks } from "../static_checks.js";

function writeFile(root, relPath, content) {
  const abs = path.join(root, relPath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content, "utf8");
}

function makeTempWorkspace() {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "static-cache-"));
  const taskDir = path.join(workspaceRoot, "artifacts", "runs", "run-1", "task_1");
  fs.mkdirSync(taskDir, { recursive: true });
  return { workspaceRoot, taskDir };
}

// Mock Redis that returns cached "pass" for any key
function mockRedisHit() {
  return {
    get: async () => "pass",
    set: async () => {},
    _getCalls: [],
    _setCalls: [],
  };
}

// Mock Redis that returns null (cache miss) and records set calls
function mockRedisMiss() {
  const setCalls = [];
  return {
    get: async () => null,
    set: async (key, value, mode, ttl) => { setCalls.push({ key, value, mode, ttl }); },
    _setCalls: setCalls,
  };
}

// Mock Redis that throws on get (simulates Redis down)
function mockRedisDown() {
  return {
    get: async () => { throw new Error("ECONNREFUSED"); },
    set: async () => { throw new Error("ECONNREFUSED"); },
  };
}

async function testCacheHit() {
  const { workspaceRoot, taskDir } = makeTempWorkspace();
  writeFile(workspaceRoot, "workspace/sandbox/data/valid.json", '{"ok": true}');

  const redis = mockRedisHit();
  const result = await runStaticChecks({
    workspaceRoot,
    filesChanged: ["workspace/sandbox/data/valid.json"],
    taskDir,
    redis,
  });

  assert.equal(result.checked, true);
  assert.equal(result.ok, true);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].cached, true, "record should be marked as cached");
  assert.equal(result.records[0].ok, true);
  assert.equal(result.records[0].kind, "json_parse");
  console.log("  ✓ cache hit: skips check, marks cached=true");
}

async function testCacheMissAndStore() {
  const { workspaceRoot, taskDir } = makeTempWorkspace();
  writeFile(workspaceRoot, "workspace/sandbox/data/good.json", '{"hello": "world"}');

  const redis = mockRedisMiss();
  const result = await runStaticChecks({
    workspaceRoot,
    filesChanged: ["workspace/sandbox/data/good.json"],
    taskDir,
    redis,
  });

  assert.equal(result.checked, true);
  assert.equal(result.ok, true);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].ok, true);
  // Should NOT be marked cached since it was a miss
  assert.ok(!result.records[0].cached, "record should not be marked cached on miss");
  // redis.set should have been called to store the result
  assert.ok(redis._setCalls.length > 0, "redis.set should be called to cache the result");
  assert.equal(redis._setCalls[0].value, "pass");
  assert.equal(redis._setCalls[0].ttl, 86400, "TTL should be 24h");
  console.log("  ✓ cache miss: runs check, stores result in Redis");
}

async function testRedisFallback() {
  const { workspaceRoot, taskDir } = makeTempWorkspace();
  writeFile(workspaceRoot, "workspace/sandbox/data/fallback.json", '{"still": "works"}');

  const redis = mockRedisDown();
  const result = await runStaticChecks({
    workspaceRoot,
    filesChanged: ["workspace/sandbox/data/fallback.json"],
    taskDir,
    redis,
  });

  // Should still succeed — Redis errors are non-fatal
  assert.equal(result.checked, true);
  assert.equal(result.ok, true);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].ok, true);
  assert.equal(result.records[0].kind, "json_parse");
  console.log("  ✓ Redis down: falls back to normal check, no crash");
}

async function main() {
  await testCacheHit();
  await testCacheMissAndStore();
  await testRedisFallback();
  console.log("static_checks_cache.test.js: all tests passed");
}

main().catch((err) => {
  console.error("static_checks_cache.test.js: failed");
  console.error(err);
  process.exit(1);
});
