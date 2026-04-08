import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { createRepoContextService } from "../src/domain/repo_context_service.js";

function makeWorkspace() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "repo-ctx-cache-"));
  fs.writeFileSync(path.join(root, "package.json"), JSON.stringify({ name: "test-app" }), "utf8");
  fs.mkdirSync(path.join(root, "src"), { recursive: true });
  fs.writeFileSync(path.join(root, "src", "index.js"), "export const main = 1;\n", "utf8");
  // Initialize a git repo so getGitHeadSha works
  return root;
}

async function initGit(cwd) {
  const { exec } = await import("child_process");
  return new Promise((resolve) => {
    exec('git init && git add -A && git commit -m "init" --allow-empty', { cwd, timeout: 10000 }, (err) => {
      resolve(!err);
    });
  });
}

function mockRedisStore() {
  const store = new Map();
  const getCalls = [];
  const setCalls = [];
  return {
    get: async (key) => { getCalls.push(key); return store.get(key) || null; },
    set: async (key, value, mode, ttl) => { setCalls.push({ key, value, mode, ttl }); store.set(key, value); },
    _getCalls: getCalls,
    _setCalls: setCalls,
    _store: store,
  };
}

function mockRedisDown() {
  return {
    get: async () => { throw new Error("ECONNREFUSED"); },
    set: async () => { throw new Error("ECONNREFUSED"); },
  };
}

async function testCacheHitOnSecondCall() {
  const workspaceRoot = makeWorkspace();
  const gitOk = await initGit(workspaceRoot);
  if (!gitOk) {
    console.log("  ⚠ skipping cache hit test (git init failed)");
    return;
  }

  const redis = mockRedisStore();
  const svc = createRepoContextService({ workspaceRoot, redis });

  // First call — cache miss, should store
  const map1 = await svc.buildRepoMap({ targetPaths: ["src"] });
  assert.ok(map1.candidate_files.length > 0, "should have candidate files");
  assert.ok(redis._setCalls.length > 0, "first call should store to cache");

  // Second call — cache hit
  const prevGetCount = redis._getCalls.length;
  const map2 = await svc.buildRepoMap({ targetPaths: ["src"] });
  assert.ok(redis._getCalls.length > prevGetCount, "second call should check cache");
  assert.deepEqual(map2.candidate_files, map1.candidate_files, "cached result should match original");
  console.log("  ✓ cache hit: second call returns cached result");
}

async function testCacheMissStoresResult() {
  const workspaceRoot = makeWorkspace();
  const gitOk = await initGit(workspaceRoot);
  if (!gitOk) {
    console.log("  ⚠ skipping cache miss test (git init failed)");
    return;
  }

  const redis = mockRedisStore();
  const svc = createRepoContextService({ workspaceRoot, redis });

  const packet = await svc.buildContextPacket({
    role: "backend",
    stepId: "impl_be",
    targetPaths: ["src"],
    memoryHints: ["test hint"],
  });

  assert.equal(packet.step_id, "impl_be");
  assert.equal(packet.role, "backend");
  assert.ok(redis._setCalls.length > 0, "should store context_packet to cache");
  assert.equal(redis._setCalls[0].mode, "EX", "should use EX for TTL");
  assert.equal(redis._setCalls[0].ttl, 3600, "TTL should be 1 hour");
  console.log("  ✓ cache miss: stores result with 1h TTL");
}

async function testRedisFallback() {
  const workspaceRoot = makeWorkspace();
  const redis = mockRedisDown();
  const svc = createRepoContextService({ workspaceRoot, redis });

  // Should still succeed — Redis errors are non-fatal
  const map = await svc.buildRepoMap({ targetPaths: ["src"] });
  assert.ok(map.workspace_root, "should return valid repo map despite Redis failure");
  assert.ok(Array.isArray(map.candidate_files));

  const packet = await svc.buildContextPacket({
    role: "backend",
    stepId: "impl_be",
    targetPaths: ["src"],
  });
  assert.equal(packet.step_id, "impl_be");
  console.log("  ✓ Redis down: falls back to uncached computation");
}

async function testNoRedisPassthrough() {
  const workspaceRoot = makeWorkspace();
  // No redis passed — should work exactly like before
  const svc = createRepoContextService({ workspaceRoot });

  const map = await svc.buildRepoMap({ targetPaths: ["src"] });
  assert.ok(map.workspace_root);
  assert.ok(Array.isArray(map.candidate_files));
  console.log("  ✓ no redis: works as passthrough (no caching)");
}

async function main() {
  await testCacheHitOnSecondCall();
  await testCacheMissStoresResult();
  await testRedisFallback();
  await testNoRedisPassthrough();
  console.log("repo_context_cache.test.js: all tests passed");
}

main().catch((err) => {
  console.error("repo_context_cache.test.js: failed");
  console.error(err);
  process.exit(1);
});
