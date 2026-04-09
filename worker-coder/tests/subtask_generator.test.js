import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import { validateSubtaskRequest, executeSubtask, createSubtaskTracker } from "../subtask_generator.js";

let tmpDir;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "subtask_test_"));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("validateSubtaskRequest", () => {
  it("rejects null request", () => {
    assert.equal(validateSubtaskRequest(null).ok, false);
  });

  it("rejects invalid parent_step_id", () => {
    const result = validateSubtaskRequest({
      parent_step_id: "pm_spec",
      subtask_type: "install_dependency",
      target_paths: ["src/"],
    });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("not allowed"));
  });

  it("rejects invalid subtask_type", () => {
    const result = validateSubtaskRequest({
      parent_step_id: "impl_be",
      subtask_type: "delete_file",
      target_paths: ["src/"],
    });
    assert.equal(result.ok, false);
  });

  it("rejects depth > 0", () => {
    const result = validateSubtaskRequest({
      parent_step_id: "impl_be",
      subtask_type: "install_dependency",
      target_paths: ["src/"],
      depth: 1,
    });
    assert.equal(result.ok, false);
    assert.ok(result.error.includes("depth"));
  });

  it("rejects empty target_paths", () => {
    const result = validateSubtaskRequest({
      parent_step_id: "impl_be",
      subtask_type: "create_config",
      target_paths: [],
    });
    assert.equal(result.ok, false);
  });

  it("accepts valid install_dependency request", () => {
    const result = validateSubtaskRequest({
      parent_step_id: "impl_be",
      subtask_type: "install_dependency",
      target_paths: ["workspace/sandbox/"],
    });
    assert.equal(result.ok, true);
  });

  it("accepts valid create_config for impl_fe", () => {
    const result = validateSubtaskRequest({
      parent_step_id: "impl_fe",
      subtask_type: "create_config",
      target_paths: ["workspace/sandbox/"],
    });
    assert.equal(result.ok, true);
  });
});

describe("executeSubtask — create_config", () => {
  it("creates config file within target_paths", async () => {
    const targetDir = path.join(tmpDir, "src");
    fs.mkdirSync(targetDir, { recursive: true });

    const result = await executeSubtask({
      parent_step_id: "impl_be",
      subtask_type: "create_config",
      target_paths: ["src/"],
      params: {
        file_path: "src/config.json",
        content: '{"port": 3000}',
      },
    }, tmpDir);

    assert.equal(result.ok, true);
    assert.equal(result.result.action, "created");
    assert.ok(fs.existsSync(path.join(tmpDir, "src/config.json")));
  });

  it("refuses to create file outside target_paths", async () => {
    const targetDir = path.join(tmpDir, "src");
    fs.mkdirSync(targetDir, { recursive: true });

    const result = await executeSubtask({
      parent_step_id: "impl_be",
      subtask_type: "create_config",
      target_paths: ["src/"],
      params: {
        file_path: "outside/config.json",
        content: "{}",
      },
    }, tmpDir);

    assert.equal(result.ok, false);
    assert.ok(result.error.includes("outside target_paths"));
  });

  it("does not overwrite existing file", async () => {
    const targetDir = path.join(tmpDir, "src");
    fs.mkdirSync(targetDir, { recursive: true });
    fs.writeFileSync(path.join(targetDir, "existing.json"), '{"existing": true}');

    const result = await executeSubtask({
      parent_step_id: "impl_fe",
      subtask_type: "create_config",
      target_paths: ["src/"],
      params: {
        file_path: "src/existing.json",
        content: '{"new": true}',
      },
    }, tmpDir);

    assert.equal(result.ok, true);
    assert.equal(result.result.action, "already_exists");
    // Original content unchanged
    const content = fs.readFileSync(path.join(targetDir, "existing.json"), "utf8");
    assert.ok(content.includes("existing"));
  });
});

describe("executeSubtask — install_dependency", () => {
  it("rejects invalid package_name (shell injection)", async () => {
    const result = await executeSubtask({
      parent_step_id: "impl_be",
      subtask_type: "install_dependency",
      target_paths: ["src/"],
      params: {
        package_name: "express; rm -rf /",
        package_manager: "npm",
      },
    }, tmpDir);

    assert.equal(result.ok, false);
    assert.ok(result.error.includes("invalid package_name"));
  });

  it("rejects unsupported package_manager", async () => {
    const result = await executeSubtask({
      parent_step_id: "impl_be",
      subtask_type: "install_dependency",
      target_paths: ["src/"],
      params: {
        package_name: "express",
        package_manager: "yarn",
      },
    }, tmpDir);

    assert.equal(result.ok, false);
    assert.ok(result.error.includes("unsupported"));
  });

  it("rejects empty package_name", async () => {
    const result = await executeSubtask({
      parent_step_id: "impl_be",
      subtask_type: "install_dependency",
      target_paths: ["src/"],
      params: { package_name: "" },
    }, tmpDir);

    assert.equal(result.ok, false);
    assert.ok(result.error.includes("package_name required"));
  });
});

describe("createSubtaskTracker", () => {
  it("allows first subtask", () => {
    const tracker = createSubtaskTracker();
    assert.equal(tracker.canSpawn("impl_be"), true);
    assert.equal(tracker.getCount("impl_be"), 0);
  });

  it("blocks after max subtasks reached", () => {
    const tracker = createSubtaskTracker();
    tracker.record("impl_be");
    assert.equal(tracker.canSpawn("impl_be"), false);
    assert.equal(tracker.getCount("impl_be"), 1);
  });

  it("tracks steps independently", () => {
    const tracker = createSubtaskTracker();
    tracker.record("impl_be");
    assert.equal(tracker.canSpawn("impl_be"), false);
    assert.equal(tracker.canSpawn("impl_fe"), true);
  });
});
