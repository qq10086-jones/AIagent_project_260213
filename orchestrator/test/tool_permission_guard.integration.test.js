import test from "node:test";
import assert from "node:assert/strict";

import { validateToolPermission } from "../src/vnext/tool_permission_guard.js";

test("tool permission guard allows current registry role names", () => {
  assert.equal(validateToolPermission("pm", "coding.delegate").allowed, true);
  assert.equal(validateToolPermission("architect", "coding.delegate").allowed, true);
  assert.equal(validateToolPermission("frontend", "bash.execute").allowed, true);
  assert.equal(validateToolPermission("backend", "database.query").allowed, true);
  assert.equal(validateToolPermission("qa", "coding.execute").allowed, true);
});

test("tool permission guard denies unauthorized combinations", () => {
  const denied = validateToolPermission("pm", "bash.execute");
  assert.equal(denied.allowed, false);
  assert.match(String(denied.reason || ""), /not permitted/);
});
