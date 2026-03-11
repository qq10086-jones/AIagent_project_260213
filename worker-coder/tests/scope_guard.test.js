import assert from "node:assert/strict";

import {
  isProtectedRoot,
  normalizeRelPath,
  validateAllowedTargetPaths,
  validateChangedFilesWithinScope,
  validateRequestedWrite,
} from "../scope_guard.js";

function main() {
  assert.equal(normalizeRelPath("\\sandbox\\crm_site\\app.js"), "sandbox/crm_site/app.js");
  assert.equal(isProtectedRoot("configs/runtime"), true);
  assert.equal(isProtectedRoot("sandbox/crm_site"), false);

  const allowed = validateAllowedTargetPaths({
    workspaceRoot: process.cwd(),
    allowedTargetPaths: ["sandbox/crm_site/app.js"],
  });
  assert.equal(allowed.ok, true, JSON.stringify(allowed));

  const blocked = validateAllowedTargetPaths({
    workspaceRoot: process.cwd(),
    allowedTargetPaths: ["configs/runtime/runtime_defaults.json"],
  });
  assert.equal(blocked.ok, false);
  assert.match(blocked.error, /protected target path/);

  const inScope = validateRequestedWrite({
    workspaceRoot: process.cwd(),
    targetPath: "sandbox/crm_site/app.js",
    allowedTargetPaths: ["sandbox/crm_site"],
  });
  assert.equal(inScope.ok, true, JSON.stringify(inScope));

  const outOfScope = validateRequestedWrite({
    workspaceRoot: process.cwd(),
    targetPath: "sandbox/other/file.js",
    allowedTargetPaths: ["sandbox/crm_site"],
  });
  assert.equal(outOfScope.ok, false);
  assert.match(outOfScope.error, /outside allowed target_paths/);

  const changedInScope = validateChangedFilesWithinScope({
    filesChanged: ["sandbox/crm_site/app.js"],
    allowedTargetPaths: ["sandbox/crm_site"],
  });
  assert.equal(changedInScope.ok, true, JSON.stringify(changedInScope));

  const changedOutOfScope = validateChangedFilesWithinScope({
    filesChanged: ["sandbox/crm_site/app.js", "sandbox/other/file.js"],
    allowedTargetPaths: ["sandbox/crm_site"],
  });
  assert.equal(changedOutOfScope.ok, false);
  assert.match(changedOutOfScope.error, /outside scope/);

  console.log("scope_guard.test.js: all tests passed");
}

try {
  main();
} catch (err) {
  console.error("scope_guard.test.js: failed");
  console.error(err);
  process.exit(1);
}
