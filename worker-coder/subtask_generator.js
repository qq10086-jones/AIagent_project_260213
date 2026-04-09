/**
 * subtask_generator.js — Phase 3 (P3-3): Scoped Dynamic Subtask Generation.
 *
 * Allows impl_be and impl_fe steps to generate at most 1 child subtask during
 * execution. Subtask types are whitelisted to:
 *   - install_dependency: install missing npm/pip package
 *   - create_config: create a missing config file within target_paths
 *
 * Constraints (v3.2 Section 5.4 P3-3):
 *   - Max depth = 1 (subtasks cannot spawn subtasks)
 *   - Cannot modify arch_design output
 *   - Cannot access paths outside target_paths
 *   - Max 1 subtask per parent step
 */
import fs from "fs";
import path from "path";
import { exec } from "child_process";

const ALLOWED_SUBTASK_TYPES = new Set(["install_dependency", "create_config"]);
const ALLOWED_PARENT_STEPS = new Set(["impl_be", "impl_fe"]);
const MAX_SUBTASKS_PER_STEP = 1;

/**
 * Validate a subtask request.
 * @param {Object} request
 * @param {string} request.parent_step_id - must be impl_be or impl_fe
 * @param {string} request.subtask_type - install_dependency or create_config
 * @param {string} request.description
 * @param {string[]} request.target_paths - inherited from parent
 * @param {Object} [request.params]
 * @param {number} [request.depth=0] - current depth (must be 0)
 * @returns {{ ok: boolean, error?: string }}
 */
export function validateSubtaskRequest(request) {
  if (!request || typeof request !== "object") {
    return { ok: false, error: "request is null or not an object" };
  }
  const parentStep = String(request.parent_step_id || "").trim();
  if (!ALLOWED_PARENT_STEPS.has(parentStep)) {
    return { ok: false, error: `parent_step_id '${parentStep}' not allowed (only impl_be/impl_fe)` };
  }
  const subtaskType = String(request.subtask_type || "").trim();
  if (!ALLOWED_SUBTASK_TYPES.has(subtaskType)) {
    return { ok: false, error: `subtask_type '${subtaskType}' not allowed (only install_dependency/create_config)` };
  }
  const depth = Number(request.depth || 0);
  if (depth > 0) {
    return { ok: false, error: "subtask depth > 0 not allowed (no nested subtasks)" };
  }
  if (!Array.isArray(request.target_paths) || request.target_paths.length === 0) {
    return { ok: false, error: "target_paths required" };
  }
  return { ok: true };
}

/**
 * Execute a subtask synchronously.
 * @param {Object} request - validated subtask request
 * @param {string} workspaceRoot
 * @returns {Promise<{ ok: boolean, result?: Object, error?: string }>}
 */
export async function executeSubtask(request, workspaceRoot) {
  const validation = validateSubtaskRequest(request);
  if (!validation.ok) {
    return { ok: false, error: validation.error };
  }

  const subtaskType = String(request.subtask_type || "");
  const params = request.params || {};

  if (subtaskType === "install_dependency") {
    return executeInstallDependency(params, workspaceRoot, request.target_paths);
  }
  if (subtaskType === "create_config") {
    return executeCreateConfig(params, workspaceRoot, request.target_paths);
  }

  return { ok: false, error: `unhandled subtask_type: ${subtaskType}` };
}

/**
 * Install a dependency via npm install (within target_paths scope).
 */
async function executeInstallDependency(params, workspaceRoot, targetPaths) {
  const packageName = String(params.package_name || "").trim();
  if (!packageName) {
    return { ok: false, error: "package_name required" };
  }
  // Validate package name: no shell injection
  if (!/^[@a-zA-Z0-9][\w./-]*$/.test(packageName)) {
    return { ok: false, error: `invalid package_name: ${packageName}` };
  }
  const manager = String(params.package_manager || "npm").trim().toLowerCase();
  if (!["npm", "pip"].includes(manager)) {
    return { ok: false, error: `unsupported package_manager: ${manager}` };
  }

  // Find package.json or requirements.txt within target_paths
  const absRoot = path.resolve(workspaceRoot);
  let installDir = absRoot;
  for (const tp of targetPaths) {
    const absPath = path.resolve(absRoot, tp);
    if (!absPath.startsWith(absRoot)) continue;
    const pkgJson = manager === "npm"
      ? path.join(absPath, "package.json")
      : path.join(absPath, "requirements.txt");
    if (fs.existsSync(pkgJson)) {
      installDir = absPath;
      break;
    }
    // Check parent dir
    const parentPkgJson = manager === "npm"
      ? path.join(path.dirname(absPath), "package.json")
      : path.join(path.dirname(absPath), "requirements.txt");
    if (fs.existsSync(parentPkgJson)) {
      installDir = path.dirname(absPath);
      break;
    }
  }

  const cmd = manager === "npm"
    ? `npm install --save ${packageName}`
    : `pip install ${packageName}`;

  return new Promise((resolve) => {
    exec(cmd, { cwd: installDir, timeout: 60000 }, (error, stdout, stderr) => {
      if (error) {
        resolve({
          ok: false,
          error: `${manager} install failed: ${error.message}`,
          result: { stdout: stdout?.slice(0, 2000), stderr: stderr?.slice(0, 2000) },
        });
      } else {
        resolve({
          ok: true,
          result: {
            package_name: packageName,
            manager,
            stdout: stdout?.slice(0, 1000),
          },
        });
      }
    });
  });
}

/**
 * Create a config file within target_paths scope.
 */
async function executeCreateConfig(params, workspaceRoot, targetPaths) {
  const filePath = String(params.file_path || "").trim();
  if (!filePath) {
    return { ok: false, error: "file_path required" };
  }

  const absRoot = path.resolve(workspaceRoot);
  const absPath = path.resolve(absRoot, filePath);

  // Security: must be within workspace root
  if (!absPath.startsWith(absRoot)) {
    return { ok: false, error: "file_path outside workspace root" };
  }

  // Security: must be within one of the target_paths
  const withinScope = targetPaths.some((tp) => {
    const absTarget = path.resolve(absRoot, tp);
    return absPath.startsWith(absTarget);
  });
  if (!withinScope) {
    return { ok: false, error: `file_path '${filePath}' outside target_paths scope` };
  }

  // Don't overwrite existing files
  if (fs.existsSync(absPath)) {
    return { ok: true, result: { file_path: filePath, action: "already_exists" } };
  }

  const content = String(params.content || "{}");
  try {
    fs.mkdirSync(path.dirname(absPath), { recursive: true });
    fs.writeFileSync(absPath, content, "utf8");
    return {
      ok: true,
      result: { file_path: filePath, action: "created", bytes: content.length },
    };
  } catch (err) {
    return { ok: false, error: `create_config failed: ${err.message}` };
  }
}

/**
 * Track subtask count per step to enforce MAX_SUBTASKS_PER_STEP limit.
 */
export function createSubtaskTracker() {
  const counts = new Map();

  return {
    canSpawn(stepId) {
      return (counts.get(stepId) || 0) < MAX_SUBTASKS_PER_STEP;
    },
    record(stepId) {
      counts.set(stepId, (counts.get(stepId) || 0) + 1);
    },
    getCount(stepId) {
      return counts.get(stepId) || 0;
    },
  };
}
