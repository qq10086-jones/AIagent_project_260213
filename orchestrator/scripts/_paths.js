import path from "path";
import { fileURLToPath } from "url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));

export const SCRIPTS_DIR = MODULE_DIR;
export const ORCHESTRATOR_ROOT = path.resolve(SCRIPTS_DIR, "..");
export const REPO_ROOT = path.resolve(ORCHESTRATOR_ROOT, "..");

export function resolveRepoPath(...segments) {
  return path.resolve(REPO_ROOT, ...segments);
}

export function resolveOrchestratorPath(...segments) {
  return path.resolve(ORCHESTRATOR_ROOT, ...segments);
}

export function resolveCanaryInputPath(fileName) {
  return resolveOrchestratorPath("canary_inputs", fileName);
}

export function resolveOrchestratorArtifactPath(...segments) {
  return resolveOrchestratorPath("artifacts", ...segments);
}

export function getDefaultWorkspaceRoot() {
  return REPO_ROOT;
}
