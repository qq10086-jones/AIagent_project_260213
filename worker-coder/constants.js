// Centralized constants for worker-coder
// All magic numbers and configuration defaults live here.

// --- Timeouts (milliseconds) ---
export const EXEC_TIMEOUT_MS = 30_000;
export const GIT_EXEC_TIMEOUT_MS = 20_000;

// --- Timeouts (seconds) ---
export const DEFAULT_MAX_RUNTIME_S = 600;
export const MIN_WALL_CLOCK_S = 60;
export const MAX_WALL_CLOCK_S = 3600;
export const FALLBACK_WALL_CLOCK_S = 300;

// --- Retry budgets ---
export const MIN_ATTEMPTS = 1;
export const MAX_ATTEMPTS = 3;
export const DEFAULT_ATTEMPTS = 1;
export const MIN_SAME_ERROR_REPEAT = 1;
export const MAX_SAME_ERROR_REPEAT = 3;
export const DEFAULT_SAME_ERROR_REPEAT = 2;

// --- Worker stream ---
export const MAX_RESULT_OUTPUT_BYTES = 48 * 1024;
export const STALE_TASK_MIN_IDLE_MS = 60_000;

// --- Smoke test ---
export const SMOKE_TEST_PORT = 13099;

// --- Artifact directories ---
export const ARTIFACTS_DIR = "artifacts";
export const RUNS_DIR = "runs";

// --- Supported providers ---
export const SUPPORTED_PROVIDERS = new Set(["auto", "opencode", "codex"]);
export const DEFAULT_PROVIDER_RESOLVED = "opencode";

// --- Error codes ---
export const ErrorCode = {
    PROVIDER_UNAVAILABLE: "E_PROVIDER_UNAVAILABLE",
    UNAUTHORIZED_WRITE: "E_UNAUTHORIZED_WRITE",
    WALL_CLOCK_TIMEOUT: "E_WALL_CLOCK_TIMEOUT",
    STATIC_CHECK_FAILED: "E_STATIC_CHECK_FAILED",
    VERIFICATION_FAILED: "E_VERIFICATION_FAILED",
    DELEGATE_FAILED: "E_DELEGATE_FAILED",
    PROMOTION_FAILED: "E_PROMOTION_FAILED",
    HANDOFF_INVALID: "E_HANDOFF_INVALID",
    WORKFLOW_ARTIFACT_INVALID: "E_WORKFLOW_ARTIFACT_INVALID",
    COMMAND_VALIDATION_FAILED: "E_COMMAND_VALIDATION_FAILED",
};
