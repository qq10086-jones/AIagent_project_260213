// Phase 0: Shared feature flags for Redis caching layer.
// Used by worker-coder (static_checks) and orchestrator (repo_context_service).

export function isCacheEnabled() {
  const raw = String(process.env.REDIS_CACHE_ENABLED ?? "true").trim().toLowerCase();
  return !["0", "false", "no", "off"].includes(raw);
}
