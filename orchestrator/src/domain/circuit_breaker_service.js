/**
 * WS-25-05  Automated Circuit-Breaker Service
 *
 * Monitors rolling window of gated-parallel run outcomes.
 * Activates force-sequential if threshold is breached.
 * Does NOT auto-recover — manual operator reset required.
 */

import { readFileSync, writeFileSync, renameSync } from "fs";
import { join, dirname } from "path";

function loadJson(p) {
  try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; }
}

// Write via tmp + rename for atomic disk update.  If the write fails the
// original file is untouched; there is no partial-write window.
function saveJson(p, value) {
  const tmp = join(dirname(p), "." + Date.now() + ".tmp.json");
  try {
    writeFileSync(tmp, JSON.stringify(value, null, 2), "utf8");
    renameSync(tmp, p);
  } catch (err) {
    try { writeFileSync(tmp, "", "utf8"); renameSync(tmp, tmp + ".dead"); } catch { /* best-effort cleanup */ }
    throw new Error(`circuit-breaker: failed to persist config to ${p}: ${err.message}`);
  }
}

/**
 * Evaluate whether the circuit-breaker should activate based on recent results.
 *
 * @param {object[]} recentResults - array of run result objects with { partial_failure, rollback_triggered }
 * @param {object}   config        - circuit_breaker config block from production_parallel_rollout.json
 * @returns {{ should_activate: boolean, trigger_metric?: string, threshold?: number, observed_value?: number }}
 */
export function evaluateCircuitBreaker(recentResults, config = {}) {
  const windowSize = config.rolling_window_size ?? 100;
  const pfThreshold = (config.partial_failure_rate_threshold_pct ?? 25) / 100;
  const rollbackThreshold = config.rollback_trigger_event_threshold ?? 3;

  const window = recentResults.slice(-windowSize);
  if (window.length === 0) return { should_activate: false };

  const pfCount = window.filter((r) => r.partial_failure === true).length;
  const pfRate = pfCount / window.length;
  if (pfRate > pfThreshold) {
    return {
      should_activate: true,
      trigger_metric: "partial_failure_rate",
      threshold: pfThreshold,
      observed_value: pfRate,
    };
  }

  const rollbackCount = window.filter((r) => r.rollback_triggered === true).length;
  if (rollbackCount >= rollbackThreshold) {
    return {
      should_activate: true,
      trigger_metric: "rollback_trigger_events",
      threshold: rollbackThreshold,
      observed_value: rollbackCount,
    };
  }

  return { should_activate: false };
}

/**
 * Activate the circuit-breaker by writing to the rollout config file.
 * Persists force-sequential state — operator must explicitly reset.
 *
 * @param {string} configPath - absolute path to production_parallel_rollout.json
 * @param {object} triggerInfo - { trigger_metric, threshold, observed_value }
 * @param {Function} [alertFn] - optional alert callback (destination from config)
 */
export function activateCircuitBreaker(configPath, triggerInfo, alertFn) {
  const config = loadJson(configPath);
  if (!config) throw new Error(`circuit-breaker: cannot load config at ${configPath}`);

  const activationRecord = {
    activated: true,
    activated_at: new Date().toISOString(),
    activation_reason: triggerInfo.trigger_metric,
    trigger_metric: triggerInfo.trigger_metric,
    threshold: triggerInfo.threshold,
    observed_value: triggerInfo.observed_value,
  };

  config.circuit_breaker = { ...config.circuit_breaker, ...activationRecord };
  // force_sequential is set at top level so rollout gate layer 1 catches it
  config.force_sequential = true;
  config.last_policy_change = new Date().toISOString();

  saveJson(configPath, config);

  const alertMsg = [
    `[CIRCUIT-BREAKER ACTIVATED]`,
    `  trigger_metric:  ${triggerInfo.trigger_metric}`,
    `  threshold:       ${triggerInfo.threshold}`,
    `  observed_value:  ${triggerInfo.observed_value}`,
    `  activated_at:    ${activationRecord.activated_at}`,
    `  action:          force_sequential=true persisted`,
    `  recovery:        manual operator reset required`,
  ].join("\n");

  if (typeof alertFn === "function") {
    alertFn(alertMsg);
  } else {
    console.error(alertMsg);
  }

  return activationRecord;
}

/**
 * Reset the circuit-breaker. Operator-only action.
 * Clears force-sequential and activated flag.
 *
 * @param {string} configPath - absolute path to production_parallel_rollout.json
 */
export function resetCircuitBreaker(configPath) {
  const config = loadJson(configPath);
  if (!config) throw new Error(`circuit-breaker reset: cannot load config at ${configPath}`);
  if (!config.circuit_breaker?.activated) {
    return { reset: false, reason: "circuit-breaker was not activated" };
  }

  config.circuit_breaker = {
    ...config.circuit_breaker,
    activated: false,
    activated_at: undefined,
    activation_reason: undefined,
    trigger_metric: undefined,
    threshold: undefined,
    observed_value: undefined,
  };
  // Remove undefined keys
  for (const k of Object.keys(config.circuit_breaker)) {
    if (config.circuit_breaker[k] === undefined) delete config.circuit_breaker[k];
  }
  config.force_sequential = false;
  config.last_policy_change = new Date().toISOString();

  saveJson(configPath, config);
  return { reset: true, reset_at: config.last_policy_change };
}
