/**
 * classifier_health_monitor.js  (Layer 3 — Domain)
 *
 * WS-31-02: Classifier Health Monitoring for Limited Dynamic Exposure Trial.
 *
 * Responsibilities:
 *  - Track classifier call outcomes (success / failure / timeout)
 *  - Compute rolling availability rate over a configurable window
 *  - Raise an operational alert when availability drops below threshold
 *  - Expose unavailability drill mode (simulates classifier failure for testing)
 *
 * Design authority: OpenClaw_Nexus_Engineering_Task_List_M7_v2.md § WS-29-01
 * Degradation semantics: Design Document v4.0 § 5.4
 *
 * Integration note:
 *  This service does NOT call the circuit-breaker directly. The trial runner
 *  reads health state and decides whether to invoke circuit_breaker_service.js.
 *  This keeps the two concerns separately testable.
 */

// ── Constants ──────────────────────────────────────────────────────────────────

const DEFAULT_WINDOW      = 50;   // number of recent calls tracked
const DEFAULT_ALERT_BELOW = 0.90; // alert when availability < 90%

// ── Factory ───────────────────────────────────────────────────────────────────

/**
 * @param {{
 *   windowSize?:       number,
 *   alertThreshold?:   number,
 *   alertFn?:          (msg: string) => void,
 *   drillMode?:        boolean,
 * }} opts
 */
export function createClassifierHealthMonitor({
  windowSize     = DEFAULT_WINDOW,
  alertThreshold = DEFAULT_ALERT_BELOW,
  alertFn        = null,
  drillMode      = false,
} = {}) {
  /**
   * Circular buffer of recent call outcomes.
   * true  = classifier returned a valid result (success)
   * false = classifier failed or timed out
   */
  const outcomes = [];

  let totalCalls    = 0;
  let totalFailures = 0;
  let alertRaised   = false;

  // ── Core recording ─────────────────────────────────────────────────────────

  /**
   * Record a classifier call outcome.
   * In drillMode, every call is recorded as a failure regardless of actual result.
   *
   * @param {boolean} success - true if classifier returned a valid response
   */
  function record(success) {
    const effective = drillMode ? false : Boolean(success);
    outcomes.push(effective);
    if (outcomes.length > windowSize) outcomes.shift();
    totalCalls++;
    if (!effective) totalFailures++;
    _maybeAlert();
  }

  /**
   * Compute availability over the rolling window.
   * Returns 1.0 (100%) if no calls have been recorded yet.
   *
   * @returns {number} fraction in [0, 1]
   */
  function availability() {
    if (outcomes.length === 0) return 1.0;
    const successes = outcomes.filter(Boolean).length;
    return successes / outcomes.length;
  }

  /**
   * Returns true when the classifier should be treated as "unavailable"
   * according to design document v4.0 § 5.4.
   *
   * Conditions:
   *  - drillMode is active, OR
   *  - rolling availability is below the alert threshold
   */
  function isUnavailable() {
    return drillMode || availability() < alertThreshold;
  }

  /**
   * Current health snapshot.
   * @returns {{ available: boolean, availability_rate: number, window_size: number, window_calls: number, total_calls: number, total_failures: number, drill_mode: boolean }}
   */
  function snapshot() {
    return {
      available:         !isUnavailable(),
      availability_rate: availability(),
      window_size:       windowSize,
      window_calls:      outcomes.length,
      total_calls:       totalCalls,
      total_failures:    totalFailures,
      drill_mode:        drillMode,
    };
  }

  /**
   * Reset recorded outcomes. Used between trial phases.
   */
  function reset() {
    outcomes.length = 0;
    totalCalls      = 0;
    totalFailures   = 0;
    alertRaised     = false;
  }

  // ── Internal ────────────────────────────────────────────────────────────────

  function _maybeAlert() {
    if (alertRaised) return;
    if (isUnavailable()) {
      alertRaised = true;
      const msg = [
        `[CLASSIFIER HEALTH ALERT]`,
        `  availability:  ${(availability() * 100).toFixed(1)}% (threshold: ${(alertThreshold * 100).toFixed(1)}%)`,
        `  window_calls:  ${outcomes.length}/${windowSize}`,
        `  total_calls:   ${totalCalls}`,
        `  total_failures:${totalFailures}`,
        `  drill_mode:    ${drillMode}`,
        `  action:        routing will fall back to static-policy-only`,
      ].join("\n");

      if (typeof alertFn === "function") {
        alertFn(msg);
      } else {
        console.error(msg);
      }
    }
  }

  return { record, availability, isUnavailable, snapshot, reset };
}
