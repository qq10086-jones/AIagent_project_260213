/**
 * WS-31-02: Classifier Health Monitor — Unit Tests
 *
 * Verifies that:
 *  1. availability() returns 1.0 when no calls recorded
 *  2. availability() tracks success/failure ratio correctly
 *  3. isUnavailable() returns false when above threshold
 *  4. isUnavailable() returns true when below threshold
 *  5. rolling window evicts oldest entries
 *  6. drillMode forces isUnavailable() = true regardless of outcomes
 *  7. drillMode records all calls as failures
 *  8. alert is raised exactly once when threshold is first breached
 *  9. reset() clears all state
 * 10. snapshot() returns correct structure
 * 11. alert is NOT raised when availability is above threshold
 */

import test from "node:test";
import assert from "node:assert/strict";

import { createClassifierHealthMonitor } from "../src/domain/classifier_health_monitor.js";

test("WS-31-02: availability() returns 1.0 when no calls recorded", () => {
  const mon = createClassifierHealthMonitor();
  assert.equal(mon.availability(), 1.0);
  assert.equal(mon.isUnavailable(), false);
});

test("WS-31-02: availability() tracks success/failure ratio correctly", () => {
  const mon = createClassifierHealthMonitor({ windowSize: 10, alertThreshold: 0.5 });
  // 7 success, 3 failure → 70% availability
  for (let i = 0; i < 7; i++) mon.record(true);
  for (let i = 0; i < 3; i++) mon.record(false);

  assert.equal(mon.availability(), 0.7);
  assert.equal(mon.isUnavailable(), false); // 0.7 >= 0.5 threshold
});

test("WS-31-02: isUnavailable() returns false when above threshold", () => {
  const mon = createClassifierHealthMonitor({ windowSize: 10, alertThreshold: 0.9 });
  for (let i = 0; i < 9; i++) mon.record(true);
  mon.record(true);
  // 100% availability → above 90% threshold
  assert.equal(mon.isUnavailable(), false);
});

test("WS-31-02: isUnavailable() returns true when below threshold", () => {
  const mon = createClassifierHealthMonitor({ windowSize: 10, alertThreshold: 0.9 });
  // 8 success, 2 failures → 80% availability < 90% threshold
  for (let i = 0; i < 8; i++) mon.record(true);
  for (let i = 0; i < 2; i++) mon.record(false);

  assert.equal(mon.isUnavailable(), true);
});

test("WS-31-02: rolling window evicts oldest entries", () => {
  const mon = createClassifierHealthMonitor({ windowSize: 5, alertThreshold: 0.5 });
  // Fill window with failures
  for (let i = 0; i < 5; i++) mon.record(false);
  assert.equal(mon.availability(), 0);

  // Push 5 successes — window should now be 100% success
  for (let i = 0; i < 5; i++) mon.record(true);
  assert.equal(mon.availability(), 1.0);
  assert.equal(mon.isUnavailable(), false);
});

test("WS-31-02: drillMode forces isUnavailable() = true regardless of outcomes", () => {
  const mon = createClassifierHealthMonitor({ drillMode: true });
  // Even if we record successes, drillMode overrides
  for (let i = 0; i < 10; i++) mon.record(true);

  assert.equal(mon.isUnavailable(), true);
});

test("WS-31-02: drillMode records all outcomes as failures", () => {
  const mon = createClassifierHealthMonitor({ windowSize: 10, drillMode: true });
  for (let i = 0; i < 10; i++) mon.record(true); // passed as success, but drill overrides

  const snap = mon.snapshot();
  assert.equal(snap.total_failures, 10);
  assert.equal(snap.availability_rate, 0);
});

test("WS-31-02: alert is raised exactly once when threshold is first breached", () => {
  const alerts = [];
  const mon = createClassifierHealthMonitor({
    windowSize:     10,
    alertThreshold: 0.9,
    alertFn:        (msg) => alerts.push(msg),
  });

  // Cause threshold breach with 2 failures out of 10
  for (let i = 0; i < 8; i++) mon.record(true);
  assert.equal(alerts.length, 0, "no alert before breach");

  mon.record(false); // 9th call success rate = 8/9 = 88.8% < 90%
  assert.equal(alerts.length, 1, "alert raised on breach");

  // Additional failures should NOT trigger another alert
  mon.record(false);
  mon.record(false);
  assert.equal(alerts.length, 1, "alert raised only once");
});

test("WS-31-02: alert is NOT raised when availability stays above threshold", () => {
  const alerts = [];
  const mon = createClassifierHealthMonitor({
    windowSize:     10,
    alertThreshold: 0.9,
    alertFn:        (msg) => alerts.push(msg),
  });

  // 100% success rate — no alert expected
  for (let i = 0; i < 20; i++) mon.record(true);
  assert.equal(alerts.length, 0);
});

test("WS-31-02: reset() clears all state", () => {
  const alerts = [];
  const mon = createClassifierHealthMonitor({
    windowSize:     5,
    alertThreshold: 0.9,
    alertFn:        (msg) => alerts.push(msg),
  });

  // Cause breach and raise alert
  for (let i = 0; i < 5; i++) mon.record(false);
  assert.equal(alerts.length, 1);

  mon.reset();
  const snap = mon.snapshot();
  assert.equal(snap.total_calls,    0);
  assert.equal(snap.total_failures, 0);
  assert.equal(snap.window_calls,   0);
  assert.equal(snap.availability_rate, 1.0);

  // After reset, a new breach should trigger a fresh alert
  for (let i = 0; i < 5; i++) mon.record(false);
  assert.equal(alerts.length, 2, "fresh alert after reset");
});

test("WS-31-02: snapshot() returns correct structure", () => {
  const mon = createClassifierHealthMonitor({ windowSize: 20, alertThreshold: 0.85, drillMode: false });
  for (let i = 0; i < 10; i++) mon.record(i % 2 === 0); // 5 success, 5 failure = 50%

  const snap = mon.snapshot();

  assert.equal(typeof snap.available,         "boolean");
  assert.equal(typeof snap.availability_rate, "number");
  assert.equal(typeof snap.window_size,       "number");
  assert.equal(typeof snap.window_calls,      "number");
  assert.equal(typeof snap.total_calls,       "number");
  assert.equal(typeof snap.total_failures,    "number");
  assert.equal(typeof snap.drill_mode,        "boolean");

  assert.equal(snap.window_size,   20);
  assert.equal(snap.window_calls,  10);
  assert.equal(snap.total_calls,   10);
  assert.equal(snap.total_failures, 5);
  assert.equal(snap.drill_mode,    false);
  assert.equal(snap.available,     false); // 50% < 85% threshold
});
