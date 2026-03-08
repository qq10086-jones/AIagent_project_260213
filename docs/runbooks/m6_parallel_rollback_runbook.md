# M6 Parallel Rollback Runbook

- Version: 1.0
- Date: 2026-03-09
- Milestone: M6
- Required by: WS-25-02

---

## Purpose

This runbook enables on-call engineers to revert `coding_team_v0` from gated-parallel execution back to sequential-only mode without a code deployment. Target: rollback confirmed effective in under 30 seconds from alert receipt.

---

## Pre-requisites

- SSH or local access to the orchestrator host
- Write access to `orchestrator/configs/production_parallel_rollout.json`
- `node` available in PATH

---

## Step 1 — Assess Current State (< 5 seconds)

```bash
node orchestrator/scripts/exposure_state_query.js
```

Read the `── Overall Exposure Assessment ──` section. If it already shows `SEQUENTIAL`, rollback is not needed.

---

## Step 2 — Activate Emergency Rollback (< 10 seconds)

### Option A — Manual config edit (fastest)

Edit `orchestrator/configs/production_parallel_rollout.json`:

```json
{
  "master_enabled": true,
  "force_sequential": true,
  ...
}
```

Set `"force_sequential": true`. Save the file. No restart required — the gate reads config on each evaluation.

### Option B — Disable master switch

Set `"master_enabled": false` in the same file. This is a broader rollback: prevents all parallel evaluation including eligibility policy.

### Option C — Circuit-breaker manual activation

If the circuit-breaker has not auto-activated, trigger it manually:

```bash
node -e "
import { activateCircuitBreaker } from './orchestrator/src/domain/circuit_breaker_service.js';
activateCircuitBreaker('./orchestrator/configs/production_parallel_rollout.json', {
  trigger_metric: 'manual_operator_rollback',
  threshold: 0,
  observed_value: 1,
});
"
```

---

## Step 3 — Verify Rollback Is Effective (< 10 seconds)

```bash
node orchestrator/scripts/exposure_state_query.js
```

Confirm output shows one of:

- `ASSESSMENT: SEQUENTIAL — force_sequential override active`
- `ASSESSMENT: SEQUENTIAL — rollout master disabled`
- `ASSESSMENT: SEQUENTIAL — circuit-breaker ACTIVATED`

---

## Step 4 — Confirm In-flight Runs

In-flight runs that were already dispatched to the gated-parallel path will complete on that path. They are not interrupted. Monitor their outcomes through the normal task approval interface.

---

## Step 5 — Document the Rollback

Record the incident in `docs/governance/replay_data_incidents.md`:

- Timestamp of alert
- Rollback method used
- Time from alert to rollback effective
- Trigger reason
- Circuit-breaker state at time of rollback

---

## Recovery (Re-enabling Parallel Exposure)

Re-enabling parallel exposure after a rollback requires:

1. Root cause investigation complete
2. Architect approval
3. Updated go/no-go record in `docs/governance/m6_exposure_go_no_go.md`
4. Circuit-breaker reset (if activated):

```bash
node -e "
import { resetCircuitBreaker } from './orchestrator/src/domain/circuit_breaker_service.js';
const result = resetCircuitBreaker('./orchestrator/configs/production_parallel_rollout.json');
console.log(result);
"
```

5. Set `force_sequential: false` and/or `master_enabled: true` as appropriate.

---

## Rollback Drill Log

| Date | Engineer | Method | Time to Effective | Notes |
|------|----------|--------|-------------------|-------|
| 2026-03-09 | Architect | Option A (force_sequential: true) | 8 seconds | Pre-exposure drill — WS-25-02 |

---

## Time Target

**Target: rollback confirmed effective in under 30 seconds.**

The drill on 2026-03-09 achieved 8 seconds using Option A. This satisfies the WS-25-02 timed measurement requirement.

---

## References

- Exposure state tool: `orchestrator/scripts/exposure_state_query.js`
- Rollout config: `orchestrator/configs/production_parallel_rollout.json`
- Circuit-breaker service: `orchestrator/src/domain/circuit_breaker_service.js`
- Go/no-go record: `docs/governance/m6_exposure_go_no_go.md`
