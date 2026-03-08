# OpenClaw Nexus M6 — Approval Note

- Version: 1.0
- Date: 2026-03-09
- Milestone: M6 — Production Parallel Rollout Readiness + Staging Validation
- Prepared by: Architect (project owner)
- Status: APPROVED FOR STAY_GATED

---

## 1. Scope

M6 is a rollout-governance milestone. Its purpose is to decide whether production parallel execution of `coding_team_v0` can be safely opened for a limited approved subset of workflows.

This approval note covers the M6 implementation scope: infrastructure, contracts, gate wiring, baseline metrics, and governance artifacts. It does not authorize production parallel execution — the go/no-go decision is separately recorded in `docs/governance/m6_exposure_go_no_go.md`.

---

## 2. Replay Corpus Coverage and Distribution Summary

Source: `orchestrator/replay/manifests/m6_staging_replay_manifest.json`

| Dimension | Required | Actual | Status |
|-----------|----------|--------|--------|
| Total cases | ≥ 50 | 50 | PASS |
| PM-heavy | ≥ 7 | 7 | PASS |
| Architect-heavy | ≥ 7 | 7 | PASS |
| BE-led | ≥ 7 | 10 | PASS |
| FE-led | ≥ 7 | 10 | PASS |
| QA-heavy | ≥ 7 | 7 | PASS |
| Mixed ambiguous | ≥ 7 | 9 | PASS |
| FE-safe cases | ≥ 10 | 11 | PASS |
| Non-FE-safe cases | ≥ 10 | 39 | PASS |
| Dirty-repo cases | ≥ 5 | 5 | PASS |

---

## 3. Replay Data Governance and Sanitization Boundary

Source: `docs/governance/replay_data_governance_m6.md`, `orchestrator/replay/README.md`

- Raw Discord prompts: NOT retained (all `raw_prompt_ref` fields are null)
- Sanitization applied: user IDs, channel IDs, links, attachments, secrets, sensitive paths, timestamps
- Review process: two-person review required before fixture commit
- Fixture generation rights: Architect-level only
- Staging run artifacts: 90-day retention under `orchestrator/artifacts/m6_staging_replay/`
- Approval packages: retained indefinitely

---

## 4. Baseline Metrics with Measured Values Against Pre-defined Thresholds

Source: `metrics/` directory. All values from simulation baseline; re-measurement from live LLM execution is required before GO_LIMITED_EXPOSURE.

### 4.1 Context Budget Baseline (WS-26-01)

Source: `metrics/baseline_context_budget.json`

| Role | p50 | p90 | Max | Overflow Risk |
|------|-----|-----|-----|---------------|
| PM | 60% | 76% | 79% | LOW |
| Architect | 58% | 76% | 79% | LOW |
| BE | 56% | 74% | 79% | LOW |
| FE | 57% | 74% | 79% | LOW |
| QA | 57% | 74% | 79% | LOW |

No high-risk overflow tails detected in simulation baseline. Values expected to increase under real LLM execution.

### 4.2 Diff-first Reliability (WS-26-02)

Source: `metrics/diff_first_baseline.json`, `metrics/patch_reliability.json`

| Metric | Go/No-Go Threshold | Measured (Simulation) | Status |
|--------|-------------------|----------------------|--------|
| Diff-first hit rate (clean-repo) | ≥ 60% | 100% | PASS (simulation) |
| Patch anchor mismatch rate | ≤ 15% | 0.0% | PASS (simulation) |

Note: Simulation produces ideal diff-first outcomes. Real execution will show lower hit rates, particularly for dirty-repo cases.

### 4.3 Parallel vs Sequential Comparison (WS-23-03)

Source: `metrics/parallel_vs_sequential.json`

| Metric | Go/No-Go Threshold | Measured (Simulation) | Status |
|--------|-------------------|----------------------|--------|
| Success rate delta | ≤ 5% | 0.0% | PASS (simulation) |
| Partial failure rate (parallel) | ≤ 10% | 0.0% | PASS (simulation) |

Note: All parallel runs defaulted to sequential (master_enabled=false). Delta is 0 by construction.

### 4.4 Parallel Eligibility (WS-26-03)

Source: `metrics/parallel_eligibility.json`

| Metric | Go/No-Go Threshold | Status |
|--------|-------------------|--------|
| FE-safe eligibility qualification | ≥ 1 workflow type at ≥ 80% | DEFERRED — requires live run with master enabled |

Policy declares `fe_led` as FE-safe. Structural guard confirms eligibility. Live distribution measurement deferred.

---

## 5. Exposure Policy

Source: `orchestrator/configs/parallel_exposure_policy.json`

- Allowed workflow types: `coding_team_v0`
- Allowed project types: `crm`
- FE-safe eligible input classes: `fe_led`
- Deny conditions: 7 active, including `structural_completion_impossible` with `override_allow: true`
- Runtime configurable: YES — no service restart required

---

## 6. Rollback Readiness

Source: `docs/runbooks/m6_parallel_rollback_runbook.md`

- Emergency rollback method: set `force_sequential: true` in `production_parallel_rollout.json`
- No code deployment required: YES
- Rollback drill completed: 2026-03-09
- Drill time: **8 seconds** (target: < 30 seconds) — PASS
- Exposure state diagnostic tool: `orchestrator/scripts/exposure_state_query.js`

---

## 7. Circuit-Breaker Configuration and Test Evidence

Source: `orchestrator/configs/production_parallel_rollout.json`, `orchestrator/src/domain/circuit_breaker_service.js`

| Parameter | Value |
|-----------|-------|
| Rolling window size | 100 runs |
| Partial failure rate threshold | 25% |
| Rollback trigger event threshold | 3 events |
| Alert destination | ops-alerts |
| Auto-recovery | Disabled (manual reset required) |

Test evidence: `orchestrator/artifacts/canary/m6_rollout_gate/canary_m6_rollout_gate.json` — 10/10 PASS including circuit-breaker activation, no-auto-recovery, and operator reset paths.

---

## 8. Open Risks

| Risk ID | Risk | Status |
|---------|------|--------|
| R-9 | Production exposure mismatch (dispatch allowed, completion impossible) | MITIGATED — structural guard (WS-24.5-02) + negative test (WS-24-04) |
| R-10 | Replay data leakage | MITIGATED — governance + sanitization rules enforced |
| R-11 | Rollback slow in practice | MITIGATED — drill completed in 8 seconds |
| R-12 | Eligibility policy too permissive or too conservative | OPEN — requires live run data for tuning |
| R-13 | No automated safety net | MITIGATED — circuit-breaker implemented (WS-25-05) |

---

## 9. Go/No-Go Decision Reference

See `docs/governance/m6_exposure_go_no_go.md`.

Current decision: **STAY_GATED**

Re-evaluation trigger: live LLM staging run producing metrics from actual workflow execution.

---

## 10. Test Suite Status

- Full orchestrator test suite: **84/84 PASS**
- WS-24-04 integration tests (including negative misconfiguration test): **15/15 PASS**
- WS-23-04 staging canary: **7/7 PASS**
- WS-25-04 rollout gate canary: **10/10 PASS**

---

## 11. Approval

- Approved by: Architect (project owner)
- Date: 2026-03-09
- Scope: M6 infrastructure, contracts, gate wiring, and governance artifacts
- Not approved: production parallel execution (requires GO_LIMITED_EXPOSURE decision)
