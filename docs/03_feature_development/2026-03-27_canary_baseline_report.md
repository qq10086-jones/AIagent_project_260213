# Canary Baseline Report

- Date: 2026-03-27
- Type: Historical Backfill Baseline
- Source: existing release artifacts and canary outputs already present in the repository

## Summary

This baseline is a backfill measurement created after v2 warning-only instrumentation was added. It does not represent a fresh live rerun set. Instead, it re-evaluates existing release artifacts against the v2 rubric to estimate the current false-positive rate.

Key result:

- sampled runs: 5
- GO verdicts: 5
- non-demo-usable under v2 rubric: 5
- estimated false-positive rate: 100%
- preview issue rate: 60%

## Interpretation

The current system is still optimized for artifact completeness and pipeline success, not for demo-usable product quality. Even historical happy-path canaries remain GO under legacy gates while failing the v2 bar because preview proof is missing or the product remains scaffold-level.

## Sample Set

- artifacts/release/93ec9528-0dd7-4e40-b5d1-3d9d649bf6c3
- artifacts/release/0167df14-b591-486f-a472-5b422e5a9c68
- artifacts/release/065c4cd5-789e-42a2-986f-c1e2f2fd1221
- orchestrator/artifacts/canary/coding_team_e2e/happy_path/artifacts/release/e2e-happy-run
- orchestrator/artifacts/canary/m4_compat/happy_path/artifacts/release/m4-compat-run

## Caveat

This report should be replaced by a fresh live Discord canary baseline once 3-5 representative prompts are rerun under the new warning-only instrumentation.
