# Runtime Startup Path Note

- Date: 2026-03-11
- Scope: authoritative startup-path guidance after WS-NEXT-02 hardening

---

## 1. Authoritative Path

Governance and rollout runtime files are authoritative under repo root:

- `configs/production_parallel_rollout.json`
- `configs/parallel_exposure_policy.json`
- `configs/m7_exposure_cohorts.json`
- `configs/runtime/runtime_defaults.json`

Orchestrator-local config remains authoritative for orchestrator-owned files:

- `orchestrator/configs/llm_providers.json`
- `orchestrator/configs/llm_role_policy.json`
- `orchestrator/configs/context_budget_policy.json`

---

## 2. Startup Rule

For local manual startup:

- start from `orchestrator/`
- if `WORKSPACE_ROOT` is unset, runtime now resolves workspace root to repo root automatically

For docker-compose startup:

- governance files are mounted from repo root into `/app/configs`
- workspace root remains `/workspace`

---

## 3. Verification Command

Run:

```powershell
npm.cmd --prefix orchestrator run validate:runtime_boot_sources
```

Output artifact:

- `orchestrator/artifacts/validation/runtime_boot_sources/runtime_boot_sources_report.json`

---

## 4. Intent

This change removes silent drift between:

- local manual startup path
- workflow runtime governance path
- docker-compose mounted governance files
