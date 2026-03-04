# Next Sprint Plan (2 weeks) — Coding Team v0: Strict-Mode Stability

**Sprint window:** 2026-03-03 → 2026-03-17 (Asia/Tokyo)  
**North Star:** `coding_team_v0` strict mode runs **20 consecutive green canaries** with **0 missing** and **0 invalid** artifacts.

---

## Success Metrics (Sprint Exit Criteria)

1. **Canary reliability**
   - `canary:coding_team` passes **20 consecutive runs** in strict mode.

2. **Artifact integrity**
   - Missing artifact ⇒ `ARTIFACT_MISSING`
   - Malformed/empty artifact ⇒ `ARTIFACT_INVALID`

3. **QA step determinism**
   - `qa_verify` always produces:
     - `tests/test_plan.md`
     - `qa/smoke_report.md`
     - `qa/verification.json`
   - `qa/verification.json` maps acceptance items to pass/fail + evidence.

4. **Reporting**
   - Each run produces a release pack + strict canary report with GO/NO_GO.

---

## Scope

### In-scope (must ship)
- Strict-mode stabilization for `coding_team_v0`
- Artifact **quality** validation (not only presence)
- Deterministic QA artifacts
- Canary harness: one command → one report

### Out-of-scope (defer)
- New project types (ecom/video) beyond skeletons
- Major orchestration refactors / new OSS frameworks
- Large dashboard expansions

---

## Workstreams & Backlog

### WS1 — Artifact Gate: presence → quality (P0)
**Deliverables**
- Contracts for required artifacts (paths + type + schema/min-content rules)
- Validator supports:
  - JSON schema validation
  - Markdown minimum-content validation
  - Summary in canary/release report

**Tasks**
- [P0] Define contracts for all required artifacts in `coding_team_v0`
  - **DoD:** contracts checked-in + versioned
- [P0] Add JSON schema validation for:
  - `acceptance/acceptance.json`
  - `qa/verification.json`
  - `risk/risk_report.json`
  - **DoD:** schema fail ⇒ `ARTIFACT_INVALID` with `path` + `reason`
- [P0] Add MD minimum-content validation for:
  - `tests/test_plan.md`, `qa/smoke_report.md`
  - **DoD:** empty/placeholder fails ⇒ `ARTIFACT_INVALID`
- [P1] Add “artifact quality summary” to reports
  - **DoD:** required/present/invalid counts + top failures

---

### WS2 — QA Verify: deterministic artifact writing (P0)
**Deliverables**
- `qa_verify` always writes the 3 QA artifacts at canonical paths
- Acceptance→verification mapping + evidence

**Tasks**
- [P0] Lock `expected_artifacts` for QA step in registry/workflow
  - **DoD:** strict mode expects canonical paths only
- [P0] Template-based QA generation (+ fallback)
  - Add templates:
    - `templates/test_plan.md.tmpl`
    - `templates/smoke_report.md.tmpl`
    - `templates/verification.json.tmpl`
  - **DoD:** even without LLM, artifacts are valid & non-empty
- [P1] Enforce acceptance-to-verification mapping
  - **DoD:** `verification.json` references acceptance IDs; mismatch fails validation
- [P1] Evidence requirement
  - **DoD:** each acceptance item has `evidence[]` (log excerpt, diff, command output)

---

### WS3 — Canary Harness: one command, one report (P0)
**Deliverables**
- Fixed canary inputs
- CLI runner: run N times, aggregate, emit reports

**Tasks**
- [P0] Freeze 2 canary inputs:
  - `canary_inputs/crm_mini.json`
  - `canary_inputs/game_mini.json` (or another stable case)
  - **DoD:** deterministic, no external dependencies
- [P0] Add CLI: `npm run canary:coding_team -- --n 20 --strict true --input crm_mini.json`
  - **DoD:** outputs `canary_report.md` + `canary_report.json`
- [P1] Flake controls (seed/order/timeouts)
  - **DoD:** failures are reproducible and classified

---

### WS4 — Governance: strict failures are operable (P0/P1)
**Deliverables**
- Standard failure payload
- DLQ policy for strict failures

**Tasks**
- [P0] Standardize failure payload:
  - `error_code`, `failed_step`, `missing[]`, `invalid[]`, `suggested_fix`
  - **DoD:** appears in event log + reports
- [P1] DLQ routing for `ARTIFACT_MISSING/INVALID`
  - **DoD:** easy query + clear label + re-run guidance

---

## Milestones

- **M1 (Day 3–4):** QA step always produces valid artifacts
- **M2 (Day 7):** Validator blocks empty/malformed artifacts (`ARTIFACT_INVALID`)
- **M3 (Day 10):** Canary runner produces report for `--n 20`
- **M4 (Sprint end):** 20 consecutive green canaries (exit criteria met)

---

## Global Definition of Done

- Code merged + minimal docs updated
- Canary run includes the change
- Failures are deterministic and classified (missing vs invalid)
- Report shows changes clearly (before/after where relevant)

---

## Sprint Rituals

- Daily: run `canary:coding_team --n 5` and paste summary into progress log
- Mid-sprint (Day 7): verify M2, cut scope if needed
- End-sprint: run `--n 20` for both canary inputs
