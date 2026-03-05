# Coding Team Fast-Track UI Strategy (r1.1, 2026-03-05)

> Purpose: launch a usable `coding_team_v0` quickly (one-command project delivery) by reusing a proven open-source UI baseline, while keeping current OpenClaw/Nexus control-plane governance.

## Changelog (r1.1)
- Added "current state vs target state" timeout policy to avoid doc/runtime mismatch.
- Added explicit launch thresholds (`n=5` internal, `n=20` production).
- Promoted UI template pinning from "next action" to mandatory launch gate.
- Normalized wording and encoding to plain UTF-8 text.

---

## 1. Goal
- Launch a usable `coding_team_v0` quickly for one-command project delivery.
- Allow long build/runtime per project run (minutes are acceptable).
- Optimize for time-to-launch and visual quality, not hand-crafted UI generation.

## 2. Decision
- Keep current OpenClaw/Nexus control plane and workflow gates.
- Use a GitHub open-source UI baseline for CRM/webapp projects.
- Run in a continuous loop:
  - submit command
  - run workflow
  - inspect failures
  - patch and re-run

This is a launch strategy, not a research strategy.

## 3. Why This Route
- Pure LLM-generated UI is unstable and visually inconsistent.
- Existing open-source admin/CRM UI templates provide:
  - mature layout and components
  - responsive behavior
  - lower regression risk
  - faster FE/BE integration speed

## 4. Scope Boundaries
### In scope
- One-command coding-team runs for CRM/game mini baselines.
- Open-source UI template adoption for CRM frontend.
- Strict artifact gate + Go/No-Go enforcement retained.

### Out of scope
- Custom visual design from scratch.
- New framework migration across the whole platform.
- Large dashboard redesign unrelated to coding-team release.

## 5. System Integration Plan
- Keep workflow:
  - `pm_spec -> arch_design -> impl_fe -> impl_be -> qa_verify -> release_pack`
- For `impl_fe`:
  - FE role works on top of the chosen template repository/subtree.
  - only business modules are generated/modified (list/detail/form/auth flows).
- For `qa_verify`:
  - verify template still builds and key pages render.
  - validate acceptance mapping and evidence in `qa/verification.json`.

## 6. UI Baseline Strategy
### 6.1 Mandatory Gate
Template must be pinned before launch verdict is allowed.

Required fields:
- Repo URL
- License
- Commit hash or release tag
- Build tool (Next/Vite/etc.)
- UI library
- Auth baseline
- Data-layer baseline
- Known constraints

If any required field is missing, mark run as `NO_GO_TEMPLATE_UNPINNED` (operational policy gate).

### 6.2 Candidate Shortlist
- `satnaing/shadcn-admin` (template-style)
- `arhamkhnz/next-shadcn-admin-dashboard` (template-style)
- `marmelab/react-admin` (framework-style, very mature CRUD stack)

### 6.3 Locking Policy
- v0 phase must pin to commit hash or release tag (never "latest").
- Dependency upgrades must run in an upgrade branch and pass canary thresholds before merge.
- Keep a `TEMPLATE_NOTES.md` with repo, commit, license, local patches.

## 7. Engineering Rules
- Never rewrite the template design system in v0.
- Preserve template layout, tokens, and component contracts.
- Only add project-specific modules/routes.
- Keep patch size reviewable and traceable via release pack.

## 8. Operational Mode (Run-and-Fix)
### 8.1 Canary as Deployment Gate
- command:
  - `canary:coding_team --n <N> --strict true --input <fixture>`
- required reporting:
  - step status
  - failure code
  - current blocked step/task on timeout

### 8.2 Timeout Policy: Current vs Target
Current implemented behavior (as of 2026-03-05):
- task watchdog timeout is hard-fail based (`TASK_TIMEOUT`) using runtime config.
- there is no `needs-human` pause state with `/continue` in production path yet.

Target behavior (planned):
- default run budget `30m`.
- at `T=30:00` emit soft-timeout reminder and request operator decision:
  - continue (`+15m`)
  - abort (`timeout_abort`)
  - escalate (`needs_fix`)
- hard fail after extension cap or deterministic fatal errors.

Bridge policy (until target behavior is implemented):
- keep current hard-timeout runtime values.
- treat timeout diagnostics (`step_id/task_id`) as mandatory evidence in failure triage.

## 9. Release Readiness Definition
### 9.1 Internal Launch (fast-track usable)
- CRM `n=5` strict canary consecutive green achieved.
- Game `n=5` strict canary consecutive green achieved.
- Template pin gate completed.

### 9.2 Production Launch
- CRM and Game each pass `n=20` strict canary consecutive green.
- Go/No-Go remains `GO` across final batch.
- No unresolved high-severity failure family in last 24h.

## 10. Current Status (2026-03-05)
- Fast-track execution path is online.
- Single strict run evidence:
  - CRM: `GO`
  - Game: `GO`
- Remaining:
  - batch stability (`n=5` then `n=20`)
  - template pin gate completion

## 11. Immediate Execution Plan
1. Pick and pin one UI baseline (fill template record).
2. Run baseline canary on pinned template (no business customization).
3. Implement first CRM module set.
4. Run CRM/Game strict canary `n=5`.
5. Promote to `n=20` only after `n=5` stable.
