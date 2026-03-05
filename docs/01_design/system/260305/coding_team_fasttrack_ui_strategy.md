# Coding Team Fast-Track UI Strategy (2026-03-05)

## 1. Goal
- Launch a usable `coding_team_v0` quickly for one-command project delivery.
- Allow long build runtime per project run (minutes are acceptable).
- Optimize for **time-to-launch and visual quality**, not hand-crafted UI generation.

## 2. Decision
- Keep current OpenClaw/Nexus control plane and workflow gates.
- Use a **GitHub open-source UI baseline** for CRM/webapp projects.
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
  - faster implementation speed for FE/BE integration

## 4. Scope Boundaries
### In scope
- One-command coding-team run for CRM/game mini baselines.
- Open-source UI template adoption for CRM frontend.
- Strict artifact gate + Go/No-Go enforcement retained.

### Out of scope
- Custom visual design from scratch.
- New framework migration across the whole platform.
- Large dashboard redesign unrelated to coding-team release.

## 5. System Integration Plan
- Keep workflow: `pm_spec -> arch_design -> impl_fe -> impl_be -> qa_verify -> release_pack`.
- For `impl_fe`:
  - frontend role works on top of the chosen template repository/subtree.
  - only business modules are generated/modified (list/detail/form/auth flows).
- For `qa_verify`:
  - verify template still builds and key pages render.
  - check acceptance mapping and evidence into `qa/verification.json`.

## 6. Template Adoption Rules
- Selection criteria:
  - active maintenance
  - permissive license
  - React/TypeScript preference
  - clean admin/CRM information architecture
- Engineering rules:
  - never rewrite template design system in v0
  - preserve template layout, token, and component contracts
  - only add project-specific pages/modules/routes

## 7. Operational Mode (Run-and-Fix)
- Use canary harness as the deployment gate:
  - `canary:coding_team --n <N> --strict true --input <fixture>`
- Required reporting:
  - step status
  - failure code
  - current blocked step/task on timeout
- Any failure is patched before the next batch run.

## 8. Release Readiness Definition
- Functional readiness:
  - one-command CRM flow succeeds end-to-end.
- Quality readiness:
  - strict artifact checks pass
  - Go/No-Go is `GO`
- Stability readiness:
  - consecutive green canaries reaches release threshold.

## 9. Current Status (2026-03-05)
- Fast-track execution path is online.
- CRM mini strict run: `GO`.
- Game mini strict run: `GO`.
- Remaining work: scale from single green runs to consecutive green batches.

## 10. Immediate Next Actions
1. Lock one open-source UI baseline for CRM.
2. Implement first CRM module set on that baseline.
3. Run `n=5` strict canary for CRM + game.
4. Expand to larger consecutive-green target after `n=5` stabilizes.
