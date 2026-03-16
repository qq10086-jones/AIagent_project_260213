# OpenClaw Nexus: M12 Discord UX & Live Preview (v2.2)

**Date:** 2026-03-16
**Author:** PM / Architecture Team
**Status:** APPROVED FOR IMPLEMENTATION

---

## 1. Executive Summary

### 1.1 The Problem
Currently, the Nexus `coding_team_v0` workflow successfully generates, tests, and packages code, delivering a Release Pack to local storage and MinIO. However, from an end-user perspective in Discord, the experience still suffers from two UX gaps:
1. **Black Box Anxiety:** Workflows take 3-5 minutes and users need visible progress while the DAG is running.
2. **Anti-climactic Delivery:** Users receive files and notes instead of a running product they can click and inspect immediately.

### 1.2 The Goal
M12 upgrades Nexus from "code generator" to "interactive product generator":
1. **Dynamic Progress Tracking:** Provide a real-time, updating progress bar in Discord that reflects the exact workflow step currently being executed.
2. **Live Preview:** Launch an ephemeral preview runtime and return a clickable preview URL when the app is eligible.

---

## 2. Architecture & Design

### 2.1 Dynamic Progress Tracking
Discord does not support native animated widgets, but we can achieve a live progress bar using the `Message.edit()` API in `discord.js`.

* The orchestrator listens to workflow transition events such as `step.started`, `step.completed`, and `workflow.failed`.
* A snapshot state keyed by `run_id` drives rendering rather than replaying the full event history.
* Message edits remain debounced for rate-limit safety, but terminal workflow events bypass debounce for immediate final rendering.

### 2.2 The New Workflow Step: `deploy_preview`
We extend the existing `coding_team_v0` DAG with a new terminal step named `deploy_preview` after `release_pack`.

**Updated DAG Flow:**
`pm_spec` -> `arch_design` -> `impl_be` / `impl_fe` -> `qa_verify` -> `release_pack` -> `deploy_preview`

### 2.3 Deployment Strategy: Local First
**Design Decision:** M12 Phase B will use a **Local Ephemeral Preview Runtime** as the primary path. Nexus will detect how to boot the generated app, allocate a safe localhost port, launch the runtime under managed TTL, and return the local preview URL.

Why local first:
* It validates the full preview workflow without requiring paid cloud infrastructure.
* It removes the current dependency on pre-created Render services.
* It lets us prove the operator UX before expanding to public HTTPS hosting.

Scope limits:
* The first implementation targets `http://localhost:<port>` URLs for local alpha users.
* Public internet preview remains a deferred enhancement.
* Preview apps must not connect to the core Nexus production DB.

### 2.4 Preview Eligibility Rules
The preview launcher only supports apps that can boot without external stateful infrastructure.

Eligible examples:
* Static HTML/CSS/JS sites
* Node apps with a simple local `start` contract
* Python apps with a simple local entrypoint

Ineligible examples:
* Apps requiring PostgreSQL, MySQL, MongoDB, Redis, or similar external stateful services
* Apps that need secret provisioning not available in the preview sandbox
* Apps whose startup contract cannot be determined deterministically

Fallback rule:
* If the app is ineligible or boot detection fails, Nexus returns the Release Pack instead of a broken preview URL.

---

## 3. Engineering Task List (M12 Revised)

### WS-39: Real-time Discord Progress Bar
| Task ID | Description | Acceptance Criteria |
| :--- | :--- | :--- |
| **WS-39-01** | **State Listener & Map** | Implement a listener for workflow step events in `orchestrator` and map them to the original Discord `message_id`. |
| **WS-39-02** | **Message Edit Engine** | Implement a debounced updater that renders a current-state snapshot rather than replaying event history. |

### WS-40: Local Preview Infrastructure & Tooling
| Task ID | Description | Acceptance Criteria |
| :--- | :--- | :--- |
| **WS-40-01** | **Local Preview Launcher** | Implement a local preview launcher that detects a runnable static / Node / Python app, allocates a safe localhost port, and starts it under managed TTL. |
| **WS-40-02** | **Deploy Agent Tool Creation** | Create or update `ops.deploy_preview` so it wraps the local preview launcher and emits structured preview metadata. |
| **WS-40-03** | **Static Dependency Scanner** | Scan `package.json`, `requirements.txt`, and `pyproject.toml` for blocked DB dependencies before preview launch. |
| **WS-40-04** | **Preview Process Registry** | Persist preview runtime metadata such as pid or container id, port, `run_id`, and `expires_at` so restart cleanup is possible. |

### WS-41: Workflow Engine Extension
| Task ID | Description | Acceptance Criteria |
| :--- | :--- | :--- |
| **WS-41-01** | **Update coding_team_v0** | Append the `deploy_preview` step and ensure result resolution prefers preview URLs when present. |
| **WS-41-02** | **Entrypoint Manifest Injection** | Update architecture prompts to force AI outputs to include a deterministic local boot contract such as static root, `npm start`, or Python entrypoint. |

### WS-42: Final UX & Cleanup
| Task ID | Description | Acceptance Criteria |
| :--- | :--- | :--- |
| **WS-42-01** | **Final Discord Render** | Upon completion, the progress bar message is edited one final time to display a highly visible local preview link such as `http://localhost:...` when available. |
| **WS-42-02** | **Preview Fallback** | If preview launch fails, catch the error gracefully and return the Release Pack URL or notes as a fallback. |
| **WS-42-03** | **Preview Cleanup** | Implement TTL-based local cleanup so stale preview processes do not accumulate after crashes or restarts. |
| **WS-42-04** | **Cloud Upgrade Path** | Document Render or equivalent public preview hosting as a deferred enhancement, explicitly out of scope for the first local-alpha implementation. |

---

## 4. Rollout Plan

1. **Phase A (UX First):** Land the Discord progress bar and state snapshot rendering.
2. **Phase B (Local Preview Integration):** Implement local preview boot detection, launch, and cleanup.
3. **Phase C (Discord Alpha):** Return localhost preview links for internal operators and validate the full user path.
4. **Phase D (Cloud Upgrade, Deferred):** Add Render or equivalent public preview after the local preview path is stable.

---

## 5. Immediate Task Order

1. Implement boot detection for static / Node / Python preview targets.
2. Persist preview runtime state for restart-safe cleanup.
3. Update Discord final result rendering to prefer preview URLs.
4. Add watchdog cleanup for expired local preview runtimes.
5. Defer Render-specific public preview work until local-alpha validation is complete.

---

## 6. Explicit Non-Goals for the First Local Preview Release

* Public HTTPS URLs for external Discord users
* Automatic provisioning of cloud services
* DB-backed app previews with production-like persistence
* Multi-tenant hardened preview isolation beyond local operator usage

This keeps M12 Phase B focused on proving end-to-end preview behavior before spending effort on cloud deployment complexity.
