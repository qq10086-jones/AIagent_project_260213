# Brain Router Classification Taxonomy v1
## Date: 2026-03-09
## Workstream: WS-28-01
## Author: PM / Architect AI

### 1. Purpose
This document defines the v1 taxonomy for the Brain Router classification engine. It acts as the foundational rule set for how incoming tasks are evaluated and mapped into execution paths and model tiers.

### 2. Layer A: Work Shape

The Work Shape describes the structural complexity and risk profile of the request.

#### 2.1 `single_branch_safe`
- **Definition:** The task is simple, localized, and can be completely handled by a single implementation branch (e.g., just frontend or just backend) without affecting shared contracts.
- **Positive Examples:**
  - "Change the button color on the login page to blue."
  - "Add a new index to the users table."
- **Negative Examples:**
  - "Add a new API endpoint and use it in the dashboard." (Requires both BE and FE).
- **Execution Path:** `sequential`

#### 2.2 `dual_branch_parallel_candidate`
- **Definition:** The task requires both backend and frontend changes, but the interface contract is deterministic and does not pose a high risk to release.
- **Positive Examples:**
  - "Add a new optional filter to the users list API and update the UI to show a dropdown for it."
  - "Add a 'deleted_at' field to the database and show it on the admin page."
- **Negative Examples:**
  - "Refactor the authentication flow." (High risk, release sensitive).
- **Execution Path:** Candidate for `gated_parallel_allowed`.

#### 2.3 `architectural_orchestration_required`
- **Definition:** The task fundamentally changes system architecture, involves creating new subsystems, or breaks existing major interfaces.
- **Positive Examples:**
  - "Migrate the project from REST to GraphQL."
  - "Implement the new M7 dynamic routing layer."
- **Negative Examples:**
  - "Add a standard CRUD endpoint."
- **Execution Path:** `sequential` (Requires heavy architect involvement).

#### 2.4 `high_risk_release_sensitive`
- **Definition:** The task touches critical paths (payments, auth, core data integrity) where partial failures are catastrophic.
- **Positive Examples:**
  - "Update the Stripe webhook handler."
  - "Fix the bug where users can bypass 2FA."
- **Execution Path:** `sequential`

### 3. Layer B: Domain Lead

The Domain Lead specifies which primary engineering discipline owns the core logic of the task.

- **`fe_led`**: Pure UI/UX, client-side logic, CSS changes.
- **`be_led`**: API logic, database schemas, background workers.
- **`fullstack`**: Balanced BE and FE changes.
- **`infra`**: Docker, CI/CD, deployment configurations.
- **`architecture`**: System design, contract creation, deep refactoring.

### 4. Ambiguity Handling & Safe Fallback

If the classifier cannot determine the Work Shape with high confidence (e.g., the user prompt is vague like "Fix the dashboard"), the system MUST default to the following safe fallback values:

- **Work Shape:** `single_branch_safe`
- **Parallel Candidate:** `false`
- **Model Tier:** `balanced_default`
- **Confidence Band:** `low`

This explicitly prevents uncertain tasks from being dynamically promoted to parallel execution.