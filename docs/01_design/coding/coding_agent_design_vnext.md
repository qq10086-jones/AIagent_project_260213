# Coding Agent Design (vNext)

## Date
2026-03-01

## Why This Version
Current implementation is a self-built patch executor (`coding.patch` + `coding.execute`).  
Target state is different: Discord `/coder:` should **delegate** real coding work to existing coding software (Codex / Claude Code), while **Nexus remains the orchestrator + governance layer**.

This document is the authoritative design for that target and is written to be **implementable** (clear workspace contract, phases, logging, gating, rollback).

---

## Problem Statement
- **Current state:** Nexus Coder generates SEARCH/REPLACE and applies patches itself.
- **Desired state:** Nexus Coder delegates implementation to external coding agents, then collects artifacts, diffs, and test outcomes.
- **Gap:** no provider adapter layer exists yet; workspace/rollback/logging/gating are underspecified.

---

## Scope
### In scope
- Discord `/coder:` as the single command entry.
- Delegation to provider adapters (`codex`, `claude_code`) via `coding.delegate`.
- Approval, audit, artifact capture, and run trace in Nexus.
- Keep existing `coding.patch` / `coding.execute` as fallback.

### Out of scope
- Rebuilding a full Codex/Claude-like CLI product.
- Training a new coding foundation model.

---

## Target Architecture
1. User sends `/coder: <task>` in Discord.
2. Orchestrator creates `run_id` and enqueues `coding.delegate`.
3. `worker-coder` resolves provider by policy (`auto|codex|claude_code`).
4. Worker materializes a **workspace** (checkout/clone/worktree) and enforces **gates** (write/exec/network).
5. Adapter runs external coding software in non-interactive mode against workspace.
6. Worker captures:
   - normalized result (`summary`, `files_changed`, `diff_stats`, `test_result`)
   - raw logs (stdout/stderr)
   - artifacts (diff bundle, patch, reports)
   - git anchor (base_ref/branch/commit_sha) and rollback status
7. Worker writes facts and task result to DB/streams.
8. Orchestrator posts summary back to Discord and exposes approval actions if needed.

---

## Core Tool Contract

### Tool name
`coding.delegate`

### Payload schema (MVP, implementable)

```json
{
  "task_prompt": "string (required)",
  "provider": "auto|codex|claude_code (optional, default auto)",
  "model": "string (optional)",

  "repo": {
    "type": "local_path|git_url|workspace_id",
    "value": "string"
  },

  "workspace": {
    "workspace_subdir": "string (optional)",
    "base_ref": "string (optional, default main)",
    "branch_name": "string (optional, default nexus/<run_id>)",
    "checkout_mode": "copy|worktree|in_place (optional, default worktree)",
    "clean_before": "bool (default true)",
    "clean_after": "bool (default true)"
  },

  "execution": {
    "apply_changes": "bool (default true)",
    "dry_run": "bool (default false)",
    "test_command_id": "string (optional, must be allowlisted; e.g. pytest|npm_test|make_test)",
    "test_args": ["string (optional, allowlisted flags only)"],
    "max_runtime_s": "int (default 600)",
    "network": "off|on (default off)",
    "dependency_policy": "lockfile_only|allow_install|deny_install (default lockfile_only)"
  },

  "debug": {
    "codex_command": ["string (optional, dev only)"]
  }
}
```

**Notes**
- `test_command` is intentionally split into `test_command_id + test_args` to reduce shell injection risk.
- If `repo.type=local_path`, the worker MUST still enforce `checkout_mode` and keep changes isolated by default.
- `debug.codex_command` is a development escape hatch to inject an explicit adapter command; it MUST be disabled in production policy.

### Output schema

```json
{
  "ok": "bool",
  "provider_used": "codex|claude_code|patch_fallback",
  "model_used": "string|null",

  "summary": "string",
  "files_changed": ["string"],
  "diff_stats": { "added": "int", "deleted": "int", "files": "int" },
  "test_result": "passed|failed|skipped",

  "git": {
    "base_ref": "string",
    "branch": "string",
    "commit_sha": "string|null",
    "dirty": "bool"
  },

  "rollback_performed": "bool",
  "artifacts": {
    "diff_bundle": "path_or_object_key|null",
    "patch_file": "path_or_object_key|null",
    "test_log": "path_or_object_key|null",
    "raw_stdout": "path_or_object_key|null",
    "raw_stderr": "path_or_object_key|null"
  },

  "diagnostics": {
    "error_code": "string|null",
    "exit_code": "int|null",
    "timeout": "bool",
    "parse_error": "bool",
    "truncated": "bool"
  },

  "error": "string|null"
}
```

---

## Workspace & Repo Contract (Critical)

### Workspace materialization
The worker must create an isolated workspace directory per run:

- `workspace_root = <workspaces>/<run_id>/`
- `repo_root = workspace_root/<repo_name>/` (or mount target)

### Checkout modes
- `worktree` (default): `git worktree add` into `workspace_root`, creates an isolated branch.
- `copy`: copy repo snapshot into workspace; useful for non-git inputs.
- `in_place`: only allowed in dev or explicitly approved; risky.

### Atomicity & rollback
- All modifications must occur in `repo_root` under a single isolation boundary (branch/worktree/copy).
- If any phase fails after modifications occur:
  - worker MUST rollback (remove worktree/reset branch or discard copy) **before** returning failure, unless `clean_after=false` for debugging in dev.
- Worker MUST report `git.dirty` and `rollback_performed`.
- Worker MUST compute `files_changed` as **delta against pre-run git status snapshot**, not full repository dirty set.

---

## Execution Phases (Deterministic Pipeline)

1. **Prepare**
   - enforce gates (write/exec/network) based on payload + policy
   - materialize workspace and set `base_ref`, `branch_name`
2. **Plan (provider-native)**
   - construct non-interactive prompt and command
3. **Apply**
   - run provider; if `apply_changes=false` or `dry_run=true`, provider must only output suggested patch/diff
4. **Verify**
   - run allowlisted test command (if provided)
5. **Package**
   - generate diff bundle (`git diff base_ref..HEAD` or patch file)
   - store raw logs and structured events
6. **Finalize**
   - optional commit (policy-driven; default: commit changes on branch)
   - on failure: rollback unless dev debugging mode is enabled

---

## Provider Adapter Layer

### Implementation layout
- `worker-coder/adapters/codex_adapter.*`
- `worker-coder/adapters/claude_code_adapter.*`
- shared wrapper:
  - timeout / process kill
  - env injection (safe allowlist)
  - output normalization + error taxonomy
  - log capture + truncation handling

### Adapter responsibilities
- Build non-interactive command and prompt.
- Execute in sandboxed workspace.
- Parse output into normalized result schema.
- Return deterministic error codes on failure.

### Required logging protocol
Adapters MUST return:
- `artifacts.raw_stdout` and `artifacts.raw_stderr` (always, even on success)
- `diagnostics.exit_code`, `diagnostics.timeout`, `diagnostics.truncated`
- structured **event stream** written by worker (recommended):
  - `phase_start/phase_end`
  - `cmd_start/cmd_end`
  - `artifact_written`
  - `gate_required/approved/denied`

---

## Safety & Governance (Non-Negotiable)

### Risk level
- `coding.delegate` default risk: `high`
- In **prod**: high-risk actions require explicit approval.
- In **dev**: allow auto-run with strict sandboxing, but still log everything.

### Governance gates (3 independent gates)
1. **Write Gate**
   - triggers when touching non-allowlisted paths or sensitive files (CI, infra, auth, secrets)
2. **Exec Gate**
   - triggers when executing non-allowlisted commands or non-allowlisted flags
3. **Network Gate**
   - triggers when `execution.network=on` or dependency installation is requested

Each gate can be:
- `auto_allow` (dev only)
- `require_approval`
- `deny`

### Allowlists
- `writable_paths`: repo-relative allowlist (default deny)
- `executable_commands`: `test_command_id` allowlist (default deny)
- `env_allowlist`: only safe env vars injected; secrets must be injected via approved mechanism and masked in logs

### Hard limits
- runtime timeout: `max_runtime_s`
- max output size: truncate and set `diagnostics.truncated=true`
- store raw logs regardless of truncation

---

## Error Taxonomy (Deterministic)
Adapters and worker should map failures into stable codes:

- `E_PROVIDER_UNAVAILABLE`
- `E_POLICY_DENY`
- `E_TIMEOUT`
- `E_WORKSPACE_DIRTY`
- `E_APPLY_FAILED`
- `E_TEST_FAILED`
- `E_PARSE_ERROR`
- `E_INTERNAL`

Fallback behavior depends on this taxonomy (below).

---

## Backward Compatibility & Fallback Rules

### Keep existing tools
- `coding.patch`
- `coding.execute`

### Fallback policy (safe)
Fallback to patch/execute is permitted **only if**:
- error is `E_PROVIDER_UNAVAILABLE` or `E_POLICY_DENY` (and user allows fallback), AND
- `git.dirty=false` (no partial modifications), OR rollback has been performed successfully.

If provider fails after making changes and rollback fails, **do not fallback**; return failure and require manual intervention.

---

## Model Strategy
- Official Codex model capability is cloud-native.
- Local GLM can still be used:
  - as fallback planner/reviewer in Nexus
  - via a custom compatible adapter endpoint if available
- Therefore:
  - cloud is not strictly required for the delegation architecture
  - cloud is effectively required if you want official Codex behavior

---

## Discord UX Contract
- `/coder: <task>`: default delegation entry.
- Optional flags (future):
  - `/coder:provider codex <task>`
  - `/coder:model <name> <task>`
  - `/coder:dry-run <task>`
  - `/coder:test pytest -q <task>` (maps to `test_command_id=pytest`, args `-q`)

Quant and other skills remain unchanged.

---

## Provider Policy (MVP Defaults)
To avoid ambiguity, define an explicit MVP default:

- **Phase 1:** `auto -> codex` (single provider), fallback to patch/execute if policy allows
- **Phase 2:** add `claude_code` and enable selection by repo policy
- **Phase 3:** choose provider by task type (bugfix/refactor/docs) and cost/security profile

---

## Migration Plan
1. **Phase 1 (MVP Delegation)**
   - implement `coding.delegate` spec + storage (DB/artifacts)
   - implement workspace materialization + rollback
   - implement **one** adapter end-to-end (default provider)
   - implement gates (Write/Exec/Network) with dev auto-allow, prod require-approval
2. **Phase 2 (Dual Provider)**
   - add second adapter
   - add provider selection policy and safe fallback
3. **Phase 3 (Coder-Centric Skill Fabric)**
   - extend same pattern to `/ui:`, `/db:`, etc.
   - keep Coder as primary orchestrator

---

## Acceptance Criteria
- `/coder:` can complete a real coding task via delegated provider.
- Nexus stores full audit trail (task, logs, artifacts, git anchors, result).
- Approval and rollback controls remain functional.
- Fallback never runs on a dirty/half-applied workspace.
- Quant flows are unaffected.

---

## Open Questions (Now narrowed)
- In prod, do we allow `in_place` checkout mode at all? (recommend: deny)
- Do we auto-commit on success, or keep changes uncommitted on a branch? (recommend: auto-commit with commit message `nexus: <run_id>`)
- Do we require repo-level policy file (`.nexus/policy.yml`) to define allowlists?
