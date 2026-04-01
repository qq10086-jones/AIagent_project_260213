# AIagent Project 260213

## What This Repository Is

This repository is the working monorepo for the Nexus control-plane stack built around OpenClaw-based runtime components.

The core product paths are:

- `orchestrator/`: workflow orchestration, validation, canary, runtime coordination
- `worker-coder/`: coding worker runtime and adapter layer
- `worker-quant/`: quant/report worker implementation
- `brain/`: Python-side supervisor and routing experiments that are still part of the active stack
- `shared/`: shared JS module surface
- `configs/`: tool and runtime configuration
- `infra/`: Docker Compose and deployment wiring
- `docs/`: architecture, design, governance, patch history
- `ui/`: local UI entrypoint

## Repository Navigation

### Primary Paths

These are the directories you should treat as the main product surface:

- `orchestrator/`
- `worker-coder/`
- `worker-quant/`
- `brain/`
- `shared/`
- `configs/`
- `infra/`
- `scripts/`
- `docs/`
- `ui/`

### Secondary Paths

These exist for support, experimentation, or external dependency management, but they are not the main product entrypoints:

- `external/openclaw/`: upstream or forked external runtime dependency, currently tracked as a gitlink
- `external/vendor/`: vendored external assets and references
- `workspace/sandbox/`: isolated sandboxes and demo implementations
- `workspace/scratch/`: temporary local investigation output
- `archive/backup_20260301*`: historical filesystem snapshots
- `artifacts/`: generated runtime output and retained baselines
- `metrics/`: generated metrics or benchmark output

### Governance Docs

Start here for structure and cleanup guidance:

- `docs/00_overview/workspace_reorg_plan_20260401.md`
- `docs/00_overview/docs_reorg_20260301.md`
- `docs/00_overview/docs_migration_map_20260301.md`

## Architecture Summary

At a high level, the stack is organized like this:

- OpenClaw runtime provides the AI gateway and external model/tool surface.
- `orchestrator/` coordinates workflows, validation lanes, canaries, and release gating.
- `worker-coder/` executes coding-oriented workflows and contract-bound task steps.
- `worker-quant/` handles report generation and quant-specific logic.
- `infra/` wires local services such as Compose-managed dependencies.

## Working Rules

### Source vs Runtime Output

Generated output should not be treated as first-class source code.

Examples of runtime output:

- `artifacts/`
- `orchestrator/artifacts/`
- generated reports under `worker-quant/`
- local databases such as `*.db`
- time-stamped validation and canary directories

When adding new flows, prefer writing runtime output to a dedicated runtime path rather than back into source trees.

### External Code

`external/openclaw/` and `external/vendor/` are external-code surfaces. Do not treat them as ordinary app directories.

Changes there should be deliberate and documented, especially when updating pinned revisions.

### Sandbox Code

`workspace/sandbox/` and `workspace/scratch/` are not production entrypoints. They are useful, but they must stay clearly separated from shipping paths.

## Quick Start

```bash
docker compose -f infra/docker-compose.yml up -d
```

Common endpoints:

- UI: `http://localhost:8501`
- Orchestrator health: `http://localhost:3000/health`
- MinIO: `http://localhost:9001`

## Key Files

- `infra/docker-compose.yml`
- `orchestrator/package.json`
- `worker-coder/package.json`
- `worker-quant/worker.py`
- `configs/tools.json`
- `README.md`

## Current Reorganization Direction

The repository is being normalized around three boundaries:

- source code
- runtime artifacts
- external dependencies

The immediate goal is not a risky full move, but clearer entrypoints and better governance so future cleanup can happen without breaking the active stack.
