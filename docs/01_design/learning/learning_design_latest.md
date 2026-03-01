# Learning Design (Latest)

## Scope
Consolidated latest design for learning/guardrails loop used by Nexus.

## Core Components
- Trace and feedback capture (`traces`, reaction feedback).
- Rule/Memory stores (`rules`, `mem_items`) for runtime prompt injection.
- Post-processing hard-rule check with single-pass rewrite for local model output.

## Runtime Loop
1. User/task response is generated.
2. Hard-rule validator checks project-specific constraints.
3. On violation, rewrite prompt is injected and response regenerated once.
4. User feedback is stored and can update soft rules/memory entries.

## Guardrail Objectives
- Prevent unsafe/undesired output drift.
- Keep instructions consistent across model switches.
- Make preference updates observable and replayable.

## Current Baseline
- Rule-aware context build is active.
- Reaction-based feedback ingestion is active.
- Learning guardrails patch is integrated in orchestration path.

## Related Docs
- Historical learning design snapshots archived under `docs/90_archive/legacy_workspace/AIagent_project`.
- Active guardrails patch design: `docs/01_design/learning/learning_guardrails_patch_design.md`.
