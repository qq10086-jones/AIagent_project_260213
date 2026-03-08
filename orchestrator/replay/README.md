# Replay Corpus — Sanitization and Governance Rules

Version: 1.0
Date: 2026-03-08
Governance doc: `docs/governance/replay_data_governance_m6.md`

---

## Purpose

This directory contains sanitized replay fixtures for M6 staging validation. Fixtures are derived from representative Discord-originated workflow prompt patterns and must never contain raw user data.

---

## Sanitization Rules

All replay fixtures must comply with these rules before storage:

| Field | Rule |
|-------|------|
| `user_id` | Replace with `SANITIZED_U_NNN` |
| `channel_id` | Replace with `SANITIZED_CH_NNN` |
| `message` | Remove all @mentions, channel references, URLs, file paths containing project names, and any secret-like strings |
| `attachments` | Strip all attachments; record `[]` |
| `timestamp` | Round to nearest hour; shift by random offset within ±7 days |

## Who May Create Fixtures

Only project contributors with Architect-level approval may generate or modify replay fixtures.

## Retention Rules

- Sanitized fixtures: retained indefinitely under version control
- Raw prompts: not retained (all `raw_prompt_ref` fields are null)
- Staging run artifacts: retained for 90 days under `orchestrator/artifacts/m6_staging_replay/`
- Approval packages: retained indefinitely

## Directory Structure

```
orchestrator/replay/
  manifests/
    m6_staging_replay_manifest.json   — corpus index (50 cases)
    m6_staging_coverage_summary.json  — coverage floor verification
  fixtures/
    pm_heavy.json
    arch_heavy.json
    be_led.json
    fe_led.json
    qa_heavy.json
    mixed_ambiguous.json
```
