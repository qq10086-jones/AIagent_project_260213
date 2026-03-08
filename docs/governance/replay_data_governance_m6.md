# OpenClaw Nexus M6 — Replay Data Governance

- Version: 1.0
- Date: 2026-03-09
- Milestone: M6
- Status: APPROVED

---

## 1. Purpose

This document defines the governance rules for replay corpus data used in M6 staging validation. Replay fixtures are operational artifacts that simulate real Discord-originated workflow prompts. Because they may be derived from real user interactions, they require explicit rules for retention, sanitization, and access.

---

## 2. Raw Prompt Retention Policy

**Raw Discord prompts may NOT be stored in version control or in any approval artifact.**

Rationale: raw prompts may contain user identifiers, channel identifiers, project-sensitive paths, and secrets that cannot be safely committed to a repository.

| Storage Location | Raw Prompt Allowed |
|------------------|--------------------|
| Version-controlled fixture files | NO |
| Staging run artifacts | NO |
| Approval / closure packages | NO |
| Developer local machine (temporary, for sanitization only) | YES — must be deleted after sanitization is complete |

All `raw_prompt_ref` fields in the replay manifest must be set to `null` once the sanitized fixture is finalized. Retaining a reference to a raw prompt file is only permitted if that file is stored outside version control under a separately governed data boundary, and only for traceability during the sanitization review step.

---

## 3. Mandatory Sanitization Rules

Before any replay fixture is written to disk or committed, all of the following fields must be sanitized:

### 3.1 User Identifiers

- Replace all Discord user IDs, usernames, display names, and @mentions with the token `SANITIZED_U_NNN` where NNN is a sequential index
- This includes indirect references (e.g., "ask [username] about the spec")

### 3.2 Channel Identifiers

- Replace all Discord channel IDs, channel names, server names, and #channel-references with `SANITIZED_CH_NNN`
- Guild / server identifiers must also be replaced

### 3.3 Links and Attachments

- Remove all URLs from prompt text
- Remove all attachment references
- Set `attachments` array to `[]`
- If a URL is essential context for the prompt (e.g., a design reference), replace the domain with `SANITIZED_DOMAIN` and remove the path

### 3.4 Secrets and Environment Hints

- Remove all API keys, tokens, passwords, and connection strings
- Remove environment variable names that reveal infrastructure topology (e.g., `PROD_DB_HOST`)
- Remove any reference to internal service hostnames, IP addresses, or port numbers

### 3.5 Repository-Specific Sensitive Paths

- Remove or genericize file paths that reveal proprietary module names, customer project names, or internal codenames
- Acceptable replacement: use generic path tokens such as `src/[module]/[file].js`

### 3.6 Timestamps

- Round all timestamps to the nearest hour
- Apply a random offset of ±1 to ±7 days before storing

---

## 4. Sanitization Review Process

1. A contributor drafts the sanitized prompt from a real or representative Discord interaction
2. A second contributor reviews the sanitized prompt against Section 3 rules before the fixture is committed
3. The reviewer confirms all identifiers, secrets, and sensitive paths have been removed
4. Both contributor and reviewer must have Architect-level approval to participate

No automated tooling alone is sufficient to approve a fixture. Human review is required.

---

## 5. Who May Generate and Review Fixtures

| Role | Generate Fixtures | Review Fixtures | Commit to Repo |
|------|-------------------|-----------------|----------------|
| Architect (project owner) | YES | YES | YES |
| Senior contributor (Architect-approved) | YES | YES | YES |
| Other contributor | NO | NO | NO |

Fixture generation and review rights are scoped to M6 and must be re-authorized for future milestones.

---

## 6. Retention Rules

### 6.1 Replay Fixtures

- Sanitized fixture files under `orchestrator/replay/fixtures/` are retained indefinitely in version control
- Fixtures may be updated only through the sanitization review process described in Section 4
- Deprecated fixtures must be removed rather than left in place with a comment

### 6.2 Staging Run Artifacts

- Artifacts written to `orchestrator/artifacts/m6_staging_replay/` are retained for **90 days** from the date of the staging run
- After 90 days, staging artifacts must be deleted or formally archived under `docs/90_archive/`
- Staging artifacts must never contain raw prompt text

### 6.3 Approval and Closure Packages

- Documents under `docs/governance/` are retained indefinitely
- Approval packages must reference only sanitized fixture IDs (e.g., `R001`) and structured result bundles
- No approval package may embed or reference raw prompt content

---

## 7. Redaction Procedure

If a committed fixture is discovered to contain unsanitized data:

1. Remove the fixture from the repository immediately via a targeted commit
2. Rotate any exposed secrets if applicable
3. Re-sanitize the fixture following Section 3 rules
4. Perform a fresh review following Section 4 before re-committing
5. Record the incident in `docs/governance/replay_data_incidents.md`

---

## 8. Compliance with M6 Design Addendum

This governance document fulfills the requirements of:

- `OpenClaw_Nexus_Design_Document_v3.2.md` Section 6 (Replay Corpus Governance)
- `open_claw_nexus_engineering_task_list_m6_v3.md` WS-23-01.5

The `orchestrator/replay/README.md` file provides a concise on-disk reference to these rules for day-to-day contributor use.

---

## 9. Approval

- Reviewed and approved by: Architect (project owner)
- Date: 2026-03-09
- Next review: at M6 closure or if replay corpus scope changes materially
