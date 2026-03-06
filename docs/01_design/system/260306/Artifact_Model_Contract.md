# Artifact Model Contract

## Scope

This contract defines the minimum `WS-08-01` / `WS-08-02` slice for artifact metadata and persistence.

Current phase scope:
- artifact metadata schema
- normalized asset persistence records
- release pack / step artifact / minio archive coverage

## Artifact Metadata Schema

Schema:
- [artifact_metadata.schema.json](C:\Users\linweiye\AIagent_project_260213\orchestrator\contracts\artifact_metadata.schema.json)

Required fields:
- `artifact_id`
- `task_id`
- `role`
- `type`
- `path`
- `mime`
- `created_at`
- `summary`

## Runtime Behavior

Current runtime hard checks:
- persisted assets must first normalize into artifact metadata
- artifact metadata must be schema-valid before DB insert
- asset metadata is stored inside `assets.metadata_json.artifact_metadata`

## Non-Scope

- no replay UI in this task
- no final Discord result package redesign in this task
- no artifact browser redesign in this task
