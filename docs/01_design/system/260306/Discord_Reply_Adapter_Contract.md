# Discord Reply Adapter & Artifact Timeline Contract

## Scope

This contract defines the minimum `WS-08-04` (Replay) and the final response aggregation slice for completing `WS-08`.

Current phase scope:
- Adapter to convert `final_result_package` into Discord-compliant `DispatchSuccessResponse`.
- Timeline aggregation logic to pull historical data into a linear text-based trace.
- Fulfills the `WS-08` final downstream integration.

## Input & Output Schemas

**Discord Reply Adapter:**
- Input: `final_result_package` + `task_envelope`
- Output: `dispatch_success_response` (mode: `direct_reply`)
- Validation: Hard failure (`INVALID_RESULT_PACKAGE`) if `final_result_package` is invalid.

**Artifact Timeline Replay:**
- Input: SQL pool + `workflow_run_id`
- Output: Structured JSON timeline array
- Formatter: Pure string conversion `formatTimelineAsText(timelineObj)`

## Runtime Behavior

- Any result successfully navigating the pipeline will be aggregated via `buildFinalResultPackage` and subsequently transformed to a pure reply string using `buildDiscordCompletionReply`.
- If a replay is requested, `queryWorkflowTimeline` accesses the database pool directly, retrieving runs, steps, and checkpoints. No UI rendering is allowed per project restrictions.

## Non-Scope
- No Web dashboards.
- No direct external Discord API calls (this strictly produces the payload).
- No new tables in the SQL schema (reuses existing).
