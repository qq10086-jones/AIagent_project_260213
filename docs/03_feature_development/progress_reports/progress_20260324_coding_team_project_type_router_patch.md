# Progress Report: Coding Team Project-Type Router Patch

- Date: 2026-03-24
- Scope: Discord `/coder` routing, vNext brain router, coding workflow/project-type decoupling, regression coverage

## Completed

1. Fixed the direct-chat regression where simple chat prompts could fall through to `human_review_required`.
2. Decoupled `coding_team_v0` from the CRM-only assumption.
3. Added a project-type resolver for Coding Team requests:
   - `webapp_crm`
   - `single_file_html`
   - `generic_app`
   - `generic_coding_task` fallback
4. Updated Discord `/coder` override to select project type from request content instead of forcing `webapp_crm`.
5. Updated registry validation so one shared workflow can legally serve multiple project types.
6. Updated workflow step defaults so implementation/deploy target paths are no longer hard-coded to CRM-only paths.
7. Added regression tests covering:
   - high-confidence chat direct reply
   - legacy `qa` intent normalization
   - `/coder` HTML request -> `single_file_html`
   - CRM request -> `webapp_crm`
   - shared-workflow registry validation
8. Wrote phase-1 design and task documents for the generic project-type patch.

## Verification

- Targeted tests passed:

```bash
node --test orchestrator/test/coding_project_type.test.js orchestrator/test/discord_dispatch.integration.test.js orchestrator/test/brain_router.integration.test.js orchestrator/test/runtime_dispatch.integration.test.js orchestrator/test/chat_entrypoint.integration.test.js
```

- Orchestrator restarted successfully after the patch.
- Health check passed: `http://localhost:3000/health -> ok`

## In Progress

1. Phase 2 prompt/artifact contract generalization is not done yet.
2. Existing PM/Architect/Impl/QA prompt scripts may still carry CRM-shaped artifact expectations even though routing is now generic.

## Remaining Risks

1. Historical governance/replay fixtures still reference `crm` / `webapp_crm` heavily and may need normalization in a follow-up patch.
2. Exposure/cohort policy now allows the new project types, but production behavior should still be validated with real Discord smoke tests.
3. Some non-router generated files in the workspace remain user-local or environment-local and were intentionally not included in this patch.

## Next Recommended Tasks

1. Generalize prompt-script contracts by project type so non-CRM runs produce shape-correct artifacts end to end.
2. Add Discord end-to-end smoke cases for:
   - `/coder ?â‰é àÍò¢ htmlÅCì‡óeèAé ÅFhello, world`
   - `/coder òÙàÍò¢ä«óùç@ë‰`
   - `/coder Build a CRM MVP with customer portal`
3. Review replay/governance fixtures and migrate remaining `crm` aliases to the new generic/shared workflow model where appropriate.
