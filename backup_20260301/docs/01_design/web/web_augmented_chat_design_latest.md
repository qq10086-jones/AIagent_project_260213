# Web-Augmented Chat Design (Latest)

## Scope
Latest consolidated design for real-time web-augmented chat in Nexus.

## Intent Policy
- Real-time/freshness-sensitive queries should trigger tool execution.
- Use lightweight search first, then escalate to browser-heavy capture when needed.

## Execution Path
1. Router detects search/browse intent.
2. Worker runs `web.search_and_browse` or news tools.
3. Results are synthesized into natural-language response with source references.

## Quality Controls
- Prefer concise synthesis over raw task cards.
- Keep source links and freshness context.
- For screenshots, blank artifacts are filtered before return.

## Current Baseline
- Light-track web search is active.
- Hot-news active search supports global source mix and holdings-priority enrichment.

## Related Docs
- Legacy v1/v2 web design docs archived under `docs/90_archive/legacy_workspace/AIagent_project`.
