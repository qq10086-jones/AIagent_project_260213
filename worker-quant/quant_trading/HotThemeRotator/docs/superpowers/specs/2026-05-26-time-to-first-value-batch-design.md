# Time-To-First-Value Batch Design

## Goal

Move HotThemeRotator from "components exist" to a locally usable, pull-first personal advisory system over a 10-15 day implementation batch. The batch must preserve Rule 3 advice-only behavior, Rule 12 staged activation, and the user's local-only constraint.

## Recommended Approach

Use the staged route already approved in `PROJECT_STATUS.md`:

1. Close W1 data health by hardening delayed price source access.
2. Build the P10-20 Daily Advisory Cockpit as the first everyday pull-only surface.
3. Add P10-17 Silent Watchlist Intelligence so scheduled scans can queue research events without pushing notifications.
4. Start P10-18 Anti-FOMO core models, but keep desktop/email/Telegram disabled until the full guarded-push gate is complete.

This is preferred over UI-first work because the cockpit must not make stale or uncertain scrape data look fresh. It is preferred over push-first work because the user's stated priority is fast personal use, not proactive notification.

## Scope

### P10-19 Network Hardening

Add a small HTTP policy layer for delayed price scrapers. It will provide:

- per-host rate limiting with injectable clock and sleep for deterministic tests;
- browser-like User-Agent rotation from a fixed conservative list;
- robots.txt allow/deny checks through an injectable rules provider;
- Cloudflare / anti-bot page detection that aborts before parsing;
- explicit failure reasons suitable for `PriceSourceHealth`.

This layer does not make scraping "real time" and does not bypass source access restrictions.

### P10-20 Daily Advisory Cockpit

Expose a pull-only daily payload that combines:

- positions and watchlist symbols;
- delayed quote state with source, `data_ts`, `wall_ts`, `data_ts_inferred`, `price_uncertain`;
- TDnet disclosure counts and recent items;
- seven-level ladders where a reference price is available;
- data gaps and research-only / uncalibrated status;
- silent queue summary once P10-17 exists.

The cockpit may be CLI/API/frontend visible, but must remain GET-only and must not call any notifier.

### P10-17 Silent Watchlist Intelligence

Create a silent queue of watchlist events:

- quote unavailable/stale/uncertain;
- TDnet disclosure for watched symbol;
- ladder proximity event;
- large intraday move marked study-only when chase-risk is present.

The queue is system-state read-only evidence. It does not notify the user and does not place orders.

### P10-18 Anti-FOMO Core

Implement the first pure-domain guard models:

- alert budget;
- stale-data fail-closed;
- chase filter;
- cooling-off for newly watched symbols.

These models are consumed by later guarded push work, but this batch does not enable any external notification channel.

## Data Flow

Delayed scraper HTTP fetch -> HTTP policy layer -> source parser -> `PriceQuote` -> `PriceSourceHealth` -> daily health report -> cockpit payload -> dashboard/briefing.

Watchlist symbols -> quote/disclosure/ladder checks -> silent `AlertRecord`-like queue -> cockpit summary. Guard models may downgrade or suppress queue entries but do not push them.

## Error Handling

All external source failures become structured degraded state:

- blocked by robots;
- rate limited;
- Cloudflare / anti-bot detected;
- parse failed;
- all sources failed;
- inferred timestamp;
- consensus unavailable or mismatch.

The cockpit must show unavailable/stale/degraded data rather than silently falling back to old values.

## Testing

Implementation follows TDD. Each new behavior gets a failing test first:

- unit tests for HTTP policy decisions and deterministic rate limiting;
- mock HTTP integration for Yahoo/Kabutan source access;
- cockpit payload contract tests proving GET-only, no notifier, no win-rate language;
- silent queue tests proving no notification side effects;
- guard tests for budget, stale, chase, and cooling-off behavior.

Full-suite verification remains required before claiming a batch task complete.

## Non-Goals

- No broker, order, paper-order, or execution API.
- No POST/PUT/DELETE/PATCH endpoints.
- No desktop/email/Telegram notification enabling.
- No calibrated win-rate label before Rule 9.4 evidence.
- No remote upload, push, PR, or sync unless the user explicitly asks.
