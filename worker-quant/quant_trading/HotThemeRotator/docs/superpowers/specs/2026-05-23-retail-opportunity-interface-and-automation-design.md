# Retail Opportunity Interface And Automation Design

## Goal

Build the next HotThemeRotator experience around a retail-friendly "today's opportunity center" while keeping the long-term path toward automation explicit, gated, and research-only until feedback calibration and human approval exist.

## Approved Direction

Use design option A: "今日机会中心".

The first screen must answer four practical questions in plain language:

- 今天最值得盯什么
- 什么价格可以考虑买
- 什么价格应该止损或卖出
- 系统为什么这么判断，以及当前风险在哪里

The dense table, raw Markdown, and internal reason codes remain available but no longer dominate the first screen.

## Automation Roadmap

Automation must advance in gated stages:

1. Candidate discovery: scan a broad symbol pool using point-in-time quotes, news, volume, liquidity, relative strength, and context.
2. Opportunity panel: rank candidates and generate staged entry, stop, and exit ladders.
3. Decision logging: save every candidate row with model version, input snapshot, timestamp, ladder, reasons, and data gaps.
4. Feedback joining: attach realized 1D, 3D, and 5D outcomes to every logged ladder.
5. Calibration: convert uncalibrated research scores into calibrated win-rate estimates only after enough feedback samples exist.
6. Alerts: send human-readable notifications when candidates cross watched buy/sell levels.
7. Paper trading: simulate executions under fixed risk limits and compare against realized results.
8. Broker execution: only after explicit human approval, passing paper gates, kill-switches, position limits, and audit logs.

The current implementation may improve UX and expose the roadmap, but it must not present current scores as true win rates and must not create live orders.

## UI Requirements

The Streamlit app should use a work-focused dashboard layout:

- A top summary band with candidate count, top candidate, suggested refresh interval, and calibration state.
- A primary card for the top candidate with plain-language action wording.
- Price ladder cards for aggressive, balanced, conservative entry, stop, and staged sells.
- Reason and risk areas written for a general retail user.
- A candidate list table for scanning more names.
- Detail tabs for all candidates, automation roadmap, rules, and raw Markdown.

The UI must avoid a marketing landing page. It should open directly into the usable dashboard.

## Data Model

Keep presentation preparation in `src/hot_theme_rotator/ui/opportunity_dashboard.py`.

Add small helper outputs that are easy to test without Streamlit:

- Retail candidate cards: rank, symbol, theme label, score, priority, action text, buy zone, sell zone, stop, reason summary, risk summary, data quality.
- Retail summary metrics: candidate count, top symbol, top action, calibration label.
- Automation roadmap rows: stage, current status, next gate.

Streamlit should consume these helpers and render them. It should not recompute opportunity scores or ladder formulas.

## Error Handling

The page should fail closed:

- Empty symbol pools show a user-readable error.
- Data-loading exceptions show a user-readable error and do not produce advice.
- Missing news/context data is surfaced as data quality risk.
- Any score that is not calibrated remains labeled as uncalibrated research output.

## Testing

Unit tests must cover:

- Retail card generation for the deterministic sample panel.
- Plain Chinese reason and risk translation.
- Summary metric generation.
- Automation roadmap stage labels.

Verification must include:

- Targeted UI helper tests.
- Python compilation of the Streamlit app.
- Full pytest suite.
- HTTP 200 from the local Streamlit app.

