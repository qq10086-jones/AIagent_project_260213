# OpenClaw Nexus Progress Patch

- Patch Time: 2026-02-26 13:58:51
- Version: `OpenClaw_Nexus_v1_2_5_PATCH`
- Baseline: `docs/03_feature_development/progress_reports/progress_20260226_120858_OpenClaw_Nexus_v1_2_4.md`

## Patch Goal

Address two gaps in stock-analysis output:

1. `recent news` often empty or low quality.
2. Report content did not reflect your `quant_trading` template-driven analysis style.

## Completed in This Patch

1. Template-driven quant report output
- Integrated `worker-quant/quant_trading/execution_report_template.md` into `quant.deep_analysis`.
- Render full report as:
  - Markdown (`report_markdown`)
  - HTML (`report_html_object_key` archived to MinIO)
- Report now includes structured sections from the template and quant fields.

2. Quant factor analysis added to single-symbol flow
- Added deterministic metrics in `worker-quant/worker.py`:
  - `ret_5d_pct`, `ret_20d_pct`, `ret_60d_pct`
  - `vol_20d_pct`, `rsi14`, `sma20`, `sma60`, `z20`
  - benchmark-relative return and `alpha_score`
  - derived `signal` and `risk_state`
- Brain narrative now shows these metrics in final message.

3. Recent news pipeline improved
- News merged from:
  - yfinance native news
  - Yahoo quote/news pages
  - Google News RSS
- Added relevance scoring + dedupe + filtering:
  - Drop login/noise links
  - Prefer trusted publishers
  - Block low-quality noisy publisher patterns

4. Discord long-report delivery upgraded
- Orchestrator now supports:
  - short preview in chat
  - full HTML/MD report attachment when content is too long
- Avoids 2000-char message limit while preserving full report readability.

## Changed Files

- `worker-quant/worker.py`
- `brain/supervisor.py`
- `brain/main.py`
- `brain/state.py`
- `orchestrator/src/index.js`

## Runtime Validation

1. Build/restart completed for affected services
- `worker-quant`, `brain`, `orchestrator` rebuilt and started.

2. End-to-end test on `9432.T`
- Analysis returns:
  - symbol/company/price facts
  - quant metrics + signal
  - non-empty `recent news` (includes Reuters in latest test)
  - `report_markdown` + `report_html_object_key`
- Task states in DB:
  - `quant.deep_analysis = succeeded`
  - `browser.screenshot = succeeded`

## Current Residual Risk

- `recent news` quality still depends on external feed mix and publication language.
- Next hardening step should add per-symbol/entity matching (ticker/company alias strict match) before ranking.

## Next Patch Suggestion

1. Add PDF export (`wkhtmltopdf` or headless Chromium print-to-pdf) and attach PDF in Discord.
2. Add strict news relevance guard:
   - exact ticker/company alias match score threshold
   - drop market-wide headline unless symbol is explicitly mentioned.
