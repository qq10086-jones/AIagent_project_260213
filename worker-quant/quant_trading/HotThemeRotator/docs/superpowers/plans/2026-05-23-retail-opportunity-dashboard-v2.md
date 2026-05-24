# Retail Opportunity Dashboard V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Streamlit app into a retail-friendly "今日机会中心" while documenting the gated path to automation.

**Architecture:** Keep all tested presentation preparation in `src/hot_theme_rotator/ui/opportunity_dashboard.py`. Streamlit consumes retail cards, metrics, and automation roadmap rows without recalculating scores. Governance and task docs record the automation gates and P8-07 UI milestone.

**Tech Stack:** Python, Streamlit, pandas, pytest.

---

### Task 1: Retail Presentation Data

**Files:**
- Modify: `tests/unit/test_opportunity_dashboard.py`
- Modify: `src/hot_theme_rotator/ui/opportunity_dashboard.py`

- [x] Add failing tests for retail candidate cards, summary metrics, reason/risk translation, and automation roadmap rows.
- [x] Run `python -m pytest .\tests\unit\test_opportunity_dashboard.py -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp` and confirm the new tests fail for missing helpers.
- [x] Implement retail helper functions in `opportunity_dashboard.py`.
- [x] Re-run the targeted test and confirm it passes.

### Task 2: Streamlit Retail Dashboard

**Files:**
- Modify: `tools/streamlit_opportunity_app.py`
- Modify: `README.md`

- [x] Import the new retail helper functions.
- [x] Replace the first screen with "今日机会中心": summary metrics, top-candidate action card, price ladder cards, reason/risk columns, candidate table, candidate detail, automation roadmap, rules, and raw Markdown.
- [x] Keep yfinance quote-only mode available from the sidebar.
- [x] Keep research-only and uncalibrated-score warnings visible.
- [x] Run `python -m py_compile .\tools\streamlit_opportunity_app.py`.

### Task 3: Automation Governance And Tracking

**Files:**
- Modify: `docs/02_GOVERNANCE.md`
- Modify: `docs/01_TASKS.md`
- Modify: `PROJECT_STATUS.md`

- [x] Add automation gate rules: decision logging, feedback joins, calibration, alerts, paper trading, and broker execution approval.
- [x] Add P8-07 Retail Opportunity Dashboard V2 as done when verified.
- [x] Add P9 automation milestones as pending next work.
- [x] Update current project status and changelog.

### Task 4: Final Verification

**Files:**
- No code files changed in this task.

- [x] Run the targeted UI tests.
- [x] Run `python -m py_compile .\tools\streamlit_opportunity_app.py`.
- [x] Run `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp`.
- [x] Confirm `http://localhost:8501` returns HTTP 200.
- [x] If Streamlit is not running, start it on port 8501 with logs under `.runtime/`.
- [x] Restart stale Streamlit runtime after module-cache import error and confirm `http://localhost:8501/_stcore/health` returns `ok`.
