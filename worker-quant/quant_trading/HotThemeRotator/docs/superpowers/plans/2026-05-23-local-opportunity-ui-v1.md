# Local Opportunity UI V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit interface so a general user can open the realtime opportunity candidate panel without using Python snippets or reading raw Markdown.

**Architecture:** Keep UI data preparation in `src/hot_theme_rotator/ui/opportunity_dashboard.py` so it can be tested without Streamlit. Put the Streamlit page in `tools/streamlit_opportunity_app.py`, using existing P8 scanner, adapter, ladder, and panel modules.

**Tech Stack:** Python, Streamlit, pandas, pytest.

---

### Task 1: Dashboard Data Layer

**Files:**
- Create: `src/hot_theme_rotator/ui/__init__.py`
- Create: `src/hot_theme_rotator/ui/opportunity_dashboard.py`
- Create: `tests/unit/test_opportunity_dashboard.py`

- [x] Write failing tests for symbol parsing, sample panel rows, refresh interval label, and user-facing table records.
- [x] Implement pure functions that return Streamlit-ready records and Markdown.
- [x] Run targeted tests.

### Task 2: Streamlit App

**Files:**
- Create: `tools/streamlit_opportunity_app.py`
- Modify: `README.md`

- [x] Add a wide-layout Streamlit dashboard with sidebar controls.
- [x] Support sample mode and yfinance quote-only mode.
- [x] Display ranked table, selected-row price ladder, rules/status, and raw Markdown.
- [x] Add README launch command.

### Task 3: Project Tracking And Verification

**Files:**
- Modify: `docs/01_TASKS.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `.gitignore`

- [x] Add P8-06 Local User Interface V1.
- [x] Run targeted tests.
- [x] Run the full suite.
- [x] Start Streamlit locally and record the URL.
- [x] Keep Streamlit runtime logs outside pytest temp directories.
