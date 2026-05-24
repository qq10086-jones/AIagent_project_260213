# Realtime Opportunity V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable version of the user's target system: search potential stocks from normalized inputs, compute a research-only opportunity score, and output staged buy/sell price ladders.

**Architecture:** Add a focused `opportunity` package for candidate scanning and price ladder generation. Keep data-source access outside V1; this version consumes normalized `PriceBar`, `NewsItem`, and feature objects so later real-time adapters can plug in without changing the decision rules.

**Tech Stack:** Python dataclasses, existing HotThemeRotator schemas, pytest.

---

### Task 1: Rules And Route

**Files:**
- Modify: `docs/02_GOVERNANCE.md`
- Modify: `docs/01_TASKS.md`
- Modify: `PROJECT_STATUS.md`

- [x] Add P8 rules: potential-stock search, real-time point-in-time inputs, staged entry/exit prices, uncalibrated score labels, and advice-only output.
- [x] Add P8 tasks: opportunity scanner V1, real-time adapters, ladder calibration, feedback.
- [x] Update status so next work is P8, not P7.

### Task 2: Opportunity Scanner V1

**Files:**
- Create: `src/hot_theme_rotator/opportunity/__init__.py`
- Create: `src/hot_theme_rotator/opportunity/opportunity_scanner.py`
- Create: `tests/unit/test_opportunity_scanner.py`

- [x] Write failing tests for ranked candidate output.
- [x] Implement normalized inputs, weighted scoring, reason codes, and fail-closed behavior for invalid prices.
- [x] Run targeted tests.

### Task 3: Price Ladder V1

**Files:**
- Create: `src/hot_theme_rotator/opportunity/price_ladder.py`
- Create: `tests/unit/test_price_ladder.py`

- [x] Write failing tests for aggressive, balanced, and conservative entries plus stop and three exits.
- [x] Implement deterministic ATR/range-based ladder generation.
- [x] Ensure every ladder output is research-only and not a trading order.

### Task 4: Candidate Panel Report V1

**Files:**
- Create: `src/hot_theme_rotator/reporting/realtime_opportunity_panel.py`
- Create: `tests/unit/test_realtime_opportunity_panel.py`

- [x] Write failing tests for the user-facing table shape.
- [x] Render ranked candidates with trigger theme, score label, staged buy prices, stop, staged sell prices, and reasons.
- [x] Mark all scores as `uncalibrated_research_score` until feedback calibration exists.

### Task 5: Verification

**Files:**
- Modify: `docs/01_TASKS.md`
- Modify: `PROJECT_STATUS.md`

- [x] Run targeted tests for P8 modules.
- [x] Run the full suite with `python -m pytest .\tests -q -o cache_dir=.\pytest_cache --basetemp=.\pytest_tmp`.
- [x] Record verification evidence in tasks and project status.
