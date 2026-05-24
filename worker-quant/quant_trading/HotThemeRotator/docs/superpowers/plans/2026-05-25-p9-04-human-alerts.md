# P9-04 Human Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a research-only human alert data layer for staged ladder level crossings.

**Architecture:** Implement a focused `alerts` package with pure alert generation and in-memory throttling. Keep output as local records only; no sending, no order objects, no broker or paper-trading side effects.

**Tech Stack:** Python dataclasses, hashlib deterministic ids, pytest.

---

### Task 1: Document P9-04 Boundaries

**Files:**
- Modify: `docs/02_GOVERNANCE.md`
- Modify: `docs/00_DESIGN.md`
- Modify: `docs/01_TASKS.md`

- [x] Add alert-specific advice-only constraints.
- [x] Add DESIGN module responsibility for `alerts/`.
- [x] Expand P9-04 acceptance into testable alert contracts.

### Task 2: Alert Tests

**Files:**
- Create: `tests/unit/test_human_alerts.py`

- [x] Test entry and stop levels trigger when price is at or below threshold.
- [x] Test exit levels trigger when price is at or above threshold.
- [x] Test duplicate throttle suppresses repeated symbol/level/trade-date alerts.
- [x] Test alert records expose research-only fields and no order fields.
- [x] Test invalid prices fail closed.

### Task 3: Alert Module

**Files:**
- Create: `src/hot_theme_rotator/alerts/__init__.py`
- Create: `src/hot_theme_rotator/alerts/human_alerts.py`

- [x] Implement `AlertRecord`.
- [x] Implement deterministic `compute_alert_id`.
- [x] Implement `AlertThrottle`.
- [x] Implement `build_ladder_alerts`.

### Task 4: Verification And Status

**Files:**
- Modify: `PROJECT_STATUS.md`

- [x] Run targeted alert tests.
- [x] Run API/opportunity related tests.
- [x] Run full suite with `NUMBA_CACHE_DIR=.runtime\numba_cache`.
- [x] Record evidence in `PROJECT_STATUS.md`.
