# P8-05 Ladder Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opportunity-ladder feedback evaluator that summarizes seven-tier touch evidence from existing decision-log predictions and outcome records.

**Architecture:** Reuse `PredictionRecord`, `OutcomeRecord`, and `build_calibration_report`; add a focused `calibration.ladder_feedback` module for ladder-tier summaries. Keep the first cycle backend-only and research-only.

**Tech Stack:** Python dataclasses, pytest, existing JSONL decision-log schemas, existing calibration reporter.

---

### Task 1: Document P8-05 Boundaries

**Files:**
- Modify: `docs/02_GOVERNANCE.md`
- Modify: `docs/00_DESIGN.md`
- Modify: `docs/01_TASKS.md`

- [x] Add Rule 9.7 clarifying that ladder feedback consumes existing P9 logs/outcomes and cannot be called a win rate.
- [x] Add DESIGN §6.10 note for `calibration.ladder_feedback`.
- [x] Rewrite P8-05 acceptance so it requires no new storage and no execution path.

### Task 2: Ladder Feedback Tests

**Files:**
- Create: `tests/unit/test_ladder_feedback.py`

- [x] Write tests for insufficient tier samples.
- [x] Write tests for calibrated tier touch rate when sample count reaches threshold.
- [x] Write tests that non-complete outcomes are skipped.
- [x] Write fail-closed tests for missing tier payloads.

### Task 3: Ladder Feedback Module

**Files:**
- Create: `src/hot_theme_rotator/calibration/ladder_feedback.py`
- Modify: `src/hot_theme_rotator/calibration/__init__.py`

- [x] Implement `LadderTierFeedback`.
- [x] Implement `LadderFeedbackReport`.
- [x] Implement `build_ladder_feedback_report`.
- [x] Export the new dataclasses and builder.

### Task 4: Verification And Status

**Files:**
- Modify: `PROJECT_STATUS.md`

- [x] Run targeted ladder feedback tests.
- [x] Run API/calibration related tests.
- [x] Run full suite with `NUMBA_CACHE_DIR=.runtime\numba_cache`.
- [x] Record evidence in `PROJECT_STATUS.md`.
