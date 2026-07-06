# Memory/Semi Rotation Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative memory/semiconductor rotation overlay that fixes core data coverage, flags extended leaders, and identifies supported second-line candidates without displaying probabilities.

**Architecture:** Keep the existing candidate path intact, then add a pure overlay layer after quality-gated catalyst rerank. The refresh and metadata layers make the core memory/semi basket visible and fresh; the overlay only adds explanatory fields and small ordering adjustments.

**Tech Stack:** Python stdlib, pytest, FastAPI serializers already in `api/serializers.py`, HTR candidate modules under `src/hot_theme_rotator/candidate_engine/`.

---

## File Structure

- Modify: `tools/refresh_htr_price_db.py` - add priority symbol support to price refresh.
- Modify: `src/hot_theme_rotator/data/ticker_metadata.py` - add deterministic theme overrides.
- Create: `src/hot_theme_rotator/candidate_engine/theme_rotation.py` - pure overlay calculations.
- Modify: `src/hot_theme_rotator/candidate_engine/hybrid_rerank.py` - apply small rotation score after existing catalyst/quality score.
- Modify: `api/serializers.py` - expose overlay fields and core coverage metadata.
- Test: `tests/unit/test_refresh_htr_price_db.py`
- Test: `tests/unit/test_ticker_metadata.py`
- Test: `tests/unit/test_theme_rotation.py`
- Test: `tests/unit/test_hybrid_rerank.py`
- Test: `tests/unit/test_api_symbol.py` or `tests/unit/test_api_dashboard.py` depending on existing serializer coverage.
- Update: `docs/02_GOVERNANCE.md`
- Update: `docs/06_MODEL_FACTORS.md`

---

### Task 1: Priority Memory/Semi Price Coverage

**Files:**
- Modify: `tools/refresh_htr_price_db.py`
- Test: `tests/unit/test_refresh_htr_price_db.py`

- [ ] **Step 1: Write the failing test**

Add a test that creates a DB where `285A.T` is absent from the active universe, calls `refresh(..., priority_symbols=["285A.T"])`, and asserts the injected fetch receives `285A.T`.

```python
def test_refresh_includes_priority_symbols_not_in_active_universe(tmp_path):
    db = _make_price_db(tmp_path, symbols=["1306.T"], date="2026-06-18")
    seen = {}

    def fetch(symbols, days):
        seen["symbols"] = symbols
        return [("285A.T", "2026-06-19", 100.0, 105.0, 99.0, 104.0, 1000.0)]

    result = refresh(
        htr_db=db,
        sibling_db=db,
        target_date="2026-06-19",
        fetch=fetch,
        priority_symbols=["285A.T"],
    )

    assert "285A.T" in seen["symbols"]
    assert result["appended"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_refresh_htr_price_db.py::test_refresh_includes_priority_symbols_not_in_active_universe -q --basetemp=.runtime/pytest-theme-rotation`

Expected: FAIL because `refresh()` does not accept `priority_symbols`.

- [ ] **Step 3: Implement priority symbols**

Add `MEMORY_SEMI_PRIORITY_SYMBOLS` and a `priority_symbols` argument. Merge active universe plus priority symbols before fetch.

- [ ] **Step 4: Run targeted test**

Run: `python -m pytest tests/unit/test_refresh_htr_price_db.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: PASS.

---

### Task 2: Deterministic Theme Overrides

**Files:**
- Modify: `src/hot_theme_rotator/data/ticker_metadata.py`
- Test: `tests/unit/test_ticker_metadata.py`

- [ ] **Step 1: Write failing tests**

Add tests that `themes_for_symbol("285A.T", store)` includes `memory`, `semi`, and `ai` even when yfinance metadata is missing or generic.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ticker_metadata.py::test_symbol_overrides_cover_memory_semi_core -q --basetemp=.runtime/pytest-theme-rotation`

Expected: FAIL because no override exists.

- [ ] **Step 3: Implement overrides**

Add `SYMBOL_THEME_OVERRIDES` and merge overrides in `themes_for_symbol()` and `fetch_info_one()` output.

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/unit/test_ticker_metadata.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: PASS.

---

### Task 3: Pure Theme Rotation Overlay

**Files:**
- Create: `src/hot_theme_rotator/candidate_engine/theme_rotation.py`
- Test: `tests/unit/test_theme_rotation.py`

- [ ] **Step 1: Write failing tests**

Cover:

- `classify_theme_regime({"memory": 10, "semi": 8, "bank": 2}) == "memory_semi_hot"`
- a leader with `ret20=0.30`, `ret60=0.60`, `dist_high20=-0.01`, and volume expansion is `leaderExtended=True`, `chaseRisk="study_only"`.
- a non-leader semi candidate with fresh data and two supporting facts is `secondLineCandidate=True`.
- sector membership alone does not create a second-line candidate.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_theme_rotation.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement pure functions**

Implement:

- `classify_theme_regime(theme_counts: dict[str, int]) -> str`
- `assess_core_theme_coverage(price_dates: dict[str, str], latest_trade_date: str) -> dict`
- `annotate_rotation(candidates, theme_counts, price_features, catalyst_map=None) -> list[dict]`

- [ ] **Step 4: Run targeted test**

Run: `python -m pytest tests/unit/test_theme_rotation.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: PASS.

---

### Task 4: Rerank Integration

**Files:**
- Modify: `src/hot_theme_rotator/candidate_engine/hybrid_rerank.py`
- Test: `tests/unit/test_hybrid_rerank.py`

- [ ] **Step 1: Write failing tests**

Add tests that an extended sector-only memory leader receives an extra ordering penalty and a supported second-line candidate receives a small boost.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_hybrid_rerank.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: FAIL because rerank ignores rotation fields.

- [ ] **Step 3: Implement config and score adjustment**

Add config fields:

- `extended_leader_penalty: float = 0.15`
- `second_line_boost: float = 0.08`
- `stale_core_theme_penalty: float = 0.10`

Apply them after the existing quality penalty, then expose `rotation_score` and `rotation_reasons`.

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/unit/test_hybrid_rerank.py tests/unit/test_theme_rotation.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: PASS.

---

### Task 5: API/Serializer Exposure

**Files:**
- Modify: `api/serializers.py`
- Test: `tests/unit/test_api_dashboard.py`

- [ ] **Step 1: Write failing API contract test**

Assert candidates expose `themeRegime`, `leaderExtended`, `chaseRisk`, `secondLineCandidate`, and `rotationReasons`, and dashboard meta exposes `dataQuality.coreThemeCoverage`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_api_dashboard.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: FAIL because the fields are absent.

- [ ] **Step 3: Wire serializer fields**

Map snake_case internal fields to existing API camelCase style:

- `theme_regime` -> `themeRegime`
- `leader_extended` -> `leaderExtended`
- `chase_risk` -> `chaseRisk`
- `second_line_candidate` -> `secondLineCandidate`
- `rotation_reasons` -> `rotationReasons`

- [ ] **Step 4: Run targeted API tests**

Run: `python -m pytest tests/unit/test_api_dashboard.py tests/unit/test_api_symbol.py -q --basetemp=.runtime/pytest-theme-rotation`

Expected: PASS.

---

### Task 6: Verification and Status

**Files:**
- Update: `PROJECT_STATUS.md`

- [ ] **Step 1: Run focused unit suite**

Run:

```powershell
python -m pytest tests/unit/test_refresh_htr_price_db.py tests/unit/test_ticker_metadata.py tests/unit/test_theme_rotation.py tests/unit/test_hybrid_rerank.py tests/unit/test_api_dashboard.py -q --basetemp=.runtime/pytest-theme-rotation
```

Expected: PASS.

- [ ] **Step 2: Run full unit smoke**

Run:

```powershell
python -m pytest tests/unit -q --basetemp=.runtime/pytest-unit-theme-rotation
```

Expected: PASS.

- [ ] **Step 3: Update status**

Add a concise `PROJECT_STATUS.md` entry with:

- Rule 11.14 added
- priority memory/semi coverage added
- overlay fields implemented
- test counts and any residual risk

---

## Self-Review

- Spec coverage: data coverage, metadata overrides, overlay fields, rerank integration, API exposure, and honesty rules all have implementation tasks.
- Placeholder scan: no TBD/TODO/later placeholders remain.
- Type consistency: plan uses snake_case internally and camelCase at the API boundary.
