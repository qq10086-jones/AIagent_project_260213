# Quality-Gated Catalyst Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent broad sector theme heat from being presented or ranked like company-specific stock catalysts.

**Architecture:** Extend the existing ADR-0009 catalyst map with evidence tiers, keep the reranker deterministic, and keep the serializer as the boundary that exposes dashboard fields. Reuse the existing screener snapshot adapter, but make dashboard and profile share the freshest snapshot path.

**Tech Stack:** Python 3.13, FastAPI endpoint modules, pytest unit tests, Markdown governance docs.

---

### Task 1: Catalyst Evidence Tests

**Files:**
- Modify: `tests/unit/test_catalyst.py`
- Modify: `tests/unit/test_hybrid_rerank.py`
- Modify: `tests/unit/test_theme_leaders.py`

- [ ] **Step 1: Add tests before production code**

Add tests proving sector-only evidence is separate from company evidence, and that only company evidence can set `news_catalyzed`.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/unit/test_catalyst.py tests/unit/test_hybrid_rerank.py tests/unit/test_theme_leaders.py -q`

Expected: failures referencing missing `evidence_level`, `company_catalyzed`, or unchanged leader behavior.

- [ ] **Step 3: Implement catalyst evidence and rerank behavior**

Modify `src/hot_theme_rotator/candidate_engine/catalyst.py`,
`src/hot_theme_rotator/candidate_engine/hybrid_rerank.py`, and
`src/hot_theme_rotator/candidate_engine/theme_leaders.py`.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/unit/test_catalyst.py tests/unit/test_hybrid_rerank.py tests/unit/test_theme_leaders.py -q`

Expected: all selected tests pass.

### Task 2: Serializer and Profile Consistency

**Files:**
- Modify: `api/serializers.py`
- Modify: `api/symbol.py`
- Modify: `tests/unit/test_api_symbol.py`

- [ ] **Step 1: Add failing profile-source test**

Add a test proving `_screener_row_for()` reads the HTR freshest snapshot when it is newer than the sibling file.

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/unit/test_api_symbol.py -q`

Expected: the new test fails because profile reads only `default_selected_tickers_path()`.

- [ ] **Step 3: Share freshest snapshot logic**

Move or duplicate the freshest HTR snapshot selection into a public helper in
`hot_theme_rotator.data.universe_adapter`, then use it from both `api/serializers.py`
and `api/symbol.py`.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/unit/test_api_symbol.py tests/unit/test_api_dashboard.py -q`

Expected: profile and dashboard tests pass.

### Task 3: Governance and Factor Docs

**Files:**
- Modify: `docs/02_GOVERNANCE.md`
- Modify: `docs/06_MODEL_FACTORS.md`

- [ ] **Step 1: Update Rule 11.12**

Add the company-vs-sector evidence rule and forbid sector-only labels from rendering as news-catalyst or theme-leader badges.

- [ ] **Step 2: Update model factor reference**

Document the new rerank weights and explain that sector evidence is only a weak exposure nudge.

### Task 4: Verification

**Files:**
- No additional source files.

- [ ] **Step 1: Run focused unit tests**

Run: `python -m pytest tests/unit/test_catalyst.py tests/unit/test_hybrid_rerank.py tests/unit/test_theme_leaders.py tests/unit/test_api_symbol.py tests/unit/test_api_dashboard.py -q`

- [ ] **Step 2: Run broader smoke if focused tests pass**

Run: `python -m pytest tests/unit -q`

- [ ] **Step 3: Inspect live API snapshot**

With the service running, call `/api/dashboard` and verify that sector-only rows expose `catalystEvidenceLevel="sector"` and do not set `newsCatalyzed=true`.
