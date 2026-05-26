# Time-To-First-Value Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next 10-15 day local-only batch that turns HotThemeRotator into a pull-first daily advisory system with hardened delayed price data and silent watchlist intelligence.

**Architecture:** Add a small source-access policy layer before scraper parsers, then feed structured health state into a pull-only cockpit payload. Silent watchlist events and Anti-FOMO guards are modeled as pure domain data first, with no notification channel enabled.

**Tech Stack:** Python dataclasses, pytest, FastAPI GET-only APIs, existing `PriceQuote` / `PriceSourceHealth` contracts, local JSON reports under `reports/`.

---

### Task 1: P10-19 HTTP Policy Layer

**Files:**
- Create: `src/hot_theme_rotator/data/external/realtime_price/http_policy.py`
- Test: `tests/unit/test_realtime_price_http_policy.py`
- Update: `docs/01_TASKS.md`
- Update: `PROJECT_STATUS.md`

- [ ] **Step 1: Write failing tests**

```python
from hot_theme_rotator.data.external.realtime_price.http_policy import (
    CloudflareBlockError,
    FixedRobotsPolicy,
    HttpFetchPolicy,
    RobotsBlockedError,
)


def test_policy_applies_rate_limit_between_same_host_requests():
    sleeps = []
    ticks = iter([100.0, 102.0, 102.0])
    policy = HttpFetchPolicy(
        min_interval_seconds=10.0,
        monotonic=lambda: next(ticks),
        sleep=sleeps.append,
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA1",),
    )

    policy.prepare_request("https://finance.yahoo.co.jp/quote/6779.T")
    policy.prepare_request("https://finance.yahoo.co.jp/quote/1306.T")

    assert sleeps == [8.0]


def test_policy_rotates_user_agents():
    policy = HttpFetchPolicy(
        robots_policy=FixedRobotsPolicy(allowed=True),
        user_agents=("UA1", "UA2"),
    )

    first = policy.prepare_request("https://finance.yahoo.co.jp/quote/6779.T")
    second = policy.prepare_request("https://kabutan.jp/stock/?code=6779")

    assert first.headers["User-Agent"] == "UA1"
    assert second.headers["User-Agent"] == "UA2"


def test_policy_blocks_disallowed_robots_url():
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=False))

    try:
        policy.prepare_request("https://example.com/private")
    except RobotsBlockedError as exc:
        assert "robots.txt blocked" in str(exc)
    else:
        raise AssertionError("expected RobotsBlockedError")


def test_policy_detects_cloudflare_html():
    policy = HttpFetchPolicy(robots_policy=FixedRobotsPolicy(allowed=True))

    try:
        policy.validate_response_text(
            "<html><title>Just a moment...</title><script>window._cf_chl_opt={}</script>"
        )
    except CloudflareBlockError as exc:
        assert "cloudflare" in str(exc).lower()
    else:
        raise AssertionError("expected CloudflareBlockError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_realtime_price_http_policy.py -q -o cache_dir=.runtime\pytest_cache_http_policy_red --basetemp=.runtime\pytest_tmp\http_policy_red`

Expected: FAIL because `http_policy` does not exist.

- [ ] **Step 3: Implement minimal policy module**

Create `HttpFetchPolicy`, `PreparedHttpRequest`, `FixedRobotsPolicy`, `RobotsBlockedError`, and `CloudflareBlockError`. Keep all dependencies injectable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_realtime_price_http_policy.py -q -o cache_dir=.runtime\pytest_cache_http_policy_green --basetemp=.runtime\pytest_tmp\http_policy_green`

Expected: `4 passed`.

### Task 2: P10-19 Mock HTTP Scraper Integration

**Files:**
- Update: `src/hot_theme_rotator/data/external/realtime_price/yahoo_japan_scraper.py`
- Update: `src/hot_theme_rotator/data/external/realtime_price/kabutan_scraper.py`
- Test: `tests/integration/test_realtime_price_network.py`

- [ ] **Step 1: Write failing tests**

Add tests proving Yahoo/Kabutan fetch functions call `HttpFetchPolicy.prepare_request`, reject Cloudflare HTML before parsing, and return `PriceQuote` from injected HTTP text.

- [ ] **Step 2: Run targeted integration test and verify RED**

Run: `python -m pytest tests/integration/test_realtime_price_network.py -q`

Expected: FAIL because fetch functions do not exist.

- [ ] **Step 3: Implement fetch wrappers**

Add `fetch_yahoo_japan_quote(symbol, http_get_text, policy, wall_ts=None)` and `fetch_kabutan_quote(symbol, http_get_text, policy, wall_ts=None)`.

- [ ] **Step 4: Run targeted integration test and related price tests**

Run targeted integration plus existing Yahoo/Kabutan/parser/orchestrator tests.

### Task 3: P10-20 Cockpit Payload Contract

**Files:**
- Create: `src/hot_theme_rotator/reporting/daily_advisory_cockpit.py`
- Update: `api/serializers.py`
- Update: `api/dashboard.py`
- Test: `tests/unit/test_daily_advisory_cockpit.py`
- Test: `tests/unit/test_dashboard_api_contracts.py`

- [ ] **Step 1: Write failing payload tests**

Tests must assert quote freshness fields, research-only status, no win-rate wording, and no notification side effects.

- [ ] **Step 2: Implement payload builder**

Build a pure Python payload first; API/frontend integration only consumes it.

- [ ] **Step 3: Verify GET-only route behavior**

Run API route tests and existing frontend contract tests.

### Task 4: P10-17 Silent Watchlist Queue

**Files:**
- Create: `src/hot_theme_rotator/watchlist_intelligence/__init__.py`
- Create: `src/hot_theme_rotator/watchlist_intelligence/silent_queue.py`
- Test: `tests/unit/test_silent_watchlist_queue.py`

- [ ] **Step 1: Write failing tests**

Cover TDnet disclosure event, quote unavailable event, ladder proximity event, and no notifier invocation.

- [ ] **Step 2: Implement queue records and writer/reader**

Persist to `reports/observability/silent_queue/{trade_date}.jsonl` with ISO trade-date validation.

- [ ] **Step 3: Verify cockpit can summarize queue counts**

Add cockpit test for suppressed/study-only/silent counts.

### Task 5: P10-18 Anti-FOMO Core Guards

**Files:**
- Create: `src/hot_theme_rotator/alerts/discipline.py`
- Test: `tests/unit/test_alert_discipline.py`
- Create: `configs/push_discipline.yaml`

- [ ] **Step 1: Write failing guard tests**

Cover daily budget, stale fail-closed, chase downgrade to study-only, and watchlist cooling-off.

- [ ] **Step 2: Implement pure guard evaluator**

Return guarded decisions only. Do not add notifier channels.

- [ ] **Step 3: Run alert and cockpit related tests**

Verify no broker/order/notifier path is introduced.

### Task 6: Documentation And Full Verification

**Files:**
- Update: `docs/01_TASKS.md`
- Update: `docs/03_FOLDER_MAP.md`
- Update: `PROJECT_STATUS.md`

- [ ] **Step 1: Update task statuses and evidence**

Record exactly which tasks are complete, in progress, or pending.

- [ ] **Step 2: Run related and full verification**

Run targeted suites after each task and full `python -m pytest tests -q` before final status.

- [ ] **Step 3: Check local-only constraint**

Confirm no upload, push, PR, or remote sync was performed.
