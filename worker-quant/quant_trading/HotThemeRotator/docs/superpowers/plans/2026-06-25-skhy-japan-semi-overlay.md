# SKHY Japan Semi Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Japan-first semiconductor attention overlay that uses the SK hynix ADR event as an external catalyst input, while keeping the system's execution market, candidate calibration, and advice boundary focused on Japanese equities. The system may also record the operator's external SKHY ADR position manually, but it must not route orders or mix ADR returns into JP candidate calibration.

**External fact boundary as of 2026-06-25:** SEC EDGAR shows SK hynix Inc. filed a Form F-1 registration statement on 2026-06-24, accession `0001193125-26-280172`, SIC `3674 Semiconductors & Related Devices`. The filing package includes a Form F-1 for ADSs. Treat `SKHY` as a pending/active external instrument only after a fresh listing-status check. Source: https://www.sec.gov/Archives/edgar/data/2120882/000119312526280172/0001193125-26-280172-index.htm

**Architecture:** Keep the existing HTR candidate path authoritative:

`JP price/news/metadata -> catalyst rerank -> memory/semi rotation overlay -> dashboard`

Add a separate read-only ADR watch lane:

`SKHY / 000660.KS / MU / NVDA / SOX / USDJPY snapshots -> SKHY event state -> small Japan semi sympathy annotation`

The ADR lane may annotate and lightly reorder Japanese semi candidates after existing rerank logic. It must not display probabilities, win rates, expected return, "guaranteed chance", or direct buy commands.

**Why this design has evidence:** Momentum and cross-asset lead-lag can exist, but they are noisy and decay. Investor attention creates buying pressure and overreaction risk. Therefore the system should measure attention and relative strength, then force forward evidence before promotion.

- Momentum / trend evidence: Jegadeesh and Titman (1993), Asness, Moskowitz, and Pedersen (2013).
- Investor attention and psychology: Barber and Odean (2008), Kahneman and Tversky (1979).
- News freshness and stale-news decay: Tetlock (2011).
- Multiple-testing and overfit discipline: Bailey and Lopez de Prado (Deflated Sharpe), Harvey, Liu, and Zhu (2016).
- Position sizing discipline: Kelly (1956), but only after calibrated edge exists. Until then, sizing remains operator-controlled and outside HTR advice.

**Basic math / physics framing:** Treat SKHY as an impulse input, not a forecast. Measure amplitude, freshness, decay, and confirmation. Use half-life decay for catalyst freshness, cross-sectional Rank-IC for skill, and explicit cost hurdles from Rule 16.0. Avoid "story gravity": a large narrative is not edge unless it produces repeatable, net-of-cost forward evidence.

**Tech Stack:** Python stdlib, pytest, current FastAPI serializers, current candidate modules under `src/hot_theme_rotator/candidate_engine/`, current daily routine tooling.

---

## File Structure

- Modify: `tests/unit/test_daily_routine.py` - sync expected smoke order with the existing S-kabu overlay step.
- Create: `src/hot_theme_rotator/data/external/adr_watch.py` - pure snapshot schema and stale/status helpers.
- Create: `tests/unit/test_adr_watch.py`
- Create: `tools/refresh_skhy_adr_watch.py` - write ADR watch snapshots to `reports/adr/`.
- Create: `tests/unit/test_refresh_skhy_adr_watch.py`
- Create: `src/hot_theme_rotator/candidate_engine/skhy_overlay.py` - pure event and sympathy overlay functions.
- Create: `tests/unit/test_skhy_overlay.py`
- Modify: `src/hot_theme_rotator/candidate_engine/theme_rotation.py` or `hybrid_rerank.py` - apply only a small post-rerank annotation/adjustment.
- Modify: `api/serializers.py` - expose read-only ADR event state and candidate annotations.
- Modify: `tests/unit/test_api_dashboard.py`
- Optional create: `src/hot_theme_rotator/user_state/external_adr_journal.py` - manual SKHY ADR fills separate from JP portfolio calibration.
- Optional test: `tests/unit/test_external_adr_journal.py`
- Create: `src/hot_theme_rotator/reporting/skhy_event_review.py` - forward event review once samples exist.
- Create: `tests/unit/test_skhy_event_review.py`
- Update: `docs/02_GOVERNANCE.md`
- Update: `docs/01_TASKS.md`
- Update: `PROJECT_STATUS.md`

---

### Task 1: Fix Daily-Routine Smoke Expectation

**Files:**
- Modify: `tests/unit/test_daily_routine.py`

- [ ] **Step 1: Run the focused failing smoke contract**

Run:

```powershell
python -m pytest tests/unit/test_daily_routine.py -q --basetemp=.runtime/pytest-p20-daily-routine -p no:cacheprovider
```

Expected now: a stale expectation may fail because `tools/build_s_kabu_overlay.py` was inserted into the daily routine.

- [ ] **Step 2: Update expected call order only**

Keep the production routine unchanged unless the test reveals a real code defect. The expected order should include the S-kabu overlay after metadata refresh and before emit/sweep.

Expected logical order:

```text
refresh_htr_price_db -> refresh_htr_news -> refresh_htr_macro_news -> screener -> refresh_ticker_metadata -> build_s_kabu_overlay -> emit_daily_predictions -> sweep_pending_outcomes
```

- [ ] **Step 3: Verify**

Run:

```powershell
python -m pytest tests/unit/test_daily_routine.py -q --basetemp=.runtime/pytest-p20-daily-routine -p no:cacheprovider
```

Expected: PASS.

---

### Task 2: ADR Watch Snapshot Schema

**Files:**
- Create: `src/hot_theme_rotator/data/external/adr_watch.py`
- Create: `tests/unit/test_adr_watch.py`

- [ ] **Step 1: Write failing tests**

Cover:

- valid symbols: `SKHY`, `000660.KS`, `MU`, `NVDA`, `SOXX` or `^SOX`, `USDJPY=X`;
- status enum: `pending_listing`, `active`, `stale`, `unavailable`;
- stale detection using `data_ts` and `asof`;
- no forbidden fields: `probability`, `win_rate`, `expected_return`, `edge`;
- JSON round trip preserves `source`, `currency`, `reasons`, and `status`.

- [ ] **Step 2: Implement pure dataclasses/helpers**

Suggested public API:

```python
ALLOWED_ADR_STATUSES = {"pending_listing", "active", "stale", "unavailable"}

@dataclass(frozen=True)
class AdrInstrumentSnapshot:
    symbol: str
    role: str
    asof: str
    data_ts: str | None
    status: str
    last_price: float | None
    prev_close: float | None
    overnight_return: float | None
    volume: float | None
    volume_z: float | None
    currency: str
    source: str
    stale: bool
    reasons: tuple[str, ...] = ()
```

Keep this module deterministic and dependency-free.

- [ ] **Step 3: Verify**

Run:

```powershell
python -m pytest tests/unit/test_adr_watch.py -q --basetemp=.runtime/pytest-p20-adr-watch -p no:cacheprovider
```

Expected: PASS.

---

### Task 3: SKHY ADR Watch Refresh Tool

**Files:**
- Create: `tools/refresh_skhy_adr_watch.py`
- Create: `tests/unit/test_refresh_skhy_adr_watch.py`

- [ ] **Step 1: Write failing tests with injected fetcher**

Do not hit live network in unit tests. Tests should assert:

- `SKHY` unavailable before a real quote becomes `pending_listing`, not a hard failure;
- fresh `SKHY` quote becomes `active`;
- missing source or old timestamp becomes `stale` or `unavailable`;
- output path is `reports/adr/adr_watch_{asof}.json`;
- output contains no probability/win-rate/expected-return fields.

- [ ] **Step 2: Implement CLI**

Suggested CLI:

```powershell
python tools/refresh_skhy_adr_watch.py --asof 2026-06-25 --out-dir reports/adr
```

The output should be append-friendly and human-readable:

```json
{
  "asof": "2026-06-25",
  "source": "yfinance_or_configured_fetcher",
  "listingStatusCheckedAt": "...",
  "instruments": {
    "SKHY": {"status": "pending_listing", "...": "..."},
    "000660.KS": {"status": "active", "...": "..."},
    "MU": {"status": "active", "...": "..."}
  }
}
```

If `SKHY` is not yet tradable, that is useful information. Do not silently substitute another symbol and pretend it is SKHY.

- [ ] **Step 3: Wire daily routine as non-fatal**

Add the refresh after JP price/news refresh and before candidate serialization. ADR failure must not block Japan candidate generation.

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests/unit/test_refresh_skhy_adr_watch.py tests/unit/test_daily_routine.py -q --basetemp=.runtime/pytest-p20-adr-refresh -p no:cacheprovider
```

Expected: PASS.

---

### Task 4: SKHY Event State and Japan Semi Sympathy Overlay

**Files:**
- Create: `src/hot_theme_rotator/candidate_engine/skhy_overlay.py`
- Create: `tests/unit/test_skhy_overlay.py`
- Modify: `src/hot_theme_rotator/candidate_engine/theme_rotation.py` or `src/hot_theme_rotator/candidate_engine/hybrid_rerank.py`

- [ ] **Step 1: Write failing tests for event state**

Test cases:

- stale or pending SKHY snapshot produces `skhyCatalystActive=false`;
- active SKHY with strong move but no confirmation is "watch", not a boost;
- active SKHY plus memory peers or SOX confirmation becomes `skhyCatalystActive=true`;
- the event has freshness decay;
- no candidate gets a sympathy label from sector membership alone.

- [ ] **Step 2: Write failing tests for Japan candidate annotation**

Use fake Japanese candidates:

- Kioxia `285A.T`: memory leader/reference, already extended if 20d/60d move is extreme;
- Tokyo Electron `8035.T`, Advantest `6857.T`, Screen `7735.T`, Disco `6146.T`: semi equipment;
- smaller semi-adjacent names from current metadata: require company/theme support before label.

Assert candidate fields:

```text
skhyCatalystStatus
skhyCatalystActive
skhyOvernightMove
semiSympathyScore
semiSympathyReasons
relativeStrengthVsSkhy
```

Forbidden fields:

```text
skhyWinRate
skhyProbability
expectedReturn
guaranteed
```

- [ ] **Step 3: Implement small, explicit overlay**

Suggested constraints:

- run after the existing catalyst and rotation overlays;
- cap score impact tightly, for example `[-0.05, +0.07]` in normalized rerank space until evidence exists;
- require at least two facts for a positive Japan semi sympathy label:
  - fresh SKHY/000660.KS/peer data;
  - SKHY or 000660.KS abnormal move or volume;
  - SOX/MU/NVDA confirmation;
  - candidate is already in memory/semi/AI hardware theme metadata;
  - candidate relative strength is not collapsing;
  - candidate is not flagged as extended-chase by Rule 11.14.

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests/unit/test_skhy_overlay.py tests/unit/test_theme_rotation.py tests/unit/test_hybrid_rerank.py -q --basetemp=.runtime/pytest-p20-skhy-overlay -p no:cacheprovider
```

Expected: PASS.

---

### Task 5: Dashboard/API Read-Only Surface

**Files:**
- Modify: `api/serializers.py`
- Modify: `tests/unit/test_api_dashboard.py`
- Frontend optional after backend is stable.

- [ ] **Step 1: Write failing serializer test**

Assert `/api/dashboard` style payload includes:

```json
{
  "meta": {
    "dataQuality": {
      "adrWatch": {
        "asof": "2026-06-25",
        "status": "active_or_pending_or_stale",
        "stale": false
      }
    }
  },
  "eventDesk": {
    "skhy": {
      "status": "watch",
      "disclosure": "External ADR catalyst; not a JP buy signal; no probability."
    }
  }
}
```

Candidate fields may include SKHY annotations, but never direct advice language.

- [ ] **Step 2: Implement serializer fail-open behavior**

If no ADR snapshot exists, surface `adrWatch.status="unavailable"` and leave candidate ranking unchanged.

- [ ] **Step 3: Verify**

Run:

```powershell
python -m pytest tests/unit/test_api_dashboard.py -q --basetemp=.runtime/pytest-p20-api -p no:cacheprovider
```

Expected: PASS.

---

### Task 6: Manual External ADR Position Lane

**Files:**
- Optional create: `src/hot_theme_rotator/user_state/external_adr_journal.py`
- Optional create: `tests/unit/test_external_adr_journal.py`

- [ ] **Step 1: Decide if the operator wants HTR to record SKHY fills**

If yes, implement a manual-only journal. If no, skip this task.

- [ ] **Step 2: Write failing tests**

Tests must prove:

- only manual already-completed fills can be recorded;
- no broker/order fields exist;
- ADR positions are stored separately from JP portfolio/NAV calibration;
- ADR returns do not enter JP candidate outcomes, calibration, Rank-IC, or candidate cohort review;
- currency is explicit (`USD`) and FX translation is display-only unless future governance approves otherwise.

- [ ] **Step 3: Implement separate store**

Suggested path:

```text
reports/user_state/external_adr_journal/{date}.jsonl
```

Suggested event:

```json
{"ts":"...","symbol":"SKHY","side":"buy","quantity":1,"price":100.0,"currency":"USD","source":"manual_external_broker_record"}
```

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests/unit/test_external_adr_journal.py tests/unit/test_api_portfolio_fill.py -q --basetemp=.runtime/pytest-p20-adr-journal -p no:cacheprovider
```

Expected: PASS, and existing JP manual fill behavior unchanged.

---

### Task 7: Forward Evidence Review Before Promotion

**Files:**
- Create: `src/hot_theme_rotator/reporting/skhy_event_review.py`
- Create: `tests/unit/test_skhy_event_review.py`

- [ ] **Step 1: Write failing tests**

The review must:

- group events by date to avoid duplicate same-day pseudo-samples;
- compute 1D/3D/5D forward relative returns for annotated Japanese candidates;
- report Rank-IC only when enough same-day cross-sectional candidates exist;
- return `insufficient_data` when event clusters are below the configured floor;
- never pool backdated data with live evidence;
- include transaction-cost hurdle context from Rule 16.0.

- [ ] **Step 2: Implement review harness**

Suggested output:

```json
{
  "verdict": "insufficient_data",
  "eventClusters": 3,
  "minEventClusters": 20,
  "rankIc5d": null,
  "costHurdle": 0.04,
  "promotionAllowed": false,
  "notes": ["live-only", "no probability", "requires ADR-0010 and Rule 16 gates"]
}
```

- [ ] **Step 3: Add scheduled/manual report command**

Optional CLI:

```powershell
python tools/review_skhy_event_overlay.py --asof 2026-07-31
```

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests/unit/test_skhy_event_review.py tests/unit/test_forward_signal_eval.py -q --basetemp=.runtime/pytest-p20-evidence -p no:cacheprovider
```

Expected: PASS.

---

### Task 8: Final Integration Verification

- [ ] Run focused P20 tests:

```powershell
python -m pytest tests/unit/test_daily_routine.py tests/unit/test_adr_watch.py tests/unit/test_refresh_skhy_adr_watch.py tests/unit/test_skhy_overlay.py tests/unit/test_api_dashboard.py tests/unit/test_skhy_event_review.py -q --basetemp=.runtime/pytest-p20-focused -p no:cacheprovider
```

- [ ] Run the existing memory/semi related tests:

```powershell
python -m pytest tests/unit/test_refresh_htr_price_db.py tests/unit/test_ticker_metadata.py tests/unit/test_theme_rotation.py tests/unit/test_hybrid_rerank.py tests/unit/test_api_dashboard.py -q --basetemp=.runtime/pytest-p20-semi-regression -p no:cacheprovider
```

- [ ] Run the daily smoke lane if time permits:

```powershell
python -m pytest -m "not slow" -q --basetemp=.runtime/pytest-p20-smoke -p no:cacheprovider
```

- [ ] Update `PROJECT_STATUS.md` with results:

Record exact pass/fail counts, known degraded paths, and whether the overlay remains `shadow` or has enough evidence for any limited promotion. Default answer should remain `shadow`.

---

## Acceptance Criteria

- SKHY is treated as an external catalyst input, not as a Japan candidate and not as a calibrated edge.
- Japanese market remains the primary execution/candidate market.
- If SKHY is not yet listed, unavailable, or stale, the system says so and does not create a false event.
- Candidate annotations explain "why this Japan semi name may react", not "buy this".
- ADR manual position records, if implemented, are separate from JP portfolio calibration and never route orders.
- Forward evidence is live-only and governed by Rule 16 and ADR-0010 before any promotion.
- No UI/API surface contains `probability`, `win_rate`, `expected_return`, or `edge` for the SKHY overlay unless future governance explicitly unlocks calibrated evidence.

## References For Engineer

- SEC EDGAR SK hynix F-1 filing index, 2026-06-24: https://www.sec.gov/Archives/edgar/data/2120882/000119312526280172/0001193125-26-280172-index.htm
- Jegadeesh and Titman, 1993, Returns to Buying Winners and Selling Losers: https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf
- Asness, Moskowitz, Pedersen, 2013, Value and Momentum Everywhere: https://pages.stern.nyu.edu/~lpederse/papers/ValMomEverywhere.pdf
- Barber and Odean, 2008, All That Glitters: https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/allthatglitters_rfs_2008.pdf
- Tetlock, 2011, All the News That's Fit to Reprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1018221
- Bailey and Lopez de Prado, Deflated Sharpe Ratio: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Kelly, 1956, A New Interpretation of Information Rate: https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf
- Kahneman and Tversky, 1979, Prospect Theory: https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Behavioral_Decision_Theory/Kahneman_Tversky_1979_Prospect_theory.pdf
