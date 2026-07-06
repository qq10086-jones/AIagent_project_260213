# Frontend / Interaction Design Optimization — Implementation Plan

> **For agentic workers:** implement task-by-task; steps use checkbox (`- [ ]`) syntax. Each task maps to a `docs/01_TASKS.md` entry (milestone **P13**) and is recorded in `PROJECT_STATUS.md`. Companion design intent: `docs/superpowers/specs/2026-06-06-frontend-interaction-design.md`.

**Goal:** Bring all four HTR dashboard variants to homogeneous deliverable quality (四变体均质) — eliminate display-data fabrication, consolidate duplicated render logic into shared single-source primitives, and raise every variant to the Rule 11.7 invariant bar — frontend-only, backend frozen.

**Architecture:** Zero-build React 18 (in-browser Babel). Source: `frontend/index.html` (boot + global CSS + grids + variant switcher) and `frontend/src/*.jsx|*.js`. Data: `bootWithApi()` merges `/api/dashboard` over the mock baseline; `useEnrichedCandidate` (htr-api.jsx) overlays real per-symbol `/api/symbol/{T}/*`. The structural strategy is to drive every honesty-bearing surface from ONE shared primitive in `htr-shared.jsx`/`htr-shared2.jsx`, so honesty becomes a property of one component instead of a per-variant promise.

**Tech Stack:** React 18 UMD + Babel standalone, inline-style design tokens (`--htr-*` CSS vars), `pytest` static contract tests (`tests/unit/test_frontend_ui_contracts.py`), headless Selenium screenshots (`.runtime/audit_shot*.py`) for Rule 11.7.6 evidence.

**Verification harness (every task):**
- Static contract: `python -m pytest tests/unit/test_frontend_ui_contracts.py -q` (assert real-data derivation / no-probability wording / ladder integrity).
- Smoke (no regression): `python -m pytest tests -m "not slow" -q --basetemp=.runtime/pt` (clean basetemp avoids the known Windows tmp PermissionError; expect 1294+ passed).
- Visual: re-shoot `.runtime/audit_shot.py` + `audit_shot_detail.py`, read the PNGs, confirm the fix renders and nothing clips/overlaps (Rule 11.7.6).
- Record outcome + evidence path in `PROJECT_STATUS.md`.

---

## Task P13-01: Phase 1 — Content-honesty + dead-interaction sweep

**Files:**
- Update: `frontend/src/htr-v4.jsx`, `htr-v3.jsx`, `htr-v2.jsx`, `htr-v1.jsx`
- Update: `tests/unit/test_frontend_ui_contracts.py` (new assertions lock each fix)
- Update: `docs/01_TASKS.md`, `PROJECT_STATUS.md`

- [ ] **Step 1: Write failing contract tests first.** Add assertions to `test_frontend_ui_contracts.py` that FAIL against current source:
  - `test_v4_event_counts_derived_not_hardcoded` — assert `htr-v4.jsx` does NOT contain the literal count array (e.g. regex for `['新闻'...'7'...]`) and DOES reduce over events by `kind`.
  - `test_v4_no_fabricated_score_sparkline` — assert no hardcoded `[42,...,78]` Sparkline array and no `(+12)` literal in htr-v4.jsx.
  - `test_v4_no_dead_decision_buttons_with_write_claim` — assert the `按钮仅写入决策日志` caption is gone (no false write claim).
  - `test_v3_news_high_count_derived` — assert htr-v3.jsx does not contain the literal `2 high`.
  - `test_v1_action_caveat_not_hardcoded` — assert no `跌破 156` literal in htr-v1.jsx.
  - `test_v2_chart_uses_real_overlay` — assert `V2ChartCard` references `candidate.kline` (not only the `kline` prop).
- [ ] **Step 2 (P0): V4 score sparkline + `(+12)`** — delete the two hardcoded `Sparkline data={[...]}` arrays (htr-v4.jsx:78,135) and strip the `(+12)` suffix at :134 so the label reads `研究分 · 未校准`. No per-symbol score series exists; render no spark (a real price spark may be added later from `top.kline`, explicitly labeled 价格, NOT score).
- [ ] **Step 3 (P0): V2 mock kline** — in `V2EditorialBrief` pass the enriched overlay; in `V2ChartCard` source `const bars = candidate.kline || kline;` and feed `bars` to `KLineChart` + the `{bars.length} sessions` label. (htr-v2.jsx:20,127,133)
- [ ] **Step 4 (P1): V4 event-count chips** — in `v4BuildEvents`/`V4Header` compute `const c = events.reduce((m,e)=>{m[e.kind]=(m[e.kind]||0)+1;return m;},{})`; render 新闻=`news_*` sum, 候选浮现=`candidate+leader`, 决策动作=`decision`, 宏观事件=`macro`, 风险触发=`risk`; drop any chip whose source count is undefined (not 0-from-literal). (htr-v4.jsx:54)
- [ ] **Step 5 (P1): V4 dead action buttons** — remove the `分批介入/设条件单/仅观察/放弃` button cluster and its false caption; replace with a static, non-interactive legend explaining the research-mode framing (advice-only, no write). Defer-track the decision-log-write idea in PROJECT_STATUS per Rule 11.10.2. (htr-v4.jsx:170-176)
- [ ] **Step 6 (P1): V3 `2 high`** — `const hi=(items||[]).filter(n=>n.weight==='high').length;` render `${hi} high`, hide the chip when `hi===0`. (htr-v3.jsx:759)
- [ ] **Step 7 (P1): V1 hardcoded caveat** — replace the literal `USD/JPY 跌破 156…` with `candidate.risk || (candidate.strategy && candidate.strategy.risk_warnings)`; render nothing if absent. Never hardcode a per-symbol macro caveat. (htr-v1.jsx:245)
- [ ] **Step 8 (P2): `screener_v2` + empty names** — port V3's `theme==='screener_v2'` suppression to V1/V2; filter empty `nameJa/nameCn` before the ` · ` join. (htr-v1.jsx:137,280; htr-v2.jsx:97)
- [ ] **Step 9 (P2): V2 masthead literal** — drop `vol. 4 issue 113` or derive an ordinal from `tradeDate`; keep the real `tradeDate` stamp. (htr-v2.jsx:65)
- [ ] **Step 10: Green the tests** — contract tests pass; smoke 1294+ passed; re-shoot V1–V4 screenshots, confirm each fix renders (V4 honest counts, no fake spark, static legend; V2 real chart; V3 derived count; V1 real/empty caveat). Record evidence + mark P13-01 done in PROJECT_STATUS.

---

## Task P13-02: Phase 2 — Async-state honesty + write-path reachability

**Files:** `frontend/src/htr-api.jsx`, `htr-shared.jsx`, `htr-v3.jsx`, `frontend/index.html`, `htr-data.js`; tests; docs.

- [ ] **Step 1: Failing tests** — `test_enrichment_tracks_pending_failed` (htr-api.jsx exposes per-endpoint status); `test_async_bodies_have_loading_and_failure_states`; `test_writepath_chips_in_global_nav`.
- [ ] **Step 2 (P1): per-endpoint arrival tracker** — `useEnrichedCandidate` returns `_status[endpoint] ∈ {pending,ok,failed}` for strategy/profile/factors/kline/aiBrief/debate/outcomes; stop pre-filling the overlays that have a real endpoint so the loading branch triggers (keep crash-safe shape via optional chaining in the bodies).
- [ ] **Step 3 (P1): honest async states** — AiBody/DebateBody/OutcomesBody/FactorBody render `生成中…` skeleton while `pending`, a visible `示例 · 未就绪` banner on `failed`, real content on `ok`.
- [ ] **Step 4 (P2): hoist write-path/nav chips** — move the chip cluster (search/watchlist/proposals/calibration/notifier) into the global `app-nav` in index.html so they render in every variant; remove the V3-only copies or have V3 consume the shared cluster.
- [ ] **Step 5 (P2): degraded coverage** — extend `__degraded` for `positions`/`dailyCockpit` + surface per-symbol overlay failures; add to the V3 banner SEC map.
- [ ] **Step 6 (P2): watchlist write** — wire add/remove to `POST /api/watchlist/{add,remove}` (existing whitelisted, user-allowed) with optimistic localStorage fallback; server list is source of truth, keep the honest label. Add a contract test asserting the real POST (Rule 11.10.4).
- [ ] **Step 7: Green + evidence.**

---

## Task P13-03: Phase 3 — Shared-component unification (contract-test-gated)

**Files:** `frontend/src/htr-shared.jsx`, `htr-shared2.jsx`, all `htr-v*.jsx`, `htr-cards.jsx`; tests; docs.

- [ ] **Step 1: Lock invariants with tests FIRST** — `test_ladder_keeps_seven_levels` (Rule 9.6), `test_score_label_uncalibrated_canonical` (8.3/9.4), `test_calibration_downgrade_wording_preserved`. These must stay green through every extraction.
- [ ] **Step 2:** Promote `<LadderMini>` + `<LadderTable>` + shared `ladderColor()`/`labelShort()` to htr-shared2.jsx; migrate the ~6 ladder renderers to consume them (V1 keeps its dense layout, consumes the shared body).
- [ ] **Step 3:** Extract `<ResearchScoreStat>` (+ `<LeaderIdentity>`) with ONE canonical `研究分 · 未校准`; migrate V1/V2/V3/V4 headers.
- [ ] **Step 4:** Canonical calibration strings in `HTR/GLOSSARY.LABELS`; reference from CalibPill / leader stat / OutcomesCard / CalibCard / V2 pane.
- [ ] **Step 5:** `MarketTempCell variant='hero'`; retire `V3MarketTile`. `<CockpitCard>`+`<CandidateRow>` shared; V2 renders shared `<DecisionLog>`; fix htr-v2.jsx:191 `fmtPrice(c.price,0)`.
- [ ] **Step 6:** `--htr-heat-hot` ramp distinct from `--htr-bear`; point `heatColor`/`heatBg` at it.
- [ ] **Step 7: Green (all invariant tests + smoke) + re-shoot all 4 variants (consolidation must be visually identical pre/post) + evidence.**

---

## Task P13-04: Phase 4 — Legibility-floor + IA/visual polish (V3 first, then parity)

> **GATE before Step on StrategyCard collapse:** resolve the Rule 11.6 question (spec §Governance P4-E) — clarify the rule (collapsed body OK if banner+disclaimer+risk summary stay in DOM, with a contract test) OR keep the body open. Do the rule action FIRST.

**Files:** `frontend/src/htr-v3.jsx`, `htr-shared.jsx`, `frontend/index.html`; `docs/02_GOVERNANCE.md` (Rule 11.6 clarification if chosen); tests; docs.

- [ ] **Step 1: Failing tests** — `test_v3_functional_labels_meet_floor` (no functional label <11px / mono <10.5 in htr-v3.jsx, eyebrow allowlist), `test_v3_outcomes_no_low_maxheight`.
- [ ] **Step 2 (P1):** raise sub-floor functional labels to ≥11px (mono ≥10.5); eyebrows exempt. (htr-v3.jsx:277,654,714,716,722,736,768)
- [ ] **Step 3 (P2):** collapse the type scale to a 6–7 step ramp driven by `--htr-fs-*`; replace inline px in htr-v3.jsx (preserves floor).
- [ ] **Step 4 (P2):** remove outcomes `maxHeight:180`; cap high (~600px). (htr-v3.jsx:504)
- [ ] **Step 5 (P2):** retune hierarchy — temp numeral ~22–24px, leader price ~38–40px. (htr-v3.jsx:217 vs 253)
- [ ] **Step 6 (P2):** density — drop V3LadderMini from LeaderCard; StrategyCard collapse only AFTER the Rule 11.6 gate; tab the two governance feeds. (htr-v3.jsx:66-80)
- [ ] **Step 7 (P2):** responsive — rail `repeat(2,1fr)`; K-line `padding.right = min(118, width*0.28)` + `innerW` positive floor; stack leader-grid <1320. (index.html:30,54,64; htr-shared.jsx:197,228)
- [ ] **Step 8 (merged a11y):** focus ring + ink-4 contrast (Phase-5 items that touch this edit surface).
- [ ] **Step 9: Green + re-shoot V3 collapsed + one expanded (Rule 11.7.6) + evidence.**

---

## Task P13-05: Phase 5 — Accessibility hardening

**Files:** `frontend/src/htr-shared.jsx`, `htr-v3-modals.jsx`, `frontend/index.html`; docs.

- [ ] **Step 1 (P1):** `:focus-visible` accent ring on `[role=button]`/`[tabindex='0']`; remove blanket `outline:none`.
- [ ] **Step 2 (P1):** fix `--htr-ink-4` AA failure on real text (switch to ink-3 or darken to measured ~3:1); record contrast math in PROJECT_STATUS before shipping the token.
- [ ] **Step 3 (P2):** ModalShell `role=dialog`/`aria-modal` + focus trap/restore; `aria-label` on ×/chevron; Term tooltip `role=tooltip` + Escape.
- [ ] **Step 4 (P2):** darken ink-3 ~#71695C; nav/chip `:focus-visible`; replace hardcoded `#fff` SVG fills with `var(--htr-accent-ink)`.
- [ ] **Step 5: Green + evidence.** (Items already merged into P13-04 are not repeated.)

---

## Task P13-06: Phase 6 — All-variant layout/IA parity (四变体均质, committed)

> Prerequisite: the Rule 11.7 Scope annotation (spec §Governance) recording the elective all-variant invariant commitment.

**Files:** `frontend/src/htr-v1.jsx`, `htr-v2.jsx`, `htr-v4.jsx`, `frontend/index.html`, shared; tests; docs.

- [ ] **Step 1 (P1):** V4 — pin `V4LeaderCard` to the top of the spine, outside the time-sorted stream. (htr-v4.jsx:198-205)
- [ ] **Step 2 (P1):** V2/V4 — selected-symbol picker via the shared `htr_symbol` key (mirror V1/V3 `setSel`). (htr-v2.jsx:9; htr-v4.jsx:10)
- [ ] **Step 3 (P1):** V1 — reuse FactorCard+OutcomesCard + a collapsed StrategyCard for Rule 9.6 disclosure parity. (htr-v1.jsx:31-49)
- [ ] **Step 4 (P2):** variant switcher `默认` marker on V3; pin GatesChip outside horizontal scroll. (index.html:16-18,137-162)
- [ ] **Step 5 (P2):** apply the Phase-4 legibility floor + density discipline to V1/V2/V4 (parity); clear the misc affordance grab-bag.
- [ ] **Step 6 (P2):** V2 §E evidence moved under §A/B or anchored.
- [ ] **Step 7: Green + re-shoot all 4 variants collapsed + expanded + evidence; mark P13 milestone complete in PROJECT_STATUS.**

---

## Sequencing & regression notes
- Phases are ordered to consolidate corrected code (Phase 3 after Phase 1), not buggy code, and to batch shared edit surfaces (a11y tokens ride Phase 4). Within a phase, do P0 → P1 → P2.
- Every phase ends green on the contract + smoke lanes and with fresh Rule 11.7.6 screenshot evidence recorded in PROJECT_STATUS.md.
- Open backend-dependent items stay deferred (spec §Deferred); do not stub them as if real.
- Rule-first gates: the Rule 11.7 all-variant annotation (before Phase 4/6 layout work) and the Rule 11.6 StrategyCard-collapse clarification (before P13-04 Step 6) MUST land before their dependent steps.
