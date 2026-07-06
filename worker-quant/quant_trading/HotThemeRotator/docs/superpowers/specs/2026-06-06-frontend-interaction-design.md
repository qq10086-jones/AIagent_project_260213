# Frontend / Interaction Design Optimization — Design Spec

> Status: proposed 2026-06-06. Source: 15-agent multi-dimensional design audit (7 dimensions, 71 adversarially-verified in-scope findings) of the HTR dashboard V1–V4, reviewed by Codex (codex:rescue) before authoring. Backend FROZEN this batch — frontend-only, consumes existing `/api/dashboard` + `/api/symbol/{T}/*` reads and the 7 already-existing Rule 11.5-whitelisted POST endpoints (user decision 2026-06-06). Implements milestone **P13**.

## Goal

Take the HTR research dashboard from "four working-but-uneven variants" to **four homogeneously deliverable-quality variants** (user decision 2026-06-06: 四变体均质), by (1) eliminating every remaining display-data fabrication so no surface masquerades as live, (2) collapsing the duplicated render logic that is the structural *source* of those fabrications into shared single-source primitives, and (3) raising every variant to the Rule 11.7 invariant bar (legibility, no scroll-traps, honest async states, accessibility) while keeping V3 the nominal default. All within Rule 3 advice-only, the Rule 11.5 write whitelist, and the no-probability red-lines (8.3 / 9.4).

## Recommended Approach

Sequence work so that **unconditional content-honesty debt is cleared first**, the **structural fix that prevents its recurrence (DRY consolidation) lands next**, and **per-variant visual/IA/a11y polish rides the consolidated primitives last** — minimizing rework and regression. Six phases:

1. **Phase 1 — Content-honesty + dead-interaction sweep** (binds ALL variants; P0/P1). Delete or derive every fabricated literal; fix dead/false-captioned controls. Surgical, low-regression, highest governance value.
2. **Phase 2 — Async-state honesty + write-path reachability** (P1/P2). Honest loading/stale/failed states for every `/api/symbol/{T}/*` overlay; make whitelisted write-paths reachable from every variant (user allowed existing POSTs).
3. **Phase 3 — Shared-component unification** (P2, governance-sensitive). Consolidate the ladder (~6×), market tile (2×), leader header (4×), cockpit card (2×), candidate row (2×), and calibration/score label strings (5 wordings) into shared primitives — contract-test-gated so the no-probability meaning and seven-tier ladder survive.
4. **Phase 4 — Legibility-floor + IA/visual polish, V3 first then parity** (P1/P2). Clear the 11.7.4 floor, remove the re-opened outcomes scroll-trap, retune hierarchy so the real leader dominates, collapse density. Merge the V3-affecting a11y token edits here (Codex: same edit surface).
5. **Phase 5 — Accessibility hardening** (P1/P2). Focus rings, contrast tokens, modal/ARIA semantics across variants (operability is NOT covered by the 11.7 layout carve-out).
6. **Phase 6 — All-variant layout/IA parity** (P2, now committed not optional per 四变体均质). Bring V1/V2/V4 layout/IA to the same bar as V3 using the Phase-3 shared primitives.

Why this order (Codex-reviewed): honesty fixes touch isolated literals and must not wait behind a refactor; the DRY refactor then consolidates *already-corrected* code, not buggy code; polish and a11y token work share the same screenshots so they batch after the structure is stable.

## Severity rubric (applied uniformly — addresses Codex "draw the P0/P1 line consistently")

- **P0** — fabrication of a **decision-relevant quantity or trajectory** on any variant (actively misleads a trade decision; e.g. an invented score sparkline implying momentum → Rule 9.4-adjacent; a mock price series rendered as the real chart), OR a content-red-line breach on the default V3 that a user would plausibly act on.
- **P1** — any other content-red-line breach: a static fabricated label/caveat, a dead or false-captioned interactive control, or stale-mock-rendered-as-real, on **any** variant (the Rule 11.7 carve-out does NOT relieve content honesty).
- **P2** — layout, legibility-floor, IA, visual hierarchy, color semantics, accessibility polish, and DRY/consolidation. *Severity reflects urgency, not whether we do it* — under 四变体均质 the P2 layout tail is committed work, not an optional skip.

## Design principles (north-star)

1. **Honesty over polish** — render only what the backend observed; no hardcoded count, fabricated delta, invented sparkline, or stale mock as live. Binds every variant (Rule 11.9, clarified 2026-06-05).
2. **Wired or visibly-labeled, never silently dead** — every control calls a real Rule 11.5 endpoint and shows the real result, or wears a visible "演示 · 未接线" marker and is defer-tracked. No caption may claim a write that does not happen (Rule 11.10).
3. **Single source of truth for every honesty-bearing primitive** — ladders, score/calibration labels, market tiles, leader headers from ONE shared component/string table. DRY here is a *governance control*: divergence-without-a-shared-source is exactly what produced the V4 "研究分 (+12)" fabrication.
4. **Honest uncertainty as a first-class state** — async overlays show loading / stale / failed distinctly; a canned mock that never resolves is a display-honesty breach, not an acceptable placeholder.
5. **Four equal variants, one nominal default** — all four meet the Rule 11.7 invariant bar (legibility floor, no scroll-traps on bounded primary content, in-flow expansion, responsive robustness); V3 stays the default landing variant. Content red-lines already bound all four; this batch electively raises the *layout* bar everywhere too.
6. **One clear focal point per screen** — the real #1 leader visually dominates the macro/temperature strip, never buried mid-stream.
7. **Update the rule before the work** — anchor every change to a task in `docs/01_TASKS.md` + record in `PROJECT_STATUS.md`; where a rule's scope is ambiguous, clarify the rule text FIRST (the 2026-06-05 Rule 11.7 carve-out clarification is the template).

## Scope by phase

### Phase 1 — Content-honesty + dead-interaction sweep (P13-01)
- **P0** V4 leader/header score sparklines `[42..78]` + `研究分 (+12)` delta — invented score trajectory contradicting the real 41.77 (9.4-adjacent). Delete both `Sparkline` arrays + strip `(+12)`; label plain `研究分 · 未校准`. `htr-v4.jsx:78,134-135`
- **P0** V2 price chart renders the **mock** kline: `V2ChartCard` consumes the `kline={data.kline}` prop (boot mock baseline) instead of the enriched real overlay `candidate.kline`. Source from `candidate.kline || kline`. `htr-v2.jsx:20,127,133`
- **P1** V4 header event-count chips hardcoded `[7,6,8,3,0]` — derive by `ev.kind` from `v4BuildEvents(data)`; drop any chip with no real source; render honest 0. `htr-v4.jsx:54`
- **P1** V4 right-rail 分批介入/设条件单/仅观察/放弃 buttons dead AND caption falsely claims `按钮仅写入决策日志` (decision-log is NOT a Rule 11.5 path). **Decision: remove the buttons → static legend** (Codex + user lean; demo-label rejected as it still trains a phantom action). `htr-v4.jsx:170-176`
- **P1** V3 news chip `2 high` hardcoded — derive `items.filter(weight==='high').length`; hide when 0. Reuse for the V4 counters. `htr-v3.jsx:759`
- **P1** V1 action zone shows a hardcoded `USD/JPY 跌破 156 取消激进档` caveat for **every** candidate (from the 8035.T mock) — render `candidate.risk` / `strategy.risk_warnings`, else drop. `htr-v1.jsx:245`
- **P2** placeholder `theme==='screener_v2'` rendered raw in V1/V2 (V3 already guards) — port the guard; filter empty `nameJa/nameCn` before the ` · ` join. `htr-v1.jsx:137,280; htr-v2.jsx:97`
- **P2** V2 masthead `vol. 4 issue 113` static literal implying a nonexistent publication cadence — drop or derive from `tradeDate` ordinal. `htr-v2.jsx:65`

### Phase 2 — Async-state honesty + write-path reachability (P13-02)
- **P1** No loading/stale/failed state on async overlays — broaden beyond LLM panels (Codex): `useEnrichedCandidate` starts every overlay (strategy/profile/factors/kline/aiBrief/debate/outcomes) from synthesized mock defaults, so a failed fetch leaves canned mock indistinguishable from real. Track per-endpoint arrival (`_pending`/`_failed` keyed by endpoint); each body shows `生成中…` skeleton while pending and a visible `示例 · 未就绪` banner on failure; stop pre-filling the overlays that have a real endpoint so the loading branch actually triggers. `htr-api.jsx:24,32-52; htr-data.js:322-331`
- **P2** Write-path/nav chips (search/watchlist/proposals/calibration/notifier) exist only in V3 — UX consistency, NOT a Rule 11.10 violation (Codex: 11.10.2 only needs *one* consumer, V3 satisfies it). Under 四变体均质 + existing-POST allowance, hoist the chip cluster into the global `app-nav` (index.html) so it is variant-independent. `htr-v3.jsx:111-116 vs index.html:155-162`
- **P2** `positions`/`dailyCockpit` mock fallback not tracked in `__degraded` + per-symbol overlay failures not surfaced — extend the degraded set and banner SEC map; tie per-symbol failures to the Phase-2 arrival tracker. `index.html:197-217`
- **P2** Watchlist add/remove wired to the whitelisted `POST /api/watchlist/{add,remove}` with optimistic localStorage fallback (user allowed existing POSTs). Keep the honest label; the server list becomes the source of truth, localStorage the cache. `htr-shared.jsx:572-582`

### Phase 3 — Shared-component unification (P13-03)
Governance-sensitive consolidation (NOT "pure DRY" — Codex). Each consolidation MUST be gated by `test_frontend_ui_contracts.py` assertions proving: the `未校准研究分` wording survives, the Rule 9.4 downgrade survives, and all seven ladder levels (Rule 9.6) survive.
- `<LadderMini>` + `<LadderTable>` → `htr-shared2.jsx`, shared `ladderColor()` / `labelShort()` (replaces ~6 renderers). `htr-v1.jsx:186-252; htr-v3.jsx:282-316,385-394`
- `<ResearchScoreStat>` (+ `<LeaderIdentity>`) with ONE canonical `研究分 · 未校准` label (replaces 4 headers; locks P1-B from regressing). `htr-v1/v2/v3/v4`
- Canonical calibration label strings in `HTR/GLOSSARY.LABELS` (replaces 5 wordings). `htr-cards.jsx:18,28; htr-v3.jsx:131-133,498`
- `MarketTempCell` gains a `variant='hero'` branch; retire bespoke `V3MarketTile`. `htr-v3.jsx:206-228; htr-shared2.jsx:12-37`
- `<CockpitCard>` + `<CandidateRow>` shared; normalize quote/quotes label; V2 renders shared `<DecisionLog>`; fix latent `htr-v2.jsx:191` price-decimal ternary → `fmtPrice(c.price,0)`.
- `--htr-heat-hot` amber→orange→red ramp distinct from `--htr-bear` so an overheated tile never shares a swatch with a loss/stop (decision: distinct ramp). `htr-shared.jsx:122-123`

### Phase 4 — Legibility-floor + IA/visual polish (V3 first, then parity) (P13-04)
- **P1** functional labels at 9.5–10px below the 11.7.4 floor (MiniStat, portfolio legend/holdings, decision-log code, theme leaders, TDnet chip) → ≥11px (mono ≥10.5); decorative eyebrows exempt. `htr-v3.jsx:277,654,714,716,722,736,768`
- **P2** type-scale noise (~17 inline sizes, half-px steps; `--htr-fs-*` tokens unused) → collapse to a 6–7 step ramp via CSS vars; preserves floor, clears the sub-floor labels.
- **P2** outcomes/历史命中 table clips to `maxHeight:180` (bounded primary evidence behind inner scroll, re-opens the killed trap) → remove; retain a **high cap ~600px** for pathological row counts (decision). `htr-v3.jsx:504`
- **P2** temperature hero (32px) out-shouts the real #1 leader (price 32 / symbol 30) → demote temp numeral ~22–24px, raise leader price ~38–40px (coordinate with the Phase-3 hero branch). `htr-v3.jsx:217 vs 250,253`
- **P2** V3 center restates the ladder 3× + 3 tall always-open cards above the tabs; right rail stacks 4 always-open cards → drop `V3LadderMini` from LeaderCard, default StrategyCard collapsed **(GATED: Rule 11.6 check first — see Governance)**, tab the two governance feeds. `htr-v3.jsx:66-80,266,320,347,382-396`
- **P2** V3 responsive edge cases (1320px rail orphans 4th card; K-line `padding.right:118` risks non-positive `innerW` <1080; leader-grid 244px squeezes ministats) → rail `repeat(2,1fr)`, responsive K-line padding `min(118,width*0.28)` + `innerW` floor, stack leader-grid <1320. `index.html:30,54,64; htr-shared.jsx:197,228`
- **Merged from Phase 5 (Codex):** V3-affecting a11y token edits — focus ring + ink-4 contrast — ride this batch (same screenshots).

### Phase 5 — Accessibility hardening (P13-05)
- **P1** keyboard-focusable rows/div-buttons have no visible focus ring; `.htr-term:focus` sets `outline:none` → `:focus-visible{outline:2px solid var(--htr-accent)}`; remove blanket `outline:none`. `htr-v3.jsx:569; htr-shared.jsx TOKEN_STYLE`
- **P1** `--htr-ink-4` (~2.3:1 on cream) muted text fails WCAG AA on REAL content (K-line axis dates, detail-tab hint) → switch to `--htr-ink-3` or darken ink-4 to a measured ~3:1, reserve ink-4 for non-text hairlines. `htr-shared.jsx:16,375-377`
- **P2** modals lack `role=dialog`/`aria-modal` + focus trap/restore; icon-only × and chevron lack names; tooltips not described → ModalShell ARIA + focus management; `aria-label`/`aria-expanded`; Term tooltip `role=tooltip` + Escape. `htr-v3-modals.jsx:22,31; htr-shared.jsx:617`
- **P2** borderline ink-3, missing nav/chip `:focus-visible`, hardcoded `#fff` SVG fills bypass the dark-mode `--htr-accent-ink` token path → darken ink-3 ~#71695C; add focus rings; `fill='var(--htr-accent-ink)'`. `htr-shared.jsx:16,75,363,400; index.html:20`

### Phase 6 — All-variant layout/IA parity (P13-06) — committed (四变体均质)
- **P1**(re-classified from carve-out P2) V4 buries the leader mid-timeline (emitted 15:30, time-sorted desc) → pin `V4LeaderCard` to the top of the spine, outside the sorted stream. `htr-v4.jsx:198-205`
- **P1** V2/V4 deep-dive hard-locked to `candidates[0]` (no picker) → selected-symbol state via the shared `htr_symbol` key; V2WatchlistTable rows / V4 spine rows set it. `htr-v2.jsx:9; htr-v4.jsx:10`
- **P1** V1 omits the governed StrategyCard (risk_warnings + catalyst) + factor/outcomes/AI/debate disclosure → reuse `FactorCard`+`OutcomesCard` + a collapsed StrategyCard (Rule 9.6 parity). `htr-v1.jsx:31-49`
- **P2** variant switcher: add a `默认` marker to V3; pin GatesChip outside the horizontal-scroll region. `index.html:16-18,137-162`
- **P2** V1/V2/V4 legibility-floor parity (the Phase-4 floor work applied to the other three under 四变体均质) + the misc affordance grab-bag (V1 risk-budget degraded-branch hides the toggle; V4 spine meta 9–9.5px + low-contrast buttons; V3 K-line zoom hint; manual-fill seeds empty Symbol; notifier dry-run preview cue).
- **P2** V2 §E (evidence) sits three sections below the §A claim → move under §A/B or anchor.

## Governance actions

- **No new rule needed for Phase 1** — the 2026-06-05 P12-04 clarification already binds 11.9/11.10 to every variant.
- **四变体均质 rule annotation (before Phase 4/6 layout work):** add a one-line note to the Rule 11.7 Scope clause recording that as of 2026-06-06 the project *elects* to hold all four variants to invariants 1–6 (the carve-out remains available for any future experimental variant). This is an elective tightening, not a constraint relaxation — record, don't rewrite.
- **P4-E StrategyCard default-collapse — GATE (Codex):** Rule 11.6 requires the StrategyCard to *show* risk_warnings + the seven-tier ladder (only the banner/disclaimer are explicitly non-collapsible). Before default-collapsing the body, EITHER (a) clarify Rule 11.6 to permit a collapsed body provided the banner+disclaimer+a risk summary stay in the DOM, with a contract test, OR (b) keep the body open and reduce density another way. Resolve the rule question first.
- **Phase 3 contract-test gate:** every consolidation must keep the `test_frontend_ui_contracts.py` assertions for uncalibrated wording, 9.4 downgrade, and seven ladder levels green.
- **Evidence:** re-shoot headless screenshots (collapsed + one expanded state) per Rule 11.7.6 after each phase; record in `PROJECT_STATUS.md`.

## Deferred — needs backend (out of scope this batch)
1. Real per-symbol **score-history** series (would let V4 show an honest score trajectory instead of deleting the spark) — needs a new `/api/symbol/{T}` field.
2. Real portfolio **NAV-trend** series for the live V3 navHistory sparkline — live gate already suppresses the mock; a real series needs-backend.
3. Server-side watchlist's **operational** value (Rule 12.4 cooldowns / silent intelligence) depends on Stage-2 push (ops gate) — the *wiring* (Phase 2) is frontend-only and now in scope.
4. Decision-log **write-path** for V4 action buttons — would need `decision_log` added to the Rule 11.5 whitelist (governance + backend). This batch removes the dead buttons instead.

## Locked decisions (2026-06-06)
- Endpoint scope: **existing Rule 11.5-whitelisted POSTs allowed** (no backend change). → Phase 2 wires watchlist + hoists write-path chips.
- Variant priority: **四变体均质** — all four to the Rule 11.7 invariant bar; V3 stays nominal default. → Phase 6 committed, not optional.
- Cadence: **spec/plan first, then execute from Phase 1.**
- V4 dead buttons (P1-C): **remove → static legend** (Codex).
- Outcomes cap (P4-C): **high cap ~600px** (Codex).
- Heat ramp (P3-F): **distinct `--htr-heat-hot`** (Codex).
- Watchlist (P2-D): **wire to existing POST** (follows from the endpoint-scope decision; supersedes the earlier defer option).
