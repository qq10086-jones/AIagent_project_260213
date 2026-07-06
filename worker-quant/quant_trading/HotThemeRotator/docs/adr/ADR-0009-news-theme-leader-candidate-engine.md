# ADR-0009: News-Catalyzed Theme-Leader Candidate Engine

- Status: Proposed (2026-06-14)
- Supersedes: nothing (augments the P12-03d sibling price screener)
- Related: 00_DESIGN §策略, P3 (theme_detector), P4 (leader_ranker / signal_engine),
  P15 (news pipeline), Rule 8.2 (PIT), Rule 8.3/9.4 (uncalibrated honesty), Rule 4
  (no silent parameter changes), Rule 11.9 (display honesty).

## Context

The project's namesake strategy is **新闻催化 + 市场温度 + 热点主题 + 龙头强度**
(00_DESIGN). But the LIVE candidate pipeline (`daily_routine` → sibling
`screener.py`) selects on **price momentum + liquidity + fundamentals only** — zero
news, zero theme. HTR's own engine (`theme_detection.theme_detector`,
`leader_ranking.leader_ranker`, `signal_engine`) was built and unit-tested (P3/P4)
but **never wired into candidate generation**. Fresh news now exists
(`stock_news_fetcher`, P15-01: Google News JP → `reports/news/{date}.json`,
classified into 6 themes). So the inputs are finally available to make the
news-driven engine real.

`rank_theme_leaders` needs `LeaderCandidateInput(symbol, theme_id, theme_score,
return_pct, volume_ratio, turnover_jpy, overheat_score)` — i.e. **symbols already
tagged to a theme** + factors. The missing link is **news → ticker → theme**.

## Decision

Build a **hybrid** engine that AUGMENTS, not discards, the working price screener
(the screener gives a liquid, strong universe; the news/theme layer adds the
catalyst + theme-leader intelligence the design demands):

1. **News → ticker (name match).** Match fresh-news headlines to tickers by
   company `name` (the HTR DB `tickers` table has `symbol, name, sector`). A
   headline mentioning "ソフトバンク" → 9984.T. Deterministic substring/alias match
   (Rule 8.3 no-LLM). Each match inherits the news item's themes + ts.
2. **Ticker → news-catalyst score per theme.** Aggregate: for each ticker,
   {themes it has fresh news in, news_count, latest_ts}. A ticker is "catalyzed in
   theme T" iff it has ≥1 fresh (PIT, last N days) news item classified to T.
3. **Theme heat from news.** Theme T's heat = its fresh-news count (already in the
   `stock_news_fetcher` payload's `theme_counts`).
4. **Leader ranking.** For screener-universe tickers that are news-catalyzed in a
   hot theme, build `LeaderCandidateInput` (theme_score = normalized theme heat;
   return_pct/volume_ratio/turnover from the HTR DB price layer) → `rank_theme_leaders`.
5. **Final candidate score = hybrid rerank.** Combine the screener's price/momentum
   score with the news-catalyst + leader-in-theme score. Candidates WITHOUT a fresh
   news catalyst are NOT discarded (the screener score still stands) — they are
   simply not flagged as theme leaders. This keeps the system honest when news is
   thin and never fabricates a catalyst.

**Honesty constraints (binding):**
- Output stays `uncalibrated_research_score` (Rule 8.3/9.4) — the news/theme rerank
  is a research signal, not a validated probability. It does NOT alter the P12-05
  calibration track or its locked criteria (Rule 8.2.3).
- PIT (Rule 8.2): only news with `published_ts ≤ decision_cutoff` may catalyze a
  candidate for that trade date.
- No silent parameter changes (Rule 4): the hybrid weighting + theme/sector maps are
  explicit config, changes go through the task/ADR flow.
- Display honesty (Rule 11.9): a candidate's "theme / news catalyst" annotation must
  reflect REAL matched news; "no catalyst" is surfaced, never faked.

## Phased plan (each phase = a complete, tested loop increment)

- **P15-02a** News→ticker catalyst aggregator: `news_catalyst.py` — name-match news
  to tickers, aggregate per-ticker theme catalysts. Pure + unit-tested (injected
  news + ticker map; no network).
- **P15-02b** Theme-leader builder: wire catalyst + HTR DB factors →
  `LeaderCandidateInput` → `rank_theme_leaders`; produce ranked theme leaders.
- **P15-02c** Hybrid candidate rerank: combine screener score + catalyst/leader
  score into the candidate panel; annotate each candidate with matched theme + news.
- **P15-02d** Wire into `daily_routine` / dashboard so live candidates reflect news
  catalysts; honest "no catalyst today" state when news is thin.
- **P15-03** (separate) news-factor → realized-outcome feedback (Rule 4 flow).

## Update 2026-06-14: metadata blocker found (before building — good)

Feasibility probe of the `tickers` table killed the original "news→ticker by name
match" approach and surfaced a deeper gap:
- `tickers.name` is Japanese (good for JP-news matching) BUT only **64 / 945
  (6%)** of the live universe has a name; of today's 50 candidates only **3** have
  one. Name-match would catalyze ~3/50 — useless.
- `tickers.sector` is **also 6%** coverage (and the few present are English labels).
- The sibling's `news_items.related_tickers` is `[]` for Google-News items — the
  sibling never linked them either.

**Root cause:** the HTR DB lacks ticker theme/sector/name metadata for ~94% of the
tradeable universe — so news→theme→ticker mapping is blocked at the *metadata*
layer, not just the engine-wiring layer.

**Plan adjustment — add a prerequisite phase:**
- **P15-02a′ (prereq): HTR-native ticker metadata.** Fetch `sector` + `industry`
  (+ longName) per universe ticker from **yfinance `.info`** (confirmed available:
  6584.T → "Consumer Cyclical" / "Auto Parts") into an HTR-native metadata store,
  analogous to the price/news refreshers. Idempotent, polite, offline-degrading.
- The theme map then keys off **yfinance industry/sector → 6-theme** (high coverage),
  not the sparse `tickers` table. News still supplies theme HEAT + catalyst recency.
- Everything downstream (catalyst aggregate → leader rank → hybrid rerank) is
  unchanged; only the ticker→theme key changes from name-match to yfinance-sector.

Revised phase order: **P15-02a′ (ticker metadata)** → P15-02a (catalyst aggregate,
keyed on yfinance industry→theme + news heat) → P15-02b (leader rank) → P15-02c
(hybrid rerank) → P15-02d (wire live).

## Consequences

- The candidate panel becomes genuinely news-catalyzed/theme-aware — the namesake
  feature goes live, incrementally and honestly.
- Risk: name-matching is imperfect (aliases, partial names). Mitigated by an alias
  map + recording match confidence; unmatched news simply doesn't catalyze (fail-open
  to the price screener, never fabricate).
- The forward-sample/calibration track is unaffected (still scores the same prediction
  schema; the rerank changes WHICH candidates, which is exactly what we are validating).
- Reversible: the hybrid is config-gated; setting the news weight to 0 restores the
  pure price screener.
