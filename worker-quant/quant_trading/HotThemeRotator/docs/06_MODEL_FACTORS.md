# 06 — Model Factors & Calibration Reference

> Durable reference for "how the model works, what the factors mean, where the data
> comes from, and the calibration status." Stable reference doc (like 05_USER_GUIDE),
> not running progress. Last reconciled to code: 2026-06-17.

## Pipeline at a glance

`screener (picks candidates) → news-catalyst rerank (reorders) → calibration (currently OFF) → analysis gates (Event Desk + P17)`

Output is **advice-only, an uncalibrated ranking score — NOT a probability** (Rule 9.4).

## Layer 1 — Candidate screener factors (the actual stock picker)

The live candidate list comes from the sibling `Project_optimized/screener.py`
(`screener_v2`, `_compute_alpha_factors`). The composite "alpha score" is a weighted
blend of 7 cross-sectionally-ranked factors:

| Factor | Weight | Formula | Meaning |
|---|---|---|---|
| `mom_20` | **0.25** | `close[-1]/close[-21] − 1` | 20-day (~1mo) total return — recent momentum |
| `mom_60` | 0.15 | `close[-1]/close[-61] − 1` | 60-day (~3mo) return |
| `vol_adj_mom20` | 0.15 | `mom_20 / vol_20` | momentum per unit volatility (rewards smooth trends) |
| `vol_z` | 0.15 | `(volume[-1] − mean₆₀)/std₆₀` | **trading-VOLUME z-score (成交量)** — volume-spike / breakout-attention tilt, NOT price volatility |
| `sharpe_20` | 0.10 | `(mean₂₀/vol₂₀)·√252` | annualized 20-day risk-adjusted return |
| `adv_rank` | 0.10 | percentile of avg daily turnover | liquidity rank |
| `high52w_rank` | 0.10 | proximity to 52-week high | "near highs" trend-proximity |

Factors are cross-sectionally **rank-normalized (percentile) then weighted-summed**
(`screener.py:401, :503`) — rank-norm prevents a high-variance factor from dominating
by magnitude, but does NOT fix collinearity.

- **~0.75 of the score is one correlated momentum/proximity cluster** (mom_20 + mom_60
  + vol_adj_mom20 + sharpe_20 + high52w_rank) — effectively a single momentum bet with
  five votes, NOT a diversified score.
- `fundamental_score` (0–1, EPS-based) is applied as a **POST-selection multiplier**
  AFTER the top-k is chosen (`screener.py:511`) — so it re-ranks survivors but does NOT
  influence WHICH stocks are selected. (Mismatch with comments; flagged by Codex.)
- Hard liquidity floor: `min_adv ≈ ¥20M` (a gate, not a score — reasonable).

**How the weights were set:** HAND-SET heuristic defaults in `alpha_weights` — no
optimization/grid-search/backtest. Per Rule 4, changing them requires the Change Log
governance process AND a walk-forward backtest with overfit guards + costs (not a hand-swap).

### Weight review verdict (Codex, 2026-06-17): UNREASONABLE as-is
- ~75% concentrated on one correlated momentum cluster — in a market where **momentum is
  documented as weak/absent (Japan)**, i.e. structurally fighting the evidence.
- No value / quality / low-vol / size / reversal factors at all.
- `vol_z` (+0.15) is a volume-spike tilt, name-misleading.
- Fundamentals only post-selection, not a screening criterion.
- Codex's proposed (UNVALIDATED) reweight: cap momentum cluster to ~0.45, raise
  adv_rank→0.20 + fundamental→0.20 (moved into ranking), keep vol_z. **Must be
  walk-forward-validated with costs before any live change** — and no weighting is proven
  to have edge regardless (the screener shows none OOS).

## Data provenance (where the factors come from)

- **Prices (OHLCV):** originally **J-Quants** (official JPX vendor) backfilled history →
  now HTR-native `data/raw/htr_market.db` (snapshot of the sibling DB + **daily yfinance
  refresh** per JPX trading day, `auto_adjust=False`). Pulled via `get_close_vol_multi`.
- **Turnover / ADV:** derived from `close × volume`.
- **Fundamentals:** `fundamental_snapshots` table (EPS etc.) + yfinance EPS fallback.
- Read-only from HTR's side (ADR-0005); refreshed daily by `daily_routine`.

## Layer 2 — News-catalyst rerank (P15 / ADR-0009)

Reorders (never drops) the screener candidates by news attention:
- `stock_news_fetcher` (Google News JP RSS) → `news_theme_classifier` → **8 themes**
  (semi/ai/auto/bank/defense/energy/optical/memory) + **6 macro categories**
  (monetary/fiscal/fx/trade/overseas/geopolitics).
- `theme_heat` = theme count ÷ max count (0–1); `ticker_metadata` maps ticker → theme.
- `catalyst_score` = max theme-heat across a candidate's themes.
- `hybrid_rerank`: **blended = 0.70 × minmax(screener score) + 0.30 × catalyst_score**
  (`news_weight = 0.30`, explicit config / Rule 4). `theme_leaders` flags the 👑 per theme.
- Honesty (Rule 11.12): the DISPLAYED score stays the raw screener score; only the ORDER
  reflects news. 排序≠分数.

### 2026-06-18 update: quality-gated catalyst evidence

The rerank now separates `company`, `sector`, and `none` evidence:

- `company`: served news explicitly links the symbol or matches the metadata company name in its title.
- `sector`: only sector/industry metadata maps the ticker to a hot theme. This is exposure, not company news.
- Company evidence uses the full 0.30 catalyst weight.
- Sector-only evidence uses a weak 0.08 nudge, then takes a 0.08 sector-only penalty because it is exposure rather than company news.
- Sector-only candidates with `mom_20 >= 0.30` take an additional 0.25 chase penalty.
- Low `fundamental_score` can subtract up to 0.12 from the blended ordering score; repeated unchanged recent closes can subtract 0.12.
- Theme-leader badges are company-catalyst only. Sector-only candidates expose `sectorCatalyzed=true`, keep `newsCatalyzed=false`, and cannot become theme leaders.

### 2026-06-20 planned update: memory/semi rotation overlay

The next HTR layer will add a small, explicit rotation overlay for memory and
semiconductor regimes. It is designed to fix the current blind spot where the
system can see generic theme heat but does not know whether the real battlefield
is the leader (`285A.T` Kioxia) or a second-line expansion candidate.

The overlay will add explanatory fields, not calibrated probabilities:

- `themeRegime`: `memory_hot`, `semi_hot`, `memory_semi_hot`, or `neutral`.
- `coreThemeDataFresh`: whether the configured core memory/semi basket has latest-trading-day price data.
- `leaderExtended`: whether the reference leader is already crowded or near a momentum climax.
- `chaseRisk`: `none`, `watch`, or `study_only`, based on recent returns, distance from highs, and volume expansion.
- `secondLineCandidate`: true only when several facts support expansion beyond the leader.
- `rotationScore`: a small ordering overlay used after the screener and catalyst rerank.

Initial gates are deliberately conservative:

- Kioxia (`285A.T`) and major JP semi names must be present and fresh before the
  dashboard can speak confidently about a memory/semi rotation.
- An extended leader stays visible as the reference leader but is not promoted as
  a clean entry.
- Second-line candidates need at least two independent facts: relative strength
  versus the leader, volume expansion, latest price data, company catalyst,
  better valuation/fundamental read, or improving 5d/20d trend.
- The overlay can lightly reorder but cannot display a win-rate, expected return,
  or probability. It inherits Rule 11.14.

## Layer 3 — Calibration (built, currently DOWNGRADED)

The isotonic recalibrator IS fitted (762 bootstrap samples, 3-day horizon; maps raw
score → empirical hit-rate, e.g. 0.65→47%, 0.98→68%). But the K-fold verdict is
**`downgrade`**: `OOS Brier 0.2823 ≥ random baseline 0.2500` → informationless
out-of-sample → Rule 9.4 forces display as `uncalibrated_research_score`.

So the score is a RANKING signal with **no demonstrated predictive edge — never a win-rate.**

- **When does it activate?** Only when a calibrator passes the LOCKED Rule 8.2.3 criteria
  on FORWARD (not bootstrap) data: ≥20 independent date-clusters + beats all baselines +
  cluster-bootstrap CI excludes zero + leakage audit clean + OOS Brier < random. Forward
  clusters accrue toward 20 (~early July 2026 for the count) — but it activates ONLY if it
  genuinely passes; current evidence (backdated Brier 0.311; forward 5D behind passive)
  suggests it likely won't. "Available" = eligible to be re-judged, not guaranteed on.
- **Manual activation?** Re-fit / re-validate any time (`tools/fit_isotonic_recalibrator.py`,
  `tools/kfold_validate_isotonic.py`, the purged-walk-forward validator) — safe, idempotent;
  the downgrade auto-clears IF a run passes. **Force-activating past a failing gate is
  blocked by design** (Rule 8.2.2 locked criteria) and would break the honesty-contract
  tests; it wouldn't crash, but it would display fabricated probabilities (noise dressed as
  confidence). The gate failing IS the correct, honest output.

## Layer 4-5 — Analysis support (does NOT pick candidates)

- **Event Desk (P16):** priced-in read (1d/5d/20d returns, excess vs 1306.T, distance from
  20d high, freshness label) + exposure map.
- **P17 disclosure-drift direction (ADR-0010, execution-gated, overfit-guarded):**
  `tradability` (JPX tick cost, 100-share lot / 34% cap, ¥50M ADV floor, 2× cost stress) +
  `disclosure_surprise` + `overfit_gate` (Deflated Sharpe) + `disclosure_drift_review`
  (currently `insufficient_data`, corpus accruing). Built + Codex-reviewed; not yet driving
  live candidate selection.

## One-line summary

The model picks liquid, recently-rising JP stocks (hand-weighted momentum + liquidity,
fundamental down-weight), reorders them by today's news-theme heat, and shows an
uncalibrated 0–100 score — **with no demonstrated edge.** Fundamentals, FX/rates, and the
disclosure-drift direction are analysis layers, not inputs to the score.
