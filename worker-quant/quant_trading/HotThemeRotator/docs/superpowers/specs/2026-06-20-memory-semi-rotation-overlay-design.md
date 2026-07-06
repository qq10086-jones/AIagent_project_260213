# Memory/Semi Rotation Overlay Design

Date: 2026-06-20

## Goal

Make HTR understand the current memory/semiconductor hot-money regime without
turning theme heat into fake certainty. The system should know when the leader is
already extended, when second-line names are starting to participate, and when
core theme data is too incomplete to speak confidently.

## Problem

The current pipeline can classify news into `memory` and `semi`, then map
candidate tickers to themes. That is not enough for the current market:

- Kioxia (`285A.T`) is the reference leader for memory, but it is missing from the
  local HTR price DB.
- Some major semiconductor names have stale or incomplete local price history.
- Sector-only theme heat can push candidates up even when there is no company
  catalyst.
- The system cannot distinguish "leader is too extended to chase" from "second
  line is beginning to participate".

## Recommended Approach

Use a conservative overlay after the existing screener and quality-gated catalyst
rerank.

The overlay should not replace the raw screener score, should not unlock
calibration, and should not display a probability. It should emit explanatory
fields and apply only small ordering boosts or penalties.

## Data Coverage

Add a priority memory/semi universe to the HTR price refresh path. At minimum:

- `285A.T` Kioxia
- `8035.T` Tokyo Electron
- `6857.T` Advantest
- `6146.T` Disco
- `6920.T` Lasertec
- `7735.T` Screen
- `3436.T` SUMCO
- `4063.T` Shin-Etsu Chemical
- `6525.T` Kokusai Electric
- `6526.T` Socionext
- `6723.T` Renesas

The refresh tool must append prices for these names even if they were not in the
previous active DB universe. If any configured core symbol lacks the latest
trading date, the overlay must return `coreThemeDataFresh=false`.

## Theme Metadata

Ticker metadata must not rely only on yfinance industry strings for core theme
names. Add deterministic symbol overrides:

- `285A.T`: `memory`, `semi`, `ai`
- `3436.T`: `memory`, `semi`
- `6525.T`, `6526.T`, `6723.T`, `8035.T`, `6857.T`, `6146.T`, `6920.T`, `7735.T`,
  `4063.T`: `semi`, and `ai` where supply-chain exposure is appropriate.

Overrides should merge with fetched metadata, not delete fetched names or sectors.

## Rotation Overlay

Create a focused module, tentatively
`src/hot_theme_rotator/candidate_engine/theme_rotation.py`, with pure functions
that accept candidate dictionaries, theme heat, and a price snapshot map.

Output fields:

- `themeRegime`: `memory_hot`, `semi_hot`, `memory_semi_hot`, or `neutral`
- `coreThemeDataFresh`: boolean
- `leaderSymbol`: reference leader, usually `285A.T` for memory
- `leaderExtended`: boolean
- `chaseRisk`: `none`, `watch`, or `study_only`
- `secondLineCandidate`: boolean
- `rotationScore`: small signed numeric overlay
- `rotationReasons`: short machine-readable reason codes

Initial leader-extension logic:

- `ret20 >= 0.25` or `ret60 >= 0.50`
- close within 3% of the 20-session or 52-week high
- volume expansion or high turnover confirms crowding

Initial second-line logic:

- latest price data is fresh
- symbol belongs to memory/semi
- not `leaderExtended`
- at least two supporting facts are present:
  relative strength versus Kioxia or TOPIX, volume expansion, company catalyst,
  reasonable valuation/fundamental read, improving 5d/20d trend, or proximity to
  highs without climax movement.

## Rerank Integration

Keep the existing order:

`screener -> quality-gated catalyst rerank -> theme rotation overlay -> theme leader annotation`

The overlay may:

- subtract a penalty from extended leaders that lack fresh company evidence
- add a small boost to second-line candidates with multiple facts
- add a stale-core-theme penalty when the memory/semi basket is incomplete

The overlay may not:

- replace displayed score
- create news badges
- designate sector-only names as company catalysts
- output win-rate, expected return, or probability

## Dashboard/API Contract

Candidate payloads should expose the overlay fields for inspection. The UI may
show the labels as explanatory chips, but must keep Rule 9.4 language:
`uncalibrated research ordering`, not a forecast.

The dashboard meta should expose:

- `meta.dataQuality.coreThemeCoverage.memorySemiFresh`
- `meta.dataQuality.coreThemeCoverage.missingSymbols`
- `meta.dataQuality.coreThemeCoverage.staleSymbols`

## Testing

Use test-driven implementation:

- Unit tests for priority universe refresh including symbols absent from the old
  active universe.
- Unit tests for symbol theme overrides.
- Unit tests for leader-extension classification.
- Unit tests for second-line classification requiring at least two facts.
- Unit tests for rerank behavior: extended leader is penalized; supported second
  line gets a small boost; stale core theme coverage downgrades confidence.
- API serializer tests that the new fields are present and not rendered as
  probabilities.

## Out of Scope

- No calibration unlock.
- No broker/order path.
- No hard rewrite of the sibling `Project_optimized/screener.py` weights.
- No claim that memory/semi has positive forward edge until forward data proves it.

## Governance

This design is governed by Rule 11.14. Any future increase in overlay weight must
go through Rule 4 and a walk-forward validation path with costs and stale-data
guards.
